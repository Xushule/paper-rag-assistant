# -*- coding: utf-8 -*-
"""LLM 问答：基于检索到的论文片段，调用 DeepSeek 生成带引用的回答。

调用链：
    用户问题 -> 混合检索 top-k 片段 -> 组装 prompt -> DeepSeek 生成
    -> 回答 + 引用列表（每条引用含来源论文/章节/页码）
    -> 引用校验（posthoc attribution）
    -> 校验反馈再检索（self-correction）：存在不支持/部分支持引用时，
       用 LLM 把"缺失断言"变成补充查询再检索，合并证据后重新生成并再校验。

引用溯源设计（面试亮点）：
- 检索片段按 [1]..[k] 编号放进 prompt；
- 要求模型在回答中只引用检索片段里的信息，并标注 [编号]；
- 接口同时返回 citations 元数据，前端可点击跳转对应论文/章节。
"""
import json
import re

from openai import OpenAI

from ..config import (LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, QUERY_EXPAND,
                      MMR_ENABLED, MMR_TOP_K, MMR_LAMBDA)

SYSTEM_PROMPT = """你是一位严谨的论文阅读助手。请根据【检索片段】回答用户问题，规则如下：
1. 只依据片段中的信息回答；片段里没有的内容，明确说"检索片段中未找到"。
2. 回答中用 [1] [2] 标注信息来自哪个片段（对应片段编号）。
3. 用中文回答，但保留论文中的专有名词（模型名、数据集名、指标名）的英文原文。
4. 回答控制在 3-6 句话，先给结论，再给依据。不要编造片段中没有的数字或事实。
5. 【对话历史】供参考：如果用户问题是"它""那""这个方法"等省略指代，结合历史判断指代对象。
   但历史只是参考，最终仍必须以本次【检索片段】为准。"""

# 语料库元问题：问的是"库里有什么论文"（几篇/哪些/有没有 XX 论文），
# 而不是某篇论文的内容。命中则不走语义检索——论文正文里永远不会写
# "我这里收录了几篇论文"，语义检索对这种问题只能捞到参考文献区等沾边碎片，
# 直接按 chunk.source（论文 id）分组列清单才是准确答案（M2.12）。
META_QUESTION_RE = re.compile(
    r"(几篇论文|哪些论文|有哪些论文|论文列表|语料库|收录|库里|库中|"
    r"一共.{0,6}篇|总共.{0,6}篇|有没有.{0,16}论文)"
)

# 跨论文对比意图：命中则走"对比专用链路"（逐篇子查询检索 + 每篇至少 1 块进最终）。
# 原因（M2.13 实测）：问题点名某篇论文时，检索 / MMR 相关度 / LLM 精排每一环
# 都偏向被点名方（如"AGAN 和 NightRain 有什么不同"→ NightRain 块 RRF 分数高 3 倍，
# AGAN 0 块进最终，回答直接说"未找到 AGAN"）。对比链路对每篇用"剥离他篇名"的
# 子查询 + scope 单独检索，从检索侧保证公平。
COMPARE_RE = re.compile(
    r"(对比|区别|差异|不同|相比|比较|哪个好|谁更好|优势|劣势|优缺点|"
    r"共同点|差别|差距|versus|分别|各有什么|有何不同|有什么区别|怎么比)"
)
# 指代全部论文的表述："这四个论文 / 所有论文 / 这些论文 / 每一篇"。
# 对比问题若未点名具体论文但指代全库，按"全部论文逐篇对比"处理（M2.14 补漏：
# "对比一下这四个论文有什么不同" 因无点名词退化普通检索，候选被 RC 垄断，
# LLM 把 RC 里的外部基线 IDT/RDiffusion 误当论文回答，漏掉 NTIRE2025/NightRain）。
ALL_PAPERS_RE = re.compile(
    r"(全部|所有|这几个|这几篇|这些论文|四个|四篇|四个论文|各篇|每一篇|所有论文)"
)


class QASystem:
    """论文问答系统：检索 + 生成 + 校验反馈再检索（支持论文作用域与多轮记忆）。

    - 论文作用域：ask(question, scope="xxx.pdf") 时只在该论文的块内检索，
      防止多篇论文串台；scope=None 表示全库检索。
    - 多轮记忆：内部维护最近 N 轮历史，follow-up 问题（"那损失函数呢？"）
      会结合历史理解指代；prompt 中附历史供模型参考。
    - 校验反馈再检索（self-correction）：回答生成后先做 posthoc attribution，
      若存在 unsupported/partial 引用，用 LLM 构造补充查询再检索，
      合并新旧证据重新生成并再校验（OpenScholar 同款思路，最多补 1 轮）。
    """

    HISTORY_MAX = 4  # 最多保留几轮历史
    REVISE_MAX_ROUNDS = 1  # self-correction 最多补几轮检索

    def __init__(self, store, bm25, chunks):
        self.store = store
        self.bm25 = bm25
        self.chunks = chunks
        if not LLM_API_KEY:
            raise RuntimeError("未配置 LLM_API_KEY（.env），无法调用 LLM")
        self.client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
        self.history: list[dict] = []

    def reset_history(self) -> None:
        """清空对话历史。"""
        self.history = []

    def papers(self) -> list[str]:
        """库内已有的论文（去重、按文件名排序），用于作用域选择。"""
        return sorted({c.source for c in self.chunks})

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    def _llm_queries(self, prompt: str, max_tokens: int = 200) -> list[str]:
        """调用 LLM 生成检索查询（JSON 数组，容错解析；失败返回空列表）。"""
        try:
            resp = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "你是检索查询生成器，只输出 JSON。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=max_tokens,
            )
            text = resp.choices[0].message.content.strip()
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
            data = json.loads(text)
            return [str(q).strip() for q in data if str(q).strip()][:3]
        except Exception as e:
            print(f"[query] 生成检索查询失败: {e}")
            return []

    def _expand_queries(self, question: str) -> list[str]:
        """LLM 查询扩展：把问题扩成 2 个补充检索查询（英文优先、不同角度）。"""
        if not QUERY_EXPAND:
            return []
        prompt = (
            "你在帮助论文检索。用户问题：\n" + question +
            "\n\n请生成 2 个独立的论文检索查询，用于从学术论文（如去雨/去雨滴方向）中检索证据。"
            "要求：\n1. 英文优先，可夹带中文；\n2. 从不同角度覆盖问题（专有名词、方法名、数据集名、"
            "术语同义表述等）；\n3. 直接是检索可用的短语，不要问句。"
            "只输出 JSON 数组，如 [\"query1\", \"query2\"]，不要任何其他内容。"
        )
        qs = self._llm_queries(prompt)
        if qs:
            print(f"[query-expand] 补充查询: {qs}")
        return qs

    def _retrieve(self, question: str, top_k: int, scope: str | None,
                  vector_k: int = 12, bm25_k: int = 12,
                  final_k: int = 12, rerank_top: int = 5,
                  use_history: bool = True, expand: bool = True) -> list:
        """多路检索 + 跨路 RRF 融合 + LLM 精排。

        主查询 + LLM 扩展查询各走一路混合检索（向量 + BM25），
        对每路的候选排名做 RRF 融合（解决不同查询分数不可比的问题），
        再交给 LLM 精排取 top_k。

        Args:
            expand: 是否做 LLM 查询扩展（补充检索时传 False，查询已足够具体）。
        """
        from .query_rewriter import expand_query
        from ..retrieval.hybrid import hybrid_search

        # 1) 构建检索路：主查询 + 扩展查询
        queries = [question]
        if expand:
            queries += self._expand_queries(question)

        # 2) 每路混合检索，记录 (chunk, 该路内排名)
        all_hits = []
        for i, q in enumerate(queries):
            bm25_query = expand_query(q)
            vector_query = q
            if use_history and i == 0 and self.history:
                # 拼接最近一轮问题：follow-up 常省略主语/对象，两问并排编码
                # 能让向量检索同时命中上一轮和本轮的相关片段
                vector_query = f"{self.history[-1]['question']} {q}"
            hits = hybrid_search(
                self.store, self.bm25, self.chunks,
                query=vector_query, bm25_query=bm25_query,
                vector_k=vector_k, bm25_k=bm25_k, final_k=final_k, scope=scope,
            )
            all_hits.append(hits)

        # 3) 跨路 RRF 融合：score = Σ 1/(60 + 该路排名)，跨路分数归一可比
        from collections import defaultdict
        score_map: dict[str, float] = defaultdict(float)
        for hits in all_hits:
            for rank, (chunk, _score) in enumerate(hits, start=1):
                score_map[chunk.chunk_id] += 1.0 / (60 + rank)
        chunk_map = {c.chunk_id: c for c in self.chunks}
        ordered = sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)
        hits = [(chunk_map[cid], s) for cid, s in ordered if cid in chunk_map][:final_k]

        # 3.5) 候选池覆盖增强 + MMR 多样性选片段（仅全库模式）：
        #     MMR 只能在已有候选里"分散"，候选池被某篇垄断时无法创造缺失来源
        #     （实测"AGAN 和 NightRain 有什么不同"：top12 里 NightRain 占 9 块，
        #      MMR 后 AGAN 仍 0 块）。因此先给每篇论文单独 scope 检索 top2 补进
        #     候选池（保证每篇都有代表），再由 MMR 从池里选多样化的 top_k，
        #     最后 LLM 精排。= 按论文配额保证覆盖（recall）+ MMR 保证不重复（diversity）。
        if scope is None and MMR_ENABLED:
            try:
                from ..retrieval.mmr import mmr_select
                from ..retrieval.hybrid import hybrid_search
                from .query_rewriter import expand_query

                # 1) 每篇论文单独检索 top2（用主查询，各自语境下最相关）
                quota = []
                for src in sorted({c.source for c in self.chunks}):
                    ph = hybrid_search(
                        self.store, self.bm25, self.chunks,
                        query=question, bm25_query=expand_query(question),
                        vector_k=4, bm25_k=4, final_k=4, scope=src,
                    )
                    quota.extend(ph[:2])

                # 2) 合并去重：全局 top12 + 每篇配额
                seen = {c.chunk_id for c, _ in hits}
                for chunk, _s in quota:
                    if chunk.chunk_id not in seen:
                        seen.add(chunk.chunk_id)
                        hits.append((chunk, 0.0))

                # 3) MMR 从池里选多样化的 MMR_TOP_K 个
                ids = [c.chunk_id for c, _ in hits]
                got = self.store.collection.get(ids=ids, include=["embeddings"])
                emb_map = dict(zip(got["ids"], got["embeddings"]))
                embs = [emb_map[i] for i in ids]
                rels = [s for _, s in hits]
                sel = mmr_select(ids, embs, rels, k=MMR_TOP_K, lambda_=MMR_LAMBDA)
                by_id = {c.chunk_id: (c, s) for c, s in hits}
                hits = [by_id[i] for i in sel]
                print(f"[mmr] 候选池 {len(ids)}（含逐篇配额）-> 多样化选 {len(sel)}")
            except Exception as e:
                print(f"[mmr] 失败，回退 RRF 候选: {e}")

        # 4) LLM 精排：对融合后的候选打分，取 top_k
        if hits:
            from ..retrieval.reranker import rerank
            try:
                docs = [c.content for c, _ in hits]
                ranked = rerank(question, docs, top_k=rerank_top)
                hits = [hits[idx] for idx, _ in ranked]
            except Exception as e:
                print(f"[rerank] 精排失败，回退粗排结果: {e}")
                hits = hits[:rerank_top]
        return hits[:top_k]

    # ------------------------------------------------------------------
    # 生成 + 校验
    # ------------------------------------------------------------------
    def _generate(self, question: str, hits: list,
                  include_history: bool = True) -> tuple:
        """组装片段 -> 生成回答 -> 引用校验。

        Returns:
            (answer, citations, verification)
            citations 与 [n] 一一对应（含 snippet / 章节 / 页码 / verdict）。
        """
        from .attribution import verify_citations

        context_lines = []
        citations = []
        for i, (chunk, _score) in enumerate(hits, start=1):
            context_lines.append(
                f"[{i}] (来源: {chunk.source}, 章节: {chunk.section}, 第{chunk.page}页)\n{chunk.content}"
            )
            # 表格块（Markdown 表格）的 snippet 给全文，否则校验器看不到
            # 截断点之后的关键数据行（如 Ours 31.57），会把"支持"误判成"不支持"
            is_table = "|" in chunk.content and chunk.content.count("|") >= 4
            citations.append({
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "section": chunk.section,
                "page": chunk.page,
                "snippet": chunk.content if is_table else chunk.content[:200],
            })

        prompt_parts = ["【检索片段】\n" + "\n\n".join(context_lines)]
        if include_history and self.history:
            hist_lines = [
                f"用户: {h['question']}\n助手: {h['answer'][:150]}"
                for h in self.history[-3:]
            ]
            prompt_parts.append("【对话历史】\n" + "\n\n".join(hist_lines))
        prompt_parts.append(f"【用户问题】\n{question}")
        user_prompt = "\n\n".join(prompt_parts)

        resp = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=800,
        )
        answer = resp.choices[0].message.content.strip()

        # 引用校验（posthoc attribution）：回答生成后逐条核对
        verification = None
        try:
            verification = verify_citations(question, answer, citations)
            if verification:
                verdict_map = {v["cite"]: v for v in verification}
                for i, c in enumerate(citations, start=1):
                    if i in verdict_map:
                        c["verdict"] = verdict_map[i]["verdict"]
        except Exception as e:
            print(f"[verify] 引用校验失败，跳过: {e}")

        return answer, citations, verification

    # ------------------------------------------------------------------
    # self-correction：校验反馈再检索
    # ------------------------------------------------------------------
    def _build_followup_queries(self, question: str, issues: list) -> list[str]:
        """根据校验反馈，让 LLM 生成 2 个补充检索查询。

        issues: verification 中 unsupported/partial 条目（含 cite/verdict/reason）。
        """
        lines = [
            f"[{it.get('cite')}] 判定={it.get('verdict')}: {it.get('reason', '')}"
            for it in issues
        ]
        prompt = (
            "你在辅助论文检索。下面的引用校验结果标记了回答中证据不足的断言：\n"
            + "\n".join(lines)
            + "\n\n请生成 2 个独立的论文检索查询（英文优先，可夹带专有名词），"
              "目标是找到能支撑上述断言的证据片段。"
              "只输出 JSON 数组，如 [\"query1\", \"query2\"]，不要任何其他内容。"
        )
        return self._llm_queries(prompt)

    def _supplement_retrieval(self, question: str, issues: list,
                              scope: str | None) -> list:
        """为每个补充查询做混合检索 + 精排，返回新片段（按 chunk_id 去重）。"""
        queries = self._build_followup_queries(question, issues)
        if not queries:
            return []
        seen = set()
        extra = []
        for q in queries:
            # 补充查询已由校验反馈生成、足够具体，不再二次扩展
            hits = self._retrieve(q, top_k=10, scope=scope,
                                  vector_k=10, bm25_k=10, final_k=10,
                                  rerank_top=3, use_history=False, expand=False)
            for chunk, _score in hits:
                if chunk.chunk_id not in seen:
                    seen.add(chunk.chunk_id)
                    extra.append((chunk, 0.0))
        return extra

    @staticmethod
    def _merge_hits(original: list, extra: list) -> list:
        """合并两轮片段：旧的在前面，新片段按 chunk_id 去重追加。"""
        seen = {c.chunk_id for c, _ in original}
        merged = list(original)
        for chunk, score in extra:
            if chunk.chunk_id not in seen:
                seen.add(chunk.chunk_id)
                merged.append((chunk, score))
        return merged[:10]  # 最多 10 个片段，防止 prompt 过长

    # ------------------------------------------------------------------
    # 语料库元问题：不检索，直接按论文 id 分组回答
    # ------------------------------------------------------------------
    def _corpus_meta_answer(self, question: str) -> tuple:
        """语料库元问题通道：跳过语义检索，按 chunk.source（论文 id）分组列清单。

        论文正文里不存在"我收录了几篇论文"这类文字，语义检索对此类问题
        只能捞到参考文献区等沾边碎片（实测 NightRain 0 块进候选）。
        直接查块元数据里的 source 字段分组，准确率 100%。

        Returns:
            (answer, citations, verification)
            引用指向每篇论文的代表块（优先标题页/摘要块）。
        """
        from .attribution import verify_citations

        # 1) 按论文 id（source）分组
        by_source: dict[str, list] = {}
        for c in self.chunks:
            by_source.setdefault(c.source, []).append(c)

        # 2) 每篇选代表块：优先标题页，其次摘要/引言块，否则第一个块
        def pick_rep(chunks):
            page1 = [c for c in chunks if c.page == 1]
            if page1:
                return page1[0]
            for kw in ("abstract", "introduction", "summary", "摘要", "引言"):
                hit = [c for c in chunks if kw in c.section.lower()]
                if hit:
                    return hit[0]
            return chunks[0]

        sources = sorted(by_source)
        reps = {s: pick_rep(by_source[s]) for s in sources}

        # 3) 组装"论文清单"prompt，LLM 只需基于清单回答（不编造清单外的论文）
        lines = []
        for i, s in enumerate(sources, start=1):
            rep = reps[s]
            lines.append(
                f"[{i}] 文件名: {s}（库内 {len(by_source[s])} 个检索块）\n"
                f"代表片段: {rep.content[:400]}"
            )
        prompt = (
            "以下是语料库中的全部论文清单（每篇附代表片段）。请回答用户关于"
            "论文库本身的问题（共几篇/有哪些/有没有某方向论文等），"
            "用 [编号] 引用对应论文。中文回答，先给结论。\n\n"
            + "\n\n".join(lines)
            + f"\n\n【用户问题】\n{question}"
        )
        resp = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system",
                 "content": "你是论文库助手，只根据给出的论文清单回答，不编造清单之外的论文。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=800,
        )
        answer = resp.choices[0].message.content.strip()

        # 4) 引用 + 校验（复用 posthoc attribution；表格块 snippet 给全文）
        citations = []
        for i, s in enumerate(sources, start=1):
            rep = reps[s]
            is_table = "|" in rep.content and rep.content.count("|") >= 4
            citations.append({
                "chunk_id": rep.chunk_id,
                "source": rep.source,
                "section": rep.section,
                "page": rep.page,
                "snippet": rep.content if is_table else rep.content[:200],
            })
        verification = None
        try:
            verification = verify_citations(question, answer, citations)
            if verification:
                verdict_map = {v["cite"]: v for v in verification}
                for i, c in enumerate(citations, start=1):
                    if i in verdict_map:
                        c["verdict"] = verdict_map[i]["verdict"]
        except Exception as e:
            print(f"[verify] 引用校验失败，跳过: {e}")

        return answer, citations, verification

    # ------------------------------------------------------------------
    # 对比专用链路：逐篇子查询检索 + 每篇至少 1 块进最终
    # ------------------------------------------------------------------
    def _named_papers(self, question: str) -> dict[str, list[str]]:
        """从问题里识别点名的库内论文。

        用每篇论文的"独特关键词"（该论文标题词里别篇没有的词 + arXiv id）
        在问题里做子串匹配，避免 "Raindrop" 这类 3 篇共有的词同时点名 3 篇。

        Returns:
            {source: [命中的关键词]}，未点名任何论文时返回 {}。
        """
        sources = sorted({c.source for c in self.chunks})

        def title_words(src: str) -> set[str]:
            stem = src.rsplit(".", 1)[0].replace("_", " ")
            words = set()
            for tok in stem.split():
                if any(ch.isalpha() for ch in tok) and len(tok) >= 3:
                    words.add(tok.lower())
                elif tok.replace(".", "").isdigit() and len(tok) >= 7:
                    words.add(tok)  # arXiv id，如 2407.16957
            return words

        words_by_src = {s: title_words(s) for s in sources}
        named: dict[str, list[str]] = {}
        ql = question.lower()
        for s in sources:
            # 独特词 = 该篇独有（别篇标题词里没有）
            others = set()
            for o, w in words_by_src.items():
                if o != s:
                    others |= w
            unique = words_by_src[s] - others
            hits = [w for w in unique
                    if (w.isdigit() and w in question) or (w in ql)]
            if hits:
                named[s] = hits
        return named

    def _retrieve_compare(self, question: str, top_k: int) -> list:
        """对比专用检索：识别点名论文 -> 每篇剥离他篇名子查询 + scope 检索
        -> 合并 -> LLM 精排 -> 每篇点名论文至少 1 块进最终。

        与普通检索的区别：
        - 每篇论文在"只有自己语境"下检索（问题里其他点名论文的名字被剥离），
          被点名方不会因为"名字出现次数多"而垄断候选；
        - 最终 top_k 强制保证每篇点名论文至少 1 块，对比公平性从检索到生成全程保证。
        """
        from .query_rewriter import expand_query
        from ..retrieval.hybrid import hybrid_search
        from ..retrieval.reranker import rerank

        named = self._named_papers(question)
        # 未点名具体论文但指代全库（"这四个论文/所有论文/这些"）：
        # 按全部论文逐篇对比，避免退化成普通检索、候选被单篇垄断
        if not named and ALL_PAPERS_RE.search(question):
            named = {s: [] for s in sorted({c.source for c in self.chunks})}
            print(f"[compare] 未点名具体论文，按全部 {len(named)} 篇逐篇对比")
        if not named:
            return self._retrieve(question, top_k=top_k, scope=None)

        # 1) 每篇单独检索 top3
        #    点名模式：子查询 = 问题去掉其他点名论文的名字；
        #    全部论文模式（kws 为空）：用固定的"论文简介"查询，
        #    每篇都能捞到摘要/方法/贡献块，避免原问题里的"四个论文"等指代噪音
        pool: list = []
        seen: set[str] = set()
        for src, kws in named.items():
            if kws:
                q_i = question
                for other_src, other_kws in named.items():
                    if other_src != src:
                        for kw in other_kws:
                            # 大小写不敏感剥离他篇名字（关键词是小写，原文可能大写）
                            q_i = re.sub(re.escape(kw), " ", q_i, flags=re.IGNORECASE)
            else:
                q_i = "这篇论文的研究内容、方法和主要贡献是什么"
            hits = hybrid_search(
                self.store, self.bm25, self.chunks,
                query=q_i, bm25_query=expand_query(q_i),
                vector_k=8, bm25_k=8, final_k=8, scope=src,
            )
            for chunk, _s in hits[:3]:
                if chunk.chunk_id not in seen:
                    seen.add(chunk.chunk_id)
                    pool.append((chunk, 0.0))

        # 2) 未点名论文/通用证据：补充一路全局检索
        if len(pool) < top_k * 2:
            hits = hybrid_search(
                self.store, self.bm25, self.chunks,
                query=question, bm25_query=expand_query(question),
                vector_k=8, bm25_k=8, final_k=8, scope=None,
            )
            for chunk, _s in hits:
                if chunk.chunk_id not in seen:
                    seen.add(chunk.chunk_id)
                    pool.append((chunk, 0.0))

        # 3) LLM 精排（多选一倍，给"每篇至少 1 块"配额留空间）
        try:
            docs = [c.content for c, _ in pool]
            ranked = rerank(question, docs, top_k=top_k * 2)
            final10 = [pool[idx] for idx, _ in ranked]
        except Exception as e:
            print(f"[compare] 精排失败，回退粗排: {e}")
            final10 = pool[:top_k * 2]

        # 4) 配额：确保每篇点名论文在 final10 里（缺的替换末尾低分块）
        final_ids = {c.chunk_id for c, _ in final10}
        for src in named:
            if not any(c.source == src for c, _ in final10):
                cand = [(c, s) for c, s in pool if c.source == src
                        and c.chunk_id not in final_ids]
                if cand:
                    final10[-1] = cand[0]
                    final_ids.add(cand[0][0].chunk_id)

        # 5) 组装最终 top_k：先每篇点名论文保底 1 块（按精排顺序），再补齐高分块
        picked: list = []
        seen_src: set[str] = set()
        for c, s in final10:
            if c.source in named and c.source not in seen_src:
                picked.append((c, s))
                seen_src.add(c.source)
        picked_ids = {c.chunk_id for c, _ in picked}
        for c, s in final10:
            if len(picked) >= top_k:
                break
            if c.chunk_id not in picked_ids:
                picked.append((c, s))
                picked_ids.add(c.chunk_id)
        final = picked
        print(f"[compare] 点名论文 {len(named)} 篇，候选池 {len(pool)} -> 精排 {len(final10)} -> 最终 {len(final)}")
        return final

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def ask(self, question: str, top_k: int = 5,
            scope: str | None = None) -> dict:
        """回答一个问题（含校验反馈再检索）。

        Args:
            question: 用户问题。
            top_k: 最终送入 prompt 的片段数。
            scope: 论文作用域（PDF 文件名，见 papers()）。
                为 None 时在全库检索。

        Returns:
            {
                "question": str,
                "answer": str,                       # 最终回答（含 [n] 引用）
                "citations": [...],                  # 与 [n] 对应的引用元数据
                "retrieved": 实际用到的片段数,
                "verification": [...] | None,        # 引用校验结果
                "revised": bool,                     # 是否发生过补充检索修正
            }
        """
        # 0) 语料库元问题（全库 + 问"库里有什么论文"）：
        #    不走语义检索，直接按论文 id（source）分组列清单，见 _corpus_meta_answer
        if scope is None and META_QUESTION_RE.search(question):
            print("[meta] 语料库元问题，跳过语义检索，按 source 分组回答")
            answer, citations, verification = self._corpus_meta_answer(question)
            self.history.append({"question": question, "answer": answer})
            if len(self.history) > self.HISTORY_MAX:
                self.history.pop(0)
            return {
                "question": question,
                "answer": answer,
                "citations": citations,
                "verification": verification,
                "retrieved": len(citations),
                "revised": False,
            }

        # 0.5) 跨论文对比问题（全库 + 对比意图）：逐篇子查询检索 + 每篇至少 1 块，
        #     解决点名对比时被点名方垄断候选（M2.13 实测 AGAN 0 块进最终）
        if scope is None and COMPARE_RE.search(question):
            hits = self._retrieve_compare(question, top_k=top_k)
        else:
            # 1) 混合检索 + 精排
            hits = self._retrieve(question, top_k=top_k, scope=scope)
        if not hits:
            return {
                "question": question,
                "answer": "检索片段为空：库里没有相关内容，请确认论文已入库或换一种问法。",
                "citations": [],
                "retrieved": 0,
                "verification": None,
                "revised": False,
            }

        # 2) 第一轮生成 + 校验
        answer, citations, verification = self._generate(question, hits)

        # 3) 校验反馈再检索（self-correction）：
        #    存在 unsupported/partial 引用 -> 构造补充查询 -> 合并证据 -> 重新生成再校验
        revised = False
        if verification:
            issues = [v for v in verification
                      if v.get("verdict") in ("unsupported", "partial")]
            if issues and self.REVISE_MAX_ROUNDS > 0:
                extra = self._supplement_retrieval(question, issues, scope)
                if extra:
                    merged = self._merge_hits(hits, extra)
                    if len(merged) > len(hits):
                        print(f"[self-correct] 补充检索到 {len(merged) - len(hits)} 个新片段，重新生成")
                        answer2, citations2, verification2 = \
                            self._generate(question, merged, include_history=False)
                        answer, citations, verification = answer2, citations2, verification2
                        revised = True

        # 4) 记录历史（供下一轮 follow-up 使用；记最终回答）
        self.history.append({"question": question, "answer": answer})
        if len(self.history) > self.HISTORY_MAX:
            self.history.pop(0)

        return {
            "question": question,
            "answer": answer,
            "citations": citations,
            "verification": verification,
            "retrieved": len(citations),
            "revised": revised,
        }
