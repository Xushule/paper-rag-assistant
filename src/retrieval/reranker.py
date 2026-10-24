# -*- coding: utf-8 -*-
"""Rerank 精排：对混合检索候选集重打分、取 top-k。

为什么 Rerank（面试常问）：
- 混合检索（向量 + BM25 + RRF 融合）产出的是"候选集"，排序依据是融合排名，
  不是"与问题真实相关度"的精确度量——粗排阶段只要保证"好文档别漏掉"；
- 精排用更强的信号重新排序：业界两种主流做法——
  * cross-encoder（bge-reranker-v2-m3 等）：query 与每篇文档拼接后联合编码打分；
  * LLM rerank：让大模型逐条判断候选与问题的相关性（paper-qa 官方方案）。
- 标准流水线：粗排（向量/BM25 召回 top20~50）→ 精排（取 top5~10）。

本实现：
- RERANK_MODE=llm（默认）：走已配置的 LLM（DeepSeek）对候选打分，零额外依赖；
- RERANK_MODE=ollama：预留 cross-encoder 分支。注意：Ollama 官方至今
  （0.34.x）没有 /api/rerank 端点（社区验证 RAGFlow/Dify 均无法接入），
  模型虽可 pull 下来但端点 404，此分支当前不可用；
  若未来 Ollama 支持或改用 llama.cpp server，可切回此分支。
"""
import json
import re

from openai import OpenAI

from ..config import (
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL,
    OLLAMA_BASE_URL, RERANK_MODEL, RERANK_MODE,
)

# 每条候选送入 LLM 的最大字符数（20 条 × 300 字 ≈ 6K tokens，DeepSeek 轻松容纳）
MAX_CHARS = 300

RERANK_PROMPT = """你是检索精排器。判断每个候选片段与用户问题的相关性，只输出 JSON 数组，不要任何解释：
[{{"idx": 0, "score": 0-10}}, ...]
评分标准：
- 直接给出问题的关键事实/答案：9-10
- 提供部分相关背景或细节：6-8
- 仅弱相关/泛泛提及：3-5
- 与问题无关：0-2

用户问题: {query}

候选片段:
{items}
"""


def _call_llm(query: str, documents: list[str]) -> list[tuple[int, float]]:
    """LLM 打分：一次请求对所有候选输出 JSON 分数。"""
    items = "\n".join(f"[{i}] {doc[:MAX_CHARS]}" for i, doc in enumerate(documents))
    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL,
                    timeout=60, max_retries=1)
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": RERANK_PROMPT.format(query=query, items=items)}],
        temperature=0,
    )
    content = resp.choices[0].message.content or ""

    # 容错解析：模型可能包 ```json 代码块，只取第一个 [ 到最后一个 ]
    content = re.sub(r"```(?:json)?", "", content).strip()
    start, end = content.find("["), content.rfind("]")
    if start == -1 or end <= start:
        raise ValueError(f"LLM rerank 输出无法解析: {content[:120]!r}")
    data = json.loads(content[start:end + 1])

    scores: dict[int, float] = {}
    for item in data:
        idx = int(item.get("idx", -1))
        score = float(item.get("score", 0))
        if 0 <= idx < len(documents):
            scores[idx] = score
    if not scores:
        raise ValueError("LLM rerank 未返回有效分数")
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def _call_ollama(query: str, documents: list[str], top_k: int) -> list[tuple[int, float]]:
    """cross-encoder 分支：走 Ollama /api/rerank（官方端点未发布，当前不可用）。"""
    import requests
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/rerank",
        json={"model": RERANK_MODEL, "query": query, "documents": documents},
        timeout=120,
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])
    return [(r["index"], r["relevance_score"]) for r in results[:top_k]]


def rerank(query: str, documents: list[str], top_k: int = 5) -> list[tuple[int, float]]:
    """对候选文档按相关性精排。

    Args:
        query: 用户问题（精排的评分依据）。
        documents: 候选文档文本列表（来自混合检索粗排结果）。
        top_k: 返回最相关的多少个。

    Returns:
        [(原列表下标, relevance_score), ...]，按分数降序，最多 top_k 个。
    """
    if not documents:
        return []
    if RERANK_MODE == "ollama":
        return _call_ollama(query, documents, top_k)
    ranked = _call_llm(query, documents)
    return ranked[:top_k]
