# -*- coding: utf-8 -*-
"""ragas 评测：对 Golden Set 逐条跑 QASystem 回答，再算 RAG 质量指标。

用法：
    python eval/run_eval.py                 # 全量评测
    python eval/run_eval.py --limit 3       # 只跑前 3 条（试管线）
    python eval/run_eval.py --start 3 --limit 3   # 从第 3 条续跑（断点续跑）
    python eval/run_eval.py --skip-ask      # 跳过问答阶段，直接用已有 samples 算指标

流程：
    1. ask 阶段：逐条调用 QASystem（真实链路），保存 answer/citations -> eval/results/samples.json
    2. judge 阶段：DeepSeek 判分（回答是否覆盖参考答案要点）-> 答案准确率
    3. ragas 阶段：faithfulness / answer_relevancy / context_precision / LLMContextRecall
       -> eval/results/eval_summary.json + 控制台汇总表

设计说明（为什么这样写）：
- 三阶段解耦：问答最贵（多次 LLM 调用），先把结果落盘，指标阶段可反复重跑；
- --start 断点续跑：中途 API 失败不用从头再来；
- --skip-ask：改指标配置后只重算指标。
"""
import argparse
import io
import json
import os
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openai import OpenAI

from src.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from src.qa.answer import QASystem
from src.retrieval.bm25_index import load_index
from src.retrieval.embed_store import VectorStore

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"
GOLDEN_SET = EVAL_DIR / "golden_set.json"
SAMPLES_PATH = RESULTS_DIR / "samples.json"
SUMMARY_PATH = RESULTS_DIR / "eval_summary.json"
DETAIL_PATH = RESULTS_DIR / "eval_results.json"
RAGAS_SCORES_PATH = RESULTS_DIR / "ragas_scores.json"
RUN_LOG = RESULTS_DIR / "run_log.txt"

JUDGE_PROMPT = """你是答案评判员。请判断【系统回答】是否准确回答了【用户问题】并覆盖了【参考答案】的要点。

规则：
1. 参考答案是依据论文原文整理的要点，不是逐字标准答案；回答覆盖核心要点即可判对。
2. 回答中出现明确错误事实（与参考答案矛盾）判错；回答"未找到"但参考答案有明确要点时判错。
3. 部分覆盖时判错，并在 missing 中说明遗漏的要点。
4. 口径容错：若系统回答给出的数字/表述与参考答案口径不同，但该口径在论文原文中有明确依据
   （如同一数据在不同章节/不同论文中存在不同口径，如挑战赛发布口径 vs 数据集整体构成口径），
   应视为覆盖了要点判对；仅当系统回答与参考答案在同一口径下存在事实矛盾时才判错。

只输出 JSON，不要任何解释：
{{"correct": true或false, "missing": "遗漏或错误要点（正确时为空字符串）"}}

用户问题：{question}

参考答案：{reference}

系统回答：{answer}

系统回答依据的检索片段（供核实口径/依据，未截断）：
{contexts}
"""


def load_golden_set(limit: int | None, start: int) -> list[dict]:
    data = json.loads(GOLDEN_SET.read_text(encoding="utf-8"))
    if start:
        data = data[start:]
    if limit:
        data = data[:limit]
    return data


def build_qa() -> QASystem:
    store = VectorStore()
    bm25, chunks = load_index()
    return QASystem(store, bm25, chunks)


def ask_one(qa: QASystem, item: dict) -> dict:
    """跑真实问答链路，返回评测所需的样本字段。stdout 重定向到日志，避免刷屏。

    retrieved_contexts 存"完整块内容"而非 snippet：ragas 的 faithfulness 用该
    字段判"回答是否有上下文支持"，模型生成时看到的正是完整块（约 1800 字符），
    snippet 截断会系统性压低 faithfulness（0.394/0.4586 均为该口径）。
    通过 citations 的 chunk_id 反查 qa.chunks 拿完整内容；查不到才退 snippet。
    """
    question = item["question"]
    scope = item.get("scope")
    start = time.time()
    with open(RUN_LOG, "a", encoding="utf-8") as log, redirect_stdout(log):
        result = qa.ask(question, scope=scope)
    elapsed = time.time() - start
    chunk_map = {c.chunk_id: c for c in qa.chunks}
    contexts = []
    for c in result.get("citations", []):
        ch = chunk_map.get(c.get("chunk_id"))
        ctx = ch.content if ch is not None else c.get("snippet", "")
        # 元问题通道（_corpus_meta_answer）的 citation 带 filename（含 arXiv 编号）：
        # 答案断言常引用文件名（"AGAN...1711.10098"），而标题页正文不含文件名，
        # 不拼进去 ragas faithfulness 会把这类断言判为无上下文支持。
        fn = c.get("filename")
        if fn:
            ctx = f"（库内论文文件名: {fn}）\n" + ctx
        contexts.append(ctx)
    return {
        "id": item["id"],
        "category": item.get("category", ""),
        "scope": scope,
        "question": question,
        "reference": item["reference"],
        "answer": result["answer"],
        "retrieved_contexts": contexts,
        "n_citations": len(contexts),
        "verification": result.get("verification"),
        "revised": result.get("revised", False),
        "elapsed_s": round(elapsed, 1),
    }


def judge_correctness(client: OpenAI, sample: dict) -> dict:
    """LLM 判分：回答是否覆盖参考答案要点（对应需求文档"答案准确率"）。

    传入检索片段供 judge 核实"口径是否有论文依据"（规则 4 口径容错的前提：
    不传片段时 judge 无法判断系统回答的不同口径是否真实存在于论文）。
    """
    contexts = sample.get("retrieved_contexts", [])
    ctx_block = "\n\n".join(f"[{i}] {c[:600]}" for i, c in enumerate(contexts, 1)) or "(无检索片段)"
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            question=sample["question"], reference=sample["reference"],
            answer=sample["answer"], contexts=ctx_block)}],
        temperature=0,
        max_tokens=200,
        timeout=120,  # 显式超时：防止 API 挂起时进程无限等待（no_expand 卡死根因）
    )
    text = resp.choices[0].message.content or ""
    import re as _re
    text = _re.sub(r"```(?:json)?", "", text).strip()
    start_i, end_i = text.find("{"), text.rfind("}")
    if start_i == -1 or end_i <= start_i:
        return {"correct": None, "missing": f"judge 输出无法解析: {text[:80]!r}"}
    try:
        data = json.loads(text[start_i:end_i + 1])
        return {"correct": bool(data.get("correct", False)),
                "missing": str(data.get("missing", ""))}
    except Exception as e:
        return {"correct": None, "missing": f"judge 解析失败: {e}"}


def run_ask(golden: list[dict], start: int) -> list[dict]:
    qa = build_qa()
    # 断点续跑按"id 覆盖合并"：保留旧 samples 里本次不重跑的条目，
    # 只替换本次 golden 涉及的 id（旧实现 samples[:start] 会截断全量文件丢题）
    samples = []
    if SAMPLES_PATH.exists():
        samples = json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))
    this_ids = {item["id"] for item in golden}
    samples = [s for s in samples if s["id"] not in this_ids]
    failed: list[str] = []
    for i, item in enumerate(golden):
        sid = item["id"]
        print(f"[{start + i + 1}/{start + len(golden)}] {sid} {item['question'][:30]}...")
        try:
            sample = ask_one(qa, item)
        except Exception as e:
            # 单样本容错：LLM 网络/超时异常只跳过该样本，不中断整组。
            # 失败样本不写入 samples，本轮汇总的有效样本数会少于 golden 总数，
            # 可用 --skip-ask 重跑补齐（断点按 id 覆盖合并）。
            print(f"[ask 失败] {sid}: {type(e).__name__}: {str(e)[:150]}")
            failed.append(sid)
            continue
        samples.append(sample)
        SAMPLES_PATH.write_text(
            json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    if failed:
        print(f"[ask 完成] {len(samples)} 条已落盘，{len(failed)} 条失败待补: {failed}")
    return samples


def run_judge(samples: list[dict]) -> list[dict]:
    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL,
                    timeout=120, max_retries=1)
    for i, s in enumerate(samples):
        if s.get("judge"):
            continue
        print(f"[judge {i + 1}/{len(samples)}] {s['id']}")
        s["judge"] = judge_correctness(client, s)
        time.sleep(0.3)
    return samples


def run_ragas(samples: list[dict]) -> dict:
    """用 ragas 算 RAG 质量指标。

    LLM：DeepSeek（OpenAI 兼容端点 + instructor 结构化输出，instructor 要求模型
    支持 function calling，deepseek-chat 支持）。
    Embedding：本地 Ollama bge-m3（OpenAI 兼容 /v1/embeddings 端点，避免额外下载）。

    说明（为什么用旧式指标）：ragas 0.4.3 的 evaluate() 只接受继承
    ragas.metrics.base.Metric 的指标对象，而 metrics.collections 新架构指标
    继承 SimpleBaseMetric，会直接触发 "All metrics must be initialised..." 报错；
    旧式指标（ragas.metrics 下的单例）兼容 evaluate，且 PydanticPrompt 同时
    支持 InstructorBaseRagasLLM 与 langchain 两种 LLM 接口，因此选用旧式指标
    + llm_factory（instructor）组合。
    """
    from openai import OpenAI as OpenAIClient

    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.llms import llm_factory
    from ragas.metrics import (answer_relevancy, context_precision,
                               context_recall, faithfulness)

    class OllamaEmbedWrapper:
        """langchain 风格 embeddings（embed_query/embed_documents），走本地
        Ollama bge-m3 的 OpenAI 兼容 /v1/embeddings 端点。
        旧式 answer_relevancy 指标内部调用这两个方法。"""

        def __init__(self, base_url="http://localhost:11434/v1", model="bge-m3"):
            self._client = OpenAIClient(api_key="ollama", base_url=base_url)
            self._model = model

        def embed_query(self, text: str) -> list[float]:
            resp = self._client.embeddings.create(input=text, model=self._model)
            return resp.data[0].embedding

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [self.embed_query(t) for t in texts]

    # LLM 判分走 DeepSeek。max_tokens 默认 1024，拆 statements 的 JSON 输出
    # 一长就被截断（IncompleteOutputException → 该样本 faithfulness=nan），
    # 调大到 4096 从源头解决截断；strictness 降到 1 减少一次请求 3 份的输出负担。
    client = OpenAIClient(api_key=LLM_API_KEY, base_url=LLM_BASE_URL,
                          timeout=120, max_retries=1)
    llm = llm_factory(LLM_MODEL, client=client, max_tokens=4096)
    embeddings = OllamaEmbedWrapper()

    # 旧式指标注入 llm / embeddings
    faithfulness.llm = llm
    answer_relevancy.llm = llm
    answer_relevancy.embeddings = embeddings
    answer_relevancy.strictness = 1
    context_precision.llm = llm
    context_recall.llm = llm
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    ds_samples = []
    skipped = []
    for s in samples:
        ctx = [c for c in s.get("retrieved_contexts", []) if c.strip()]
        if not ctx:
            skipped.append(s["id"])
            continue
        ds_samples.append(SingleTurnSample(
            user_input=s["question"],
            response=s["answer"],
            retrieved_contexts=ctx,
            reference=s["reference"],
        ))
    dataset = EvaluationDataset(samples=ds_samples)
    result = evaluate(dataset=dataset, metrics=metrics)

    # 逐样本分值落盘（供诊断：哪些样本 nan、异常来源）
    scores_path = RAGAS_SCORES_PATH
    scores_path.write_text(
        json.dumps(result.scores, ensure_ascii=False, indent=2), encoding="utf-8")

    # ragas 0.4.x 结果 scores 是逐样本 list[dict]；按指标名聚合取均值。
    # 关键：DeepSeek 的 instructor 结构化输出偶发截断（IncompleteOutputException），
    # 该样本指标值为 nan；nan 不是 None，若不过滤会把整个均值拉成 nan。
    # 过滤 nan 并统计每指标有效样本数，mean 只基于有效样本（有效数过少时需说明）。
    import math
    import statistics as _stats
    agg: dict[str, list[float]] = {}
    for row in result.scores:
        for name, val in row.items():
            key = str(name).replace("_with_reference", "").replace("_without_reference", "")
            if val is not None and not (isinstance(val, float) and math.isnan(val)):
                agg.setdefault(key, []).append(float(val))
    score_map = {k: round(_stats.mean(v), 4) for k, v in agg.items()}
    effective_n = {k: len(v) for k, v in agg.items()}
    return {"scores": score_map, "skipped_no_context": skipped, "effective_n": effective_n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--skip-ask", action="store_true")
    ap.add_argument("--skip-ragas", action="store_true",
                    help="跳过 ragas（冒烟/局部验证时用，避免触发全量 5 小时指标计算）")
    ap.add_argument("--tag", type=str, default="",
                    help="消融/多组评测标记：输出写到 samples_{tag}.json 等，避免覆盖")
    args = ap.parse_args()

    # 按 tag 隔离输出文件（消融评测四组配置各跑一遍时不互相覆盖）
    if args.tag:
        global SAMPLES_PATH, SUMMARY_PATH, DETAIL_PATH, RAGAS_SCORES_PATH, RUN_LOG
        SAMPLES_PATH = RESULTS_DIR / f"samples_{args.tag}.json"
        SUMMARY_PATH = RESULTS_DIR / f"eval_summary_{args.tag}.json"
        DETAIL_PATH = RESULTS_DIR / f"eval_results_{args.tag}.json"
        RAGAS_SCORES_PATH = RESULTS_DIR / f"ragas_scores_{args.tag}.json"
        RUN_LOG = RESULTS_DIR / f"run_log_{args.tag}.txt"

    RESULTS_DIR.mkdir(exist_ok=True)
    golden = load_golden_set(args.limit, args.start)
    if not golden:
        print("Golden Set 为空，检查 eval/golden_set.json")
        return

    if args.skip_ask:
        samples = json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))
    else:
        samples = run_ask(golden, args.start)

    # 阶段二：LLM 判分（答案准确率）
    samples = run_judge(samples)
    SAMPLES_PATH.write_text(
        json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")

    # 阶段三：ragas 指标
    if args.skip_ragas:
        ragas_result = {"scores": {}, "skipped_no_context": []}
    else:
        ragas_result = run_ragas(samples)

    # 汇总
    judged = [s for s in samples if s.get("judge", {}).get("correct") is not None]
    n_correct = sum(1 for s in judged if s["judge"]["correct"])
    accuracy = round(n_correct / len(judged), 4) if judged else None

    summary = {
        "n_samples": len(samples),
        "n_judged": len(judged),
        "answer_accuracy": accuracy,
        "answer_accuracy_pct": round(accuracy * 100, 1) if accuracy is not None else None,
        "faithfulness": ragas_result["scores"].get("faithfulness"),
        "answer_relevancy": ragas_result["scores"].get("answer_relevancy"),
        "context_precision": ragas_result["scores"].get("context_precision"),
        "context_recall": ragas_result["scores"].get("context_recall"),
        "skipped_no_context": ragas_result["skipped_no_context"],
        "effective_n": ragas_result.get("effective_n", {}),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # 逐条明细（含指标按 id 回填）
    detail = []
    for s in samples:
        row = {k: s[k] for k in ("id", "category", "scope", "question", "reference",
                                  "answer", "n_citations", "elapsed_s")}
        row["retrieved_contexts"] = s.get("retrieved_contexts", [])
        row["judge"] = s.get("judge")
        row["revised"] = s.get("revised", False)
        detail.append(row)

    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    DETAIL_PATH.write_text(
        json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")

    # 控制台输出
    print("\n" + "=" * 60)
    print("评测汇总（eval/results/eval_summary.json）")
    print("=" * 60)
    print(f"样本数        : {summary['n_samples']}")
    print(f"答案准确率    : {summary['answer_accuracy_pct']}%  ({n_correct}/{len(judged)})"
          f"   [需求验收 ≥80%]")
    print(f"faithfulness  : {summary['faithfulness']}  [1 - 幻觉率，需求幻觉率 ≤5%]")
    print(f"answer_relevancy: {summary['answer_relevancy']}")
    print(f"context_precision: {summary['context_precision']}")
    print(f"context_recall: {summary['context_recall']}")
    if summary["skipped_no_context"]:
        print(f"无上下文跳过的样本: {summary['skipped_no_context']}")

    # 按类别展示准确率
    from collections import Counter
    by_cat: dict[str, list] = {}
    for s in judged:
        by_cat.setdefault(s["category"], []).append(s["judge"]["correct"])
    print("\n按类别准确率：")
    for cat, lst in by_cat.items():
        print(f"  {cat}: {sum(lst)}/{len(lst)} = {round(sum(lst)/len(lst)*100,1)}%")


if __name__ == "__main__":
    main()
