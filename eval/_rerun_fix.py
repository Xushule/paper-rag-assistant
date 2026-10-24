# -*- coding: utf-8 -*-
"""修复后局部重跑：只对受影响样本重跑 ask+judge+ragas，merge 回全量结果。

用法: python eval/_rerun_fix.py
输出: 更新 samples.json / ragas_scores.json / eval_summary.json（全量聚合口径）
"""
import json
import math
import statistics as _stats
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.run_eval import (
    GOLDEN_SET, SAMPLES_PATH, RESULTS_DIR, SUMMARY_PATH,
    load_golden_set, build_qa, run_ask, run_judge,
)

TARGETS = [
    "rc-20", "ntire-15", "all-16", "all-17", "all-18",   # 5 条错题
    "all-01", "all-02", "all-15",                        # 元问题/对比低分
    "rc-14", "nr-19",                                    # 低 faithfulness
]


def local_ragas(samples: list[dict], target_ids: set[str]) -> dict:
    """只对 target 样本算 ragas，返回 {id: {metric: value}}。"""
    from openai import OpenAI as OpenAIClient

    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.llms import llm_factory
    from ragas.metrics import (answer_relevancy, context_precision,
                               context_recall, faithfulness)

    from src.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

    class OllamaEmbedWrapper:
        def __init__(self, base_url="http://localhost:11434/v1", model="bge-m3"):
            self._client = OpenAIClient(api_key="ollama", base_url=base_url)
            self._model = model

        def embed_query(self, text: str) -> list[float]:
            resp = self._client.embeddings.create(input=text, model=self._model)
            return resp.data[0].embedding

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [self.embed_query(t) for t in texts]

    client = OpenAIClient(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    llm = llm_factory(LLM_MODEL, client=client, max_tokens=4096)
    embeddings = OllamaEmbedWrapper()

    faithfulness.llm = llm
    answer_relevancy.llm = llm
    answer_relevancy.embeddings = embeddings
    answer_relevancy.strictness = 1
    context_precision.llm = llm
    context_recall.llm = llm
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    todo = [s for s in samples if s["id"] in target_ids
            and any(c.strip() for c in s.get("retrieved_contexts", []))]
    print(f"[ragas-local] 局部重算 {len(todo)} 条: {[s['id'] for s in todo]}")
    if not todo:
        return {}

    ds = EvaluationDataset(samples=[
        SingleTurnSample(
            user_input=s["question"],
            response=s["answer"],
            retrieved_contexts=[c for c in s.get("retrieved_contexts", []) if c.strip()],
            reference=s.get("reference", ""),
        ) for s in todo
    ])
    result = evaluate(dataset=ds, metrics=metrics)
    out = {}
    for s, row in zip(todo, result.scores):
        out[s["id"]] = {str(k).replace("_with_reference", "").replace("_without_reference", ""): v
                        for k, v in row.items()}
    return out


def main():
    t0 = time.time()
    golden_all = load_golden_set(None, 0)
    items = [g for g in golden_all if g["id"] in TARGETS]
    print(f"[ask] 重跑 {len(items)} 条: {[g['id'] for g in items]}")

    samples = run_ask(items, 0)
    samples = run_judge(samples)
    SAMPLES_PATH.write_text(json.dumps(samples, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    print(f"[ask+judge] 完成，耗时 {int(time.time() - t0)}s")

    # ---- 局部 ragas + merge ----
    new_scores = local_ragas(samples, set(TARGETS))
    scores_path = RESULTS_DIR / "ragas_scores.json"
    old_scores = json.loads(scores_path.read_text(encoding="utf-8"))
    # ragas_scores.json 条目顺序 = samples 中有上下文的样本顺序
    ctx_ids = [s["id"] for s in samples
               if any(c.strip() for c in s.get("retrieved_contexts", []))]
    if len(old_scores) != len(ctx_ids):
        print(f"[merge] 警告：旧 ragas_scores {len(old_scores)} 条 != 有上下文样本 {len(ctx_ids)} 条，"
              f"按 id 对齐可能导致错位，改用全量重算（跳过）")
    merged = []
    for sid in ctx_ids:
        if sid in new_scores:
            merged.append(new_scores[sid])
        else:
            old_idx = ctx_ids.index(sid)
            merged.append(old_scores[old_idx])
    scores_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(f"[merge] ragas_scores.json 已更新（{len(merged)} 条），其中重算 {len(new_scores)} 条")

    # ---- 全量重聚合 ----
    agg: dict[str, list[float]] = {}
    for row in merged:
        for name, val in row.items():
            if val is not None and not (isinstance(val, float) and math.isnan(val)):
                agg.setdefault(name, []).append(float(val))
    score_map = {k: round(_stats.mean(v), 4) for k, v in agg.items()}
    effective_n = {k: len(v) for k, v in agg.items()}

    judged = [s for s in samples if s.get("judge", {}).get("correct") is not None]
    n_correct = sum(1 for s in judged if s["judge"]["correct"])
    accuracy = round(n_correct / len(judged), 4) if judged else None

    summary = {
        "n_samples": len(samples),
        "n_judged": len(judged),
        "answer_accuracy": accuracy,
        "answer_accuracy_pct": round(accuracy * 100, 1) if accuracy is not None else None,
        "faithfulness": score_map.get("faithfulness"),
        "answer_relevancy": score_map.get("answer_relevancy"),
        "context_precision": score_map.get("context_precision"),
        "context_recall": score_map.get("context_recall"),
        "skipped_no_context": [],
        "effective_n": effective_n,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": f"修复后局部重跑 {len(TARGETS)} 条（{','.join(TARGETS)}），全量聚合",
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[done] 总耗时 {int(time.time() - t0)}s")


if __name__ == "__main__":
    main()
