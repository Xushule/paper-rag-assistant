# -*- coding: utf-8 -*-
"""重跑单条样本（ask+judge+ragas 局部），merge 回全量结果。
用法: python eval/_rerun_one.py <id>
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
    SAMPLES_PATH, RESULTS_DIR, SUMMARY_PATH,
    load_golden_set, run_ask, run_judge,
)
from eval._rerun_fix import local_ragas

SID = sys.argv[1]
golden_all = load_golden_set(None, 0)
item = next(g for g in golden_all if g["id"] == SID)
print(f"[ask] 重跑 {SID}: {item['question'][:40]}")
samples = run_ask([item], 0)
samples = run_judge(samples)
SAMPLES_PATH.write_text(json.dumps(samples, ensure_ascii=False, indent=2),
                        encoding="utf-8")

new_scores = local_ragas(samples, {SID})
scores_path = RESULTS_DIR / "ragas_scores.json"
scores = json.loads(scores_path.read_text(encoding="utf-8"))
ctx_ids = [s["id"] for s in samples
           if any(c.strip() for c in s.get("retrieved_contexts", []))]
old = None
for i, sid in enumerate(ctx_ids):
    if sid == SID:
        old = scores[i].get("faithfulness")
        if SID in new_scores:
            scores[i] = new_scores[SID]
scores_path.write_text(json.dumps(scores, ensure_ascii=False, indent=2),
                       encoding="utf-8")

s = next(x for x in samples if x["id"] == SID)
sc = new_scores.get(SID, {})
print(f"[result] {SID} judge={s['judge']['correct']} missing={s['judge']['missing'][:80]}")
print(f"  faithfulness {old} -> {sc.get('faithfulness')} | ar={sc.get('answer_relevancy')}")

# 全量重聚合
agg: dict[str, list[float]] = {}
for row in scores:
    for name, val in row.items():
        if val is not None and not (isinstance(val, float) and math.isnan(val)):
            agg.setdefault(name, []).append(float(val))
score_map = {k: round(_stats.mean(v), 4) for k, v in agg.items()}
judged = [s for s in samples if s.get("judge", {}).get("correct") is not None]
accuracy = round(sum(1 for s in judged if s["judge"]["correct"]) / len(judged), 4)
summary = {
    "n_samples": len(samples),
    "n_judged": len(judged),
    "answer_accuracy": accuracy,
    "answer_accuracy_pct": round(accuracy * 100, 1),
    "faithfulness": score_map.get("faithfulness"),
    "answer_relevancy": score_map.get("answer_relevancy"),
    "context_precision": score_map.get("context_precision"),
    "context_recall": score_map.get("context_recall"),
    "skipped_no_context": [],
    "effective_n": {k: len(v) for k, v in agg.items()},
    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "note": f"单条重跑 {SID}（修复验证），全量聚合",
}
SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                        encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
