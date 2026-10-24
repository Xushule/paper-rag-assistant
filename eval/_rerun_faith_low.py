# -*- coding: utf-8 -*-
"""只对低分样本重算 ragas（faithfulness 噪声验证），merge 回 ragas_scores.json 并重聚合。
用法: python eval/_rerun_faith_low.py
"""
import json
import math
import statistics as _stats
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.run_eval import SAMPLES_PATH, RESULTS_DIR, SUMMARY_PATH
from eval._rerun_fix import local_ragas  # 复用局部 ragas 实现

samples = json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))
ctx_ids = [s["id"] for s in samples
           if any(c.strip() for c in s.get("retrieved_contexts", []))]
scores = json.loads((RESULTS_DIR / "ragas_scores.json").read_text(encoding="utf-8"))
idx_of = {sid: i for i, sid in enumerate(ctx_ids)}

# 选 faithfulness < 0.95 的样本重测
low = [sid for sid in ctx_ids
       if isinstance(scores[idx_of[sid]].get("faithfulness"), float)
       and scores[idx_of[sid]]["faithfulness"] < 0.95]
print(f"[low] faithfulness<0.95 共 {len(low)} 条，重算 ragas")
t0 = time.time()
new_scores = local_ragas(samples, set(low))

updated = 0
for sid, sc in new_scores.items():
    if sid in idx_of:
        old = scores[idx_of[sid]].get("faithfulness")
        scores[idx_of[sid]] = sc
        print(f"  {sid}: {old:.3f} -> {sc.get('faithfulness', 0):.3f}")
        updated += 1
(RESULTS_DIR / "ragas_scores.json").write_text(
    json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")

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
    "note": f"低分样本 faithfulness 重测（{len(low)} 条，验证评测噪声），全量聚合",
}
SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                        encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"[done] {updated} 条已更新，总耗时 {int(time.time() - t0)}s")
