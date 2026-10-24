# -*- coding: utf-8 -*-
"""用 ragas 官方 faithfulness 重算（修复 nan 污染：失败样本返回 nan，
均值聚合时过滤非有限值）。单指标，快；结果写 faithfulness_ragas.json。

用法：python eval/_faithfulness_ragas.py
"""
import json
import math
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openai import OpenAI

from src.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

RESULTS = Path(__file__).resolve().parent / "results"
SAMPLES = json.loads((RESULTS / "samples.json").read_text(encoding="utf-8"))


def main():
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.llms import llm_factory
    from ragas.metrics import faithfulness

    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL,
                    timeout=60, max_retries=1)
    faithfulness.llm = llm_factory(LLM_MODEL, client=client)

    ds_samples = []
    skipped = []
    for s in SAMPLES:
        ctx = [c for c in s.get("retrieved_contexts", []) if c.strip()]
        if not ctx:
            skipped.append(s["id"])
            continue
        ds_samples.append(SingleTurnSample(
            user_input=s["question"], response=s["answer"],
            retrieved_contexts=ctx, reference=s["reference"]))
    dataset = EvaluationDataset(samples=ds_samples)
    result = evaluate(dataset=dataset, metrics=[faithfulness])

    per_sample = []
    vals = []
    for s, row in zip(ds_samples, result.scores):
        v = row.get("faithfulness")
        if v is not None and math.isfinite(float(v)):
            vals.append(float(v))
            per_sample.append({"question": s.user_input[:40], "faithfulness": round(float(v), 4)})
        else:
            per_sample.append({"question": s.user_input[:40], "faithfulness": None})
    mean = round(sum(vals) / len(vals), 4) if vals else None
    out = {
        "metric": "faithfulness (ragas 官方)",
        "n": len(ds_samples),
        "n_valid": len(vals),
        "mean_faithfulness": mean,
        "hallucination_rate_approx": round(1 - mean, 4) if mean is not None else None,
        "skipped_no_context": skipped,
        "per_sample": per_sample,
    }
    (RESULTS / "faithfulness_ragas.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("有效样本:", len(vals), "/", len(ds_samples))
    print("均值 faithfulness:", mean, "≈ 幻觉率:", out["hallucination_rate_approx"])


if __name__ == "__main__":
    main()
