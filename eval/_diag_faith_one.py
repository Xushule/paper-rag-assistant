# -*- coding: utf-8 -*-
"""单样本诊断：rc-14 / rc-15 的 faithfulness 断言拆分与支持情况。"""
import json, sys
sys.path.insert(0, r"D:\论文阅读问答助手")
from datasets import Dataset
from ragas.metrics import faithfulness
from ragas import evaluate
from openai import OpenAI
from ragas.llms import llm_factory
from src.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
faithfulness.llm = llm_factory(LLM_MODEL, client=client, max_tokens=4096)

samples = {x["id"]: x for x in json.load(open(r"D:\论文阅读问答助手\eval\results\samples.json", encoding="utf-8"))}
for idx in ["rc-14", "rc-15"]:
    s = samples[idx]
    ds = Dataset.from_dict({
        "question": [s["question"]],
        "answer": [s["answer"]],
        "contexts": [s["retrieved_contexts"]],
    })
    result = evaluate(ds, metrics=[faithfulness])
    print("=" * 70)
    print(idx, "->", result)
    row = result.to_pandas().iloc[0].to_dict()
    for k, v in row.items():
        if v is not None:
            print(" ", k, "=", v)
