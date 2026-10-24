# -*- coding: utf-8 -*-
"""冒烟验证：max_tokens=4096 + strictness=1 后，faithfulness 的 nan 是否消失。
取前 8 条 + 曾 nan 的 rc-05(idx4)/all-01(idx32)，共 10 条。"""
import json, math, sys
sys.path.insert(0, r"D:\论文阅读问答助手")
sys.path.insert(0, r"D:\论文阅读问答助手\eval")
from run_eval import run_ragas

sm = json.loads(open(r"D:\论文阅读问答助手\eval\results\samples.json",
                     encoding="utf-8").read())
ids = [0, 1, 2, 3, 4, 8, 12, 27, 31, 32]   # 含曾 nan 的 idx4/idx32
subset = [sm[i] for i in ids]

res = run_ragas(subset)
print("scores:", res["scores"])
print("effective_n:", res["effective_n"])

scores = json.loads(open(r"D:\论文阅读问答助手\eval\results\ragas_scores.json",
                         encoding="utf-8").read())
print("逐样本条数:", len(scores))
nan_cnt = {}
for row in scores:
    for k, v in row.items():
        if v is None or (isinstance(v, float) and math.isnan(v)):
            nan_cnt.setdefault(k, 0)
            nan_cnt[k] += 1
print("nan 计数:", nan_cnt or "无 nan")
