# -*- coding: utf-8 -*-
import json, math

base = r"D:\论文阅读问答助手\eval\results"
s = json.loads(open(base + r"\eval_summary.json", encoding="utf-8").read())
print("effective_n:", s.get("effective_n"))
scores = json.loads(open(base + r"\ragas_scores.json", encoding="utf-8").read())
valid = [(i, v) for i, v in enumerate(r.get("faithfulness") for r in scores)
         if v is not None and not (isinstance(v, float) and math.isnan(v))]
print("faithfulness 有效:", len(valid), "/ 43")
sm = json.loads(open(base + r"\samples.json", encoding="utf-8").read())
valid.sort(key=lambda x: x[1])
print("\n最低 6 条:")
for i, v in valid[:6]:
    sid = sm[i]["id"]; cat = sm[i]["category"]
    cl = [len(c) for c in sm[i]["retrieved_contexts"]]
    print("  idx%d %s [%s] f=%.3f ctx_len=%s" % (i, sid, cat, v, cl))
print("\n最高 3 条:")
for i, v in valid[-3:]:
    print("  idx%d %s [%s] f=%.3f" % (i, sm[i]["id"], sm[i]["category"], v))
