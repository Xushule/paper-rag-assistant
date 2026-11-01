# -*- coding: utf-8 -*-
import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
def load(path):
    s = json.load(open(path, encoding="utf-8"))
    d = {}
    for x in s:
        qid = x.get("id")
        j = x.get("judge")
        d[qid] = (bool(j.get("correct")) if isinstance(j, dict) else j == "对", str(j.get("missing"))[:120] if isinstance(j, dict) else "")
    return d
old = load(r"D:\论文阅读问答助手\eval\results\samples_full.json")
new = load(r"D:\论文阅读问答助手\eval\results\samples_full_5p.json")
out = []
for q in sorted(set(old) & set(new)):
    o, n = old[q], new[q]
    if o[0] != n[0]:
        out.append("%s: 旧=%s 新=%s  新missing: %s" % (q, o[0], n[0], n[1]))
        # 回答是否提及 Restormer
open(r"D:\论文阅读问答助手\eval\results\_full_5p_diff.txt", "w", encoding="utf-8").write("\n".join(out))
print("done, 判定变化 %d 题" % len(out))
