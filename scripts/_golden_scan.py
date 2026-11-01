# -*- coding: utf-8 -*-
import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
g = json.load(open(r"D:\论文阅读问答助手\eval\golden_set.json", encoding="utf-8"))
out = []
if isinstance(g, dict):
    items = g.get("questions", g.get("items", []))
else:
    items = g
out.append("总数: %d" % len(items))
for x in items:
    qid = x.get("id", x.get("qid", "?"))
    q = x.get("question", "")
    r = x.get("reference", "")
    if "四篇" in r or "四篇" in q or "这四篇" in r or "这四篇" in q:
        out.append("\n=== %s ===" % qid)
        out.append("Q: %s" % q)
        out.append("R: %s" % r)
open(r"D:\论文阅读问答助手\eval\results\_golden_4p.txt", "w", encoding="utf-8").write("\n".join(out))
print("done")
