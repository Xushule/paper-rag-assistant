# -*- coding: utf-8 -*-
import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
g = json.load(open(r"D:\论文阅读问答助手\eval\golden_set.json", encoding="utf-8"))
items = g.get("questions", g.get("items", g)) if isinstance(g, dict) else g
out = []
for x in items:
    out.append("[%s] (%s)" % (x["id"], x.get("category", "")))
    out.append("Q: %s" % x["question"])
    out.append("R: %s" % x["reference"])
    out.append("")
open(r"D:\论文阅读问答助手\eval\results\_golden_full_audit.txt", "w", encoding="utf-8").write("\n".join(out))
print("exported %d items" % len(items))
