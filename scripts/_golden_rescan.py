# -*- coding: utf-8 -*-
import sys, io, json, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
g = json.load(open(r"D:\论文阅读问答助手\eval\golden_set.json", encoding="utf-8"))
items = g.get("questions", g.get("items", g)) if isinstance(g, dict) else g
out = []
pats = ["四篇", "4 篇", "4篇", "这四篇", "这 4 篇", "这4篇", "库中四篇", "库中4篇", "库中 4 篇"]
paper_names = ["AGAN", "NightRain", "Raindrop Clarity", "NTIRE", "Restormer"]
for x in items:
    qid = x.get("id")
    q = x.get("question", "")
    r = x.get("reference", "")
    flags = []
    for p in pats:
        if p in q or p in r:
            flags.append(p)
    # 隐含 4 篇列举：参考/问题中出现 3 篇以上论文名但无 Restormer，且题干是列举类
    if not flags:
        names_in_r = [n for n in paper_names if n in r]
        names_in_q = [n for n in paper_names if n in q]
        if len(names_in_r) >= 4 and "Restormer" not in r:
            flags.append("列举含4篇论文名(无Restormer)")
        if len(names_in_q) >= 4 and "Restormer" not in q:
            flags.append("问题列举4篇论文名(无Restormer)")
    if flags:
        out.append("=== %s  命中: %s" % (qid, "; ".join(flags)))
        out.append("Q: %s" % q)
        out.append("R: %s" % r)
        out.append("")
open(r"D:\论文阅读问答助手\eval\results\_golden_rescan.txt", "w", encoding="utf-8").write("\n".join(out))
print("done, hits=%d" % out.count("==="))
