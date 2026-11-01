# -*- coding: utf-8 -*-
import sys, io, json, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
g = json.load(open(r"D:\论文阅读问答助手\eval\golden_set.json", encoding="utf-8"))
items = g.get("questions", g.get("items", g)) if isinstance(g, dict) else g
out = []

# A. 篇数表述检查
num_pat = r"(?:四篇|4 ?篇|五篇|5 ?篇|三篇|3 ?篇|两篇|2 ?篇|一篇|1 ?篇|共 ?\d+ ?篇)"
for x in items:
    q, r = x["question"], x["reference"]
    for field, txt in (("Q", q), ("R", r)):
        hits = re.findall(num_pat, txt)
        if hits:
            out.append("[篇数] %s %s: %s | %s" % (x["id"], field, hits, txt[:120]))

# B. 论文列举完整性：references 中列出 >=2 个论文名时检查 Restormer 是否需要
names = ["AGAN", "NightRain", "Raindrop Clarity", "NTIRE", "Restormer"]
for x in items:
    r = x["reference"]
    present = [n for n in names if n in r]
    if len(present) >= 2 and "Restormer" not in present:
        # 列举类/对比类/元问题才需要 5 篇全列；单篇事实题不需要
        q = x["question"]
        if any(k in q for k in ["这五篇", "这些论文", "库中五篇", "库里", "对比", "都有", "各自", "分别", "有哪些", "哪些"]):
            out.append("[列举缺Restormer] %s 列出=%s Q:%s" % (x["id"], present, q[:50]))

# C. Restormer 关键事实校验
for x in items:
    r = x["reference"]
    if "Restormer" in r:
        bad = []
        if "09839" in r: bad.append("错误编号09839")
        if "19881" in r: bad.append("错误编号19881")
        if "2099" in r or "2026" in r: bad.append("年份异常")
        if "数据集论文" in r and "通用图像复原" not in r and "方法" not in r:
            bad.append("Restormer可能被误归数据集")
        if bad:
            out.append("[Restormer事实] %s %s | %s" % (x["id"], "; ".join(bad), r[:150]))

# D. 全量题目 scope 与 id 前缀一致性
from collections import Counter
scopes = Counter()
for x in items:
    qid = x["id"]
    prefix = qid.split("-")[0]
    scopes[prefix] += 1
out.append("[分布] %s" % dict(scopes))

open(r"D:\论文阅读问答助手\eval\results\_golden_machine_audit.txt", "w", encoding="utf-8").write("\n".join(out))
print("audit lines: %d" % len(out))
