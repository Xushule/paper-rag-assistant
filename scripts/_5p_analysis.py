# -*- coding: utf-8 -*-
import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

def load(path):
    s = json.load(open(path, encoding="utf-8"))
    return {x["id"]: x for x in s}

def judge_ok(x):
    j = x.get("judge")
    if isinstance(j, dict):
        return bool(j.get("correct"))
    return j is True or j == "对"

groups = ["full_5p", "no_mmr_5p", "no_expand_5p", "vector_only_5p", "mineru_ab_5p", "mineru_v2_5p"]
D = {g: load(r"D:\论文阅读问答助手\eval\results\samples_%s.json" % g) for g in groups}
out = []

# 1) 类别分解
out.append("== 1) 各类别准确率（5 篇口径）==")
cats = ["rc", "agan", "nr", "ntire", "all"]
for g in groups:
    parts = []
    for c in cats:
        ids = [i for i in D[g] if i.startswith(c)]
        ok = sum(1 for i in ids if judge_ok(D[g][i]))
        parts.append("%s %d/%d" % (c, ok, len(ids)))
    out.append("%-14s %s" % (g, " | ".join(parts)))

# 2) all 类 18 题逐题矩阵
out.append("\n== 2) all 类 18 题 6 组对错矩阵（✓对 ✗错）==")
all_ids = sorted(i for i in D["full_5p"] if i.startswith("all"))
out.append("题目      " + "  ".join(g[:9].replace("_5p","") for g in groups))
for i in all_ids:
    row = "  ".join("✓" if judge_ok(D[g][i]) else "✗" for g in groups)
    q = D["full_5p"][i]["question"][:22]
    out.append("%-8s %s  %s" % (i, row, q))

# 3) mineru_ab_5p vs mineru_v2_5p 逐题 diff
out.append("\n== 3) mineru v1 vs v2（5 篇口径）判定差异 ==")
for i in sorted(D["mineru_ab_5p"]):
    a, b = judge_ok(D["mineru_ab_5p"][i]), judge_ok(D["mineru_v2_5p"][i])
    if a != b:
        out.append("%s: v1=%s v2=%s  %s" % (i, "对" if a else "错", "对" if b else "错", D["full_5p"][i]["question"][:40]))

# 4) full_5p vs 旧 full 逐题 diff（口径变化影响）
old = load(r"D:\论文阅读问答助手\eval\results\samples_full.json")
out.append("\n== 4) full 新旧口径判定差异（旧4篇→新5篇）==")
for i in sorted(D["full_5p"]):
    a, b = judge_ok(old[i]), judge_ok(D["full_5p"][i])
    if a != b:
        out.append("%s: 旧=%s 新=%s  %s" % (i, "对" if a else "错", "对" if b else "错", D["full_5p"][i]["question"][:40]))

open(r"D:\论文阅读问答助手\eval\results\_5p_analysis.txt", "w", encoding="utf-8").write("\n".join(out))
print("done, lines=%d" % len(out))
