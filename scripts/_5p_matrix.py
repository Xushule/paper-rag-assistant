# -*- coding: utf-8 -*-
import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
base = r"D:\论文阅读问答助手\eval\results"

def load(tag):
    return json.load(open(f"{base}\\samples_{tag}.json", encoding="utf-8"))

def ok(s):
    return s.get("judge", {}).get("correct") is True

tags = ["full_5p", "no_mmr_5p", "no_expand_5p", "vector_only_5p"]
samples = {t: load(t) for t in tags}
cats = {"rc": "数据集", "agan": "方法", "nr": "夜间去雨", "ntire": "挑战赛", "all": "全库对比"}

# 每组的类别正确数与总正确数
lines = []
header = f"{'类别(题数)':<14}" + "".join(f"{t:<12}" for t in tags)
lines.append(header)
tot = {t: [0, 0] for t in tags}
for prefix, cname in cats.items():
    row = []
    for t in tags:
        ss = [s for s in samples[t] if s["id"].startswith(prefix + "-")]
        okc = sum(1 for s in ss if ok(s))
        tot[t][0] += okc; tot[t][1] += len(ss)
        row.append(f"{okc}/{len(ss)}")
    lines.append(f"{cname}({len(ss):<2})".ljust(14) + "".join(f"{v:<12}" for v in row))
lines.append(f"{'总计(100)':<14}" + "".join(f"{v[0]}/{v[1]}" for v in tot.values()))

# 各组的按题明细（全库类 18 题逐题正确性）
lines.append("\n[all 类 18 题逐题]")
for t in tags:
    a = [s["id"] for s in samples[t] if s["id"].startswith("all-") and ok(s)]
    lines.append(f"  {t}: {a}")

open(r"D:\论文阅读问答助手\eval\results\_5p_matrix.txt", "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines))
