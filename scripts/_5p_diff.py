# -*- coding: utf-8 -*-
import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
base = r"D:\论文阅读问答助手\eval\results"

def load(tag):
    p = f"{base}\\samples_{tag}.json"
    d = json.load(open(p, encoding="utf-8"))
    return d

def verdict(s):
    j = s.get("judge") or {}
    if isinstance(j, dict):
        return j.get("correct")
    return False

full = {s["id"]: s for s in load("full_5p")}
v1 = {s["id"]: s for s in load("mineru_ab_5p")}
v2 = {s["id"]: s for s in load("mineru_v2_5p")}

def is_ok(s):
    return verdict(s) is True

lines = []
# 1) full vs v2
both = [q for q in full if q in v2]
diff_fv = [(q, is_ok(full[q]), is_ok(v2[q])) for q in both if is_ok(full[q]) != is_ok(v2[q])]
lines.append(f"[full vs v2] 共同 {len(both)}，差异 {len(diff_fv)} 题：")
for q, a, b in diff_fv:
    lines.append(f"  {q}: full={a} v2={b}  Q={full[q]['question'][:40]}")
    if not b:
        lines.append(f"    full 答案片段: {full[q].get('answer','')[:200]}")
        lines.append(f"    v2   答案片段: {v2[q].get('answer','')[:200]}")
# 2) v1 vs v2
dv = [(q, is_ok(v1[q]), is_ok(v2[q])) for q in v2 if q in v1 and is_ok(v1[q]) != is_ok(v2[q])]
lines.append(f"\n[v1 vs v2] 差异 {len(dv)} 题：")
for q, a, b in dv:
    lines.append(f"  {q}: v1={a} v2={b}  Q={v2[q]['question'][:40]}")
# 3) full 与 v2 差异题的类别归属
cat = {"rc": "数据集", "agan": "方法", "nr": "夜间去雨", "ntire": "挑战赛", "all": "全库对比"}
for q, a, b in diff_fv:
    lines.append(f"  类别: {q.split('-')[0]} -> {cat.get(q.split('-')[0],'?')}")

open(r"D:\论文阅读问答助手\eval\results\_5p_diff.txt", "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines))
