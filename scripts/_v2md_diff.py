# -*- coding: utf-8 -*-
"""full_5p(pdfplumber) vs mineru_v2md_5p(markdown-aware) 逐题差异 + 类别矩阵"""
import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
base = r"D:\论文阅读问答助手\eval\results"

def load(tag):
    return json.load(open(f"{base}\\samples_{tag}.json", encoding="utf-8"))

def verdict(s):
    j = s.get("judge") or {}
    return j.get("correct") if isinstance(j, dict) else False

full = {s["id"]: s for s in load("full_5p")}
v2 = {s["id"]: s for s in load("mineru_v2_5p")}
md = {s["id"]: s for s in load("mineru_v2md_5p")}

def is_ok(s):
    return verdict(s) is True

lines = []
cats = ["rc", "agan", "nr", "ntire", "all"]

# 1) 各组的类别矩阵（5p 口径）
def matrix(d):
    out = {}
    for q, s in d.items():
        c = q.split("-")[0]
        out.setdefault(c, [0, 0])
        out[c][0] += 1
        if is_ok(s):
            out[c][1] += 1
    return out

lines.append("== 类别矩阵（答对/总数）==")
for name, d in [("full_5p", full), ("mineru_v2_5p", v2), ("mineru_v2md_5p", md)]:
    m = matrix(d)
    parts = "  ".join(f"{c} {m.get(c,[0,0])[1]}/{m.get(c,[0,0])[0]}" for c in cats)
    tot = sum(1 for s in d.values() if is_ok(s))
    lines.append(f"{name}: 总计 {tot}/100 | {parts}")

# 2) full vs v2md 逐题差异
both = [q for q in full if q in md]
diff_fm = [(q, is_ok(full[q]), is_ok(md[q])) for q in both if is_ok(full[q]) != is_ok(md[q])]
lines.append(f"\n[full vs v2md] 差异 {len(diff_fm)} 题：")
for q, a, b in diff_fm:
    lines.append(f"  {q}: pdfplumber={a} v2md={b}  Q={full[q]['question'][:50]}")
    if b and not a:
        lines.append(f"    [v2md 变好] full 答案: {full[q].get('answer','')[:180]}")
        lines.append(f"                 v2md 答案: {md[q].get('answer','')[:180]}")
    elif a and not b:
        lines.append(f"    [v2md 变差] full 答案: {full[q].get('answer','')[:180]}")
        lines.append(f"                 v2md 答案: {md[q].get('answer','')[:180]}")

# 3) 旧 v2 vs v2md
both2 = [q for q in v2 if q in md]
diff_vm = [(q, is_ok(v2[q]), is_ok(md[q])) for q in both2 if is_ok(v2[q]) != is_ok(md[q])]
lines.append(f"\n[旧v2 vs v2md] 差异 {len(diff_vm)} 题：")
for q, a, b in diff_vm:
    lines.append(f"  {q}: v2降级版={a} v2md={b}  Q={md[q]['question'][:50]}")

open(r"D:\论文阅读问答助手\eval\results\_v2md_diff.txt", "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines))
