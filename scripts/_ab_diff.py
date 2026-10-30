"""A/B diff: baseline(full) vs MinerU 解析版，逐题对比 judge 结果并分类统计"""
import json, collections

base = json.load(open(r"D:\论文阅读问答助手\eval\results\samples_full.json", encoding="utf-8"))
miner = json.load(open(r"D:\论文阅读问答助手\eval\results\samples_mineru_ab.json", encoding="utf-8"))

print("base type:", type(base).__name__, "miner type:", type(miner).__name__)
b_idx = {s["id"]: s for s in base}
m_idx = {s["id"]: s for s in miner}
print("b_idx type:", type(b_idx).__name__, "len:", len(b_idx), "sample key:", next(iter(b_idx)))
bi, mi = b_idx, m_idx
common = sorted(set(bi) & set(mi))
print("共同样本:", len(common))

def correct(s):
    return bool(s.get("judge", {}).get("correct"))

stats = collections.Counter()
b_ok, m_ok, both_wrong, both_ok = [], [], [], []
detail = []
for qid in common:
    b, m = correct(bi[qid]), correct(mi[qid])
    pref = qid.split("-")[0]
    if b and m:
        stats[("都对", pref)] += 1; both_ok.append(qid)
    elif b and not m:
        stats[("A对B错", pref)] += 1; b_ok.append(qid)
        detail.append((qid, bi[qid]["category"], "B错：A对\nQ:" + bi[qid]["question"][:100]))
    elif m and not b:
        stats[("A错B对", pref)] += 1; m_ok.append(qid)
        detail.append((qid, bi[qid]["category"], "B对：A错\nQ:" + bi[qid]["question"][:100]))
    else:
        stats[("都错", pref)] += 1; both_wrong.append(qid)

print("=== 汇总 ===")
tot = collections.Counter()
for (k, p), v in sorted(stats.items()):
    print(f"{k:8s} {p:8s} {v}")
    tot[k] += v
print("---", dict(tot))
print("\nA对B错（MinerU 变差 %d）:" % len(b_ok))
for qid, cat, info in detail:
    if "A对B错" in info:
        print(" ", qid, "[", cat, "]", bi[qid]["question"][:60])
print("\nA错B对（MinerU 变好 %d）:" % len(m_ok))
for qid, cat, info in detail:
    if "A错B对" in info:
        print(" ", qid, "[", cat, "]", bi[qid]["question"][:60])

out = r"D:\论文阅读问答助手\eval\results\ab_diff_mineru.txt"
with open(out, "w", encoding="utf-8") as f:
    f.write("A=baseline(full 93%) B=MinerU版(87%)\n\n=== 汇总 ===\n")
    for (k, p), v in sorted(stats.items()):
        f.write(f"{k} {p}: {v}\n")
    f.write(f"合计: {dict(tot)}\n")
    f.write(f"\nA对B错: {b_ok}\n\nA错B对: {m_ok}\n\n都错: {both_wrong}\n")
    f.write("\n=== 明细 ===\n")
    for qid, cat, info in detail:
        f.write(f"[{qid}] {info}\n")
print("已写:", out)
