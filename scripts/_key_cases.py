# -*- coding: utf-8 -*-
import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

def load(path):
    return {x["id"]: x for x in json.load(open(path, encoding="utf-8"))}

full = load(r"D:\论文阅读问答助手\eval\results\samples_full_5p.json")
v2 = load(r"D:\论文阅读问答助手\eval\results\samples_mineru_v2_5p.json")
out = []

def show(d, qid, tag):
    x = d[qid]
    j = x.get("judge") or {}
    out.append("### %s %s  judge=%s  missing=%s" % (tag, qid, j.get("correct"), str(j.get("missing"))[:150]))
    out.append("Q: %s" % x["question"][:100])
    out.append("R: %s" % x["reference"][:200])
    out.append("A: %s" % x["answer"][:600])
    out.append("")

# all-01 全组错: 看 full 的回答
show(full, "all-01", "full_5p")
# all-04 full 错: 回答是否覆盖 Restormer
show(full, "all-04", "full_5p")
# 参考答案未变但判定翻转的题（judge 噪声候选）
show(full, "rc-25", "full_5p")
show(full, "nr-18", "full_5p")
show(full, "ntire-06", "full_5p")
# v1 对 v2 错的 nr-05 / ntire-06
show(v2, "nr-05", "mineru_v2_5p")

open(r"D:\论文阅读问答助手\eval\results\_key_cases.txt", "w", encoding="utf-8").write("\n".join(out))
print("done")
