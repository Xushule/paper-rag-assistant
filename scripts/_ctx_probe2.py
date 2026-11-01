# -*- coding: utf-8 -*-
import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
out = []
def probe(path, label):
    s = json.load(open(path, encoding="utf-8"))
    total_ctx = 0
    rest_ctx = 0
    rest_answers = 0
    for x in s:
        c = x.get("retrieved_contexts") or []
        total_ctx += len(c)
        for seg in c:
            if "Restormer" in str(seg):
                rest_ctx += 1
        if "Restormer" in str(x.get("answer", "")):
            rest_answers += 1
    out.append("%s: contexts总段=%d, 含Restormer段=%d, 回答提及Restormer题数=%d" % (
        label, total_ctx, rest_ctx, rest_answers))
for path, label in [
    (r"D:\论文阅读问答助手\eval\results\samples_full.json", "baseline full(93%)"),
    (r"D:\论文阅读问答助手\eval\results\samples_no_mmr.json", "消融-关MMR(91%)"),
    (r"D:\论文阅读问答助手\eval\results\samples_no_expand.json", "消融-关查询扩展(89%)"),
    (r"D:\论文阅读问答助手\eval\results\samples_vector_only.json", "消融-仅向量(90%)"),
    (r"D:\论文阅读问答助手\eval\results\samples_mineru_ab.json", "mineru v1(87%)"),
    (r"D:\论文阅读问答助手\eval\results\samples_mineru_v2.json", "mineru v2(92%, 5篇库)"),
]:
    probe(path, label)
open(r"D:\论文阅读问答助手\eval\results\_ctx_probe2.txt", "w", encoding="utf-8").write("\n".join(out))
print("done")
