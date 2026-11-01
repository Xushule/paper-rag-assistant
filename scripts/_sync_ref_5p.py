# -*- coding: utf-8 -*-
import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
g = json.load(open(r"D:\论文阅读问答助手\eval\golden_set.json", encoding="utf-8"))
items = g.get("questions", g.get("items", g)) if isinstance(g, dict) else g
ref_by_id = {it["id"]: it["reference"] for it in items}
groups = [
    ("full", "full_5p"),
    ("no_mmr", "no_mmr_5p"),
    ("no_expand", "no_expand_5p"),
    ("vector_only", "vector_only_5p"),
    ("mineru_ab", "mineru_ab_5p"),
    ("mineru_v2", "mineru_v2_5p"),
]
for old, new in groups:
    src = r"D:\论文阅读问答助手\eval\results\samples_%s.json" % old
    dst = r"D:\论文阅读问答助手\eval\results\samples_%s.json" % new
    s = json.load(open(src, encoding="utf-8"))
    n_ref, n_judge = 0, 0
    for x in s:
        qid = x.get("id")
        if qid in ref_by_id:
            x["reference"] = ref_by_id[qid]
            n_ref += 1
        if "judge" in x:
            x["judge"] = None
            n_judge += 1
    json.dump(s, open(dst, "w", encoding="utf-8"), ensure_ascii=False)
    print("%s -> %s  samples=%d  reference更新=%d  judge清空=%d" % (old, new, len(s), n_ref, n_judge))
