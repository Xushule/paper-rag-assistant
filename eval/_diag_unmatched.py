# -*- coding: utf-8 -*-
"""诊断 7 个未匹配 snippet：尝试包含匹配定位 chunk，判断能否换完整块。"""
import json, pickle, sys
sys.path.insert(0, r"D:\论文阅读问答助手")

chunks = pickle.load(open(r"D:\论文阅读问答助手\data\chunks.pkl", "rb"))
by_id = {c.chunk_id: c for c in chunks}

sm = json.loads(open(r"D:\论文阅读问答助手\eval\results\samples.json", encoding="utf-8").read())

unmatched = []
for s in sm:
    for c in s["retrieved_contexts"]:
        if len(c) < 1500:  # 完整块通常 >1500，短的就是残留 snippet
            unmatched.append((s["id"], c))

print("短片段总数:", len(unmatched))
for sid, snip in unmatched:
    # 包含匹配：取 snippet 中不含空白的 60 字符片段在 chunk 里找
    core = " ".join(snip.split())[:60]
    hits = [c for c in chunks if core in " ".join(c.content.split())]
    if hits:
        # 展示最长命中 chunk 与 snippet 的关系
        c = max(hits, key=lambda x: len(x.content))
        print("%s len=%d -> 命中 %s chunk(%s) len=%d  首80: %r" % (
            sid, len(snip), c.chunk_id, c.source, len(c.content), snip[:80]))
    else:
        print("%s len=%d -> 无命中  首80: %r" % (sid, len(snip), snip[:80]))
