# -*- coding: utf-8 -*-
"""dump 各论文 chunk 结构（id + 前 120 字符），供出题定位。"""
import pickle, sys
sys.path.insert(0, r"D:\论文阅读问答助手")
chunks = pickle.load(open(r"D:\论文阅读问答助手\data\chunks.pkl", "rb"))

from collections import defaultdict
by_src = defaultdict(list)
for c in chunks:
    by_src[c.source].append(c)

for src, cl in by_src.items():
    cl.sort(key=lambda x: x.chunk_id)
    print("=" * 80)
    print(src, "块数:", len(cl))
    for c in cl:
        head = " ".join(c.content.split())[:110]
        print("  %s len=%d  %s" % (c.chunk_id, len(c.content), head))
