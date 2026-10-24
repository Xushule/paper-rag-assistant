# -*- coding: utf-8 -*-
"""按需导出指定 chunk 全文，供出题锚定原文。用法: python _read_chunks.py <论文前缀> <块id列表>"""
import pickle, sys
sys.path.insert(0, r"D:\论文阅读问答助手")
chunks = pickle.load(open(r"D:\论文阅读问答助手\data\chunks.pkl", "rb"))
by_id = {c.chunk_id: c for c in chunks}

prefix = sys.argv[1]
ids = sys.argv[2:]
for cid in ids:
    key = prefix + "-" + cid
    c = by_id.get(key)
    print("=" * 70)
    print(key, "len=%d" % len(c.content) if c else "缺失")
    if c:
        print(c.content)
