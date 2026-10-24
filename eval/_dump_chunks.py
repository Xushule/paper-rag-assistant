# -*- coding: utf-8 -*-
"""临时工具：导出已入库分块内容，供人工构造 Golden Set 使用。
用法：python eval/_dump_chunks.py > eval/_chunks_dump.txt
"""
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.bm25_index import load_index

bm25, chunks = load_index()
print("总块数:", len(chunks))
print("论文分布:", dict(Counter(c.source for c in chunks)))
print()
for src in sorted({c.source for c in chunks}):
    print("=" * 72)
    print("论文:", src, "共", sum(1 for c in chunks if c.source == src), "块")
    for c in chunks:
        if c.source == src:
            sec = c.section or ""
            print(f"\n--- p{c.page} | {sec} | id={c.chunk_id} ---")
            print(c.content[:600])
