# -*- coding: utf-8 -*-
"""诊断：Restormer 上传后是否可检索"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.bm25_index import load_index
from src.config import PAPERS_DIR

bm25, chunks = load_index()
print(f"== 索引共 {len(chunks)} 块 ==")

# 1) Restormer 的块有多少、含 MDTA 的有几块
rs = [c for c in chunks if "Restormer_2111.09839" in c.source]
print(f"== Restormer 块数: {len(rs)} ==")
mdta = [c for c in rs if any(k in c.content for k in ("MDTA", "Multi-DConv", "transposed attention", "Transposed"))]
print(f"== 含 MDTA/转置注意力关键词的块: {len(mdta)} ==")
for c in mdta[:6]:
    print(f"  [{c.section}] p{c.page} len={len(c.content)} 预览: {c.content[:80]!r}")

# 2) BM25 检索测试：query 含 MDTA
print("\n== BM25 检索 'MDTA Multi-DConv Head Transposed Attention' ==")
try:
    hits = bm25.search("MDTA Multi-DConv Head Transposed Attention", k=10)
    for i, (idx, score) in enumerate(hits[:10]):
        c = chunks[idx]
        print(f"  #{i+1} score={score:.1f} [{c.source[:40]}] {c.section}")
except Exception as e:
    print("  BM25 检索失败:", e)

# 3) 向量检索测试（Chroma）
print("\n== Chroma 向量检索 'MDTA Multi-DConv Head Transposed Attention' ==")
try:
    from src.retrieval.embed_store import VectorStore
    store = VectorStore()
    res = store.search("MDTA Multi-DConv Head Transposed Attention", k=10)
    for i, (doc, score) in enumerate(res[:10]):
        print(f"  #{i+1} score={score:.3f} [{getattr(doc,'source','?')}] {getattr(doc,'section','?')}")
except Exception as e:
    print("  Chroma 检索失败:", e)
