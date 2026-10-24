# -*- coding: utf-8 -*-
"""实测 _retrieve 的 per_paper 覆盖路径对 all-17/all-18 的命中。"""
import pickle, sys, re
sys.path.insert(0, r"D:\论文阅读问答助手")
from src.retrieval.hybrid import hybrid_search
from src.retrieval.embed_store import VectorStore
from src.retrieval.bm25_index import build_index
from src.qa.query_rewriter import expand_query

DATA = r"D:\论文阅读问答助手\data"
chunks = pickle.load(open(DATA + r"\chunks.pkl", "rb"))
store = VectorStore(persist_dir=DATA + r"\chroma_db_v2")
bm25 = build_index(chunks)

ref_noise = re.compile(
    r"(in proceedings|in european conference|in ieee|in international "
    r"conference|arxiv preprint|pages \d)", re.IGNORECASE)

def _is_ref_block(content: str) -> bool:
    lines = [l.strip() for l in content.splitlines() if l.strip()]
    if len(lines) < 3:
        return False
    ref_hits = sum(1 for l in lines if ref_noise.search(l))
    return ref_hits >= max(3, int(len(lines) * 0.6))

def per_paper(question):
    out = {}
    for src in sorted({c.source for c in chunks}):
        ph = hybrid_search(store, bm25, chunks, question,
                           bm25_query=question,
                           vector_k=16, bm25_k=16, final_k=12, scope=src)
        ph = [(c, s) for c, s in ph if not _is_ref_block(c.content)]
        out[src.split("_")[0]] = [c.chunk_id for c, _ in ph[:5]]
    return out

print("### all-17 per_paper (原问题)")
pp = per_paper("这些论文中哪些提出了新的数据集？")
for k, v in pp.items():
    print(" ", k, "->", v)

print()
print("### all-18 per_paper (原问题)")
pp = per_paper("NTIRE 报告里有哪些基于 Restormer 的方法？")
for k, v in pp.items():
    print(" ", k, "->", v)

print()
print("### all-16 per_paper (原问题)")
pp = per_paper("这些论文共同的最终应用动机是什么？")
for k, v in pp.items():
    print(" ", k, "->", v)
