# -*- coding: utf-8 -*-
"""当前参数的 hybrid 明细 + ref 过滤判定。"""
import pickle, sys, re
sys.path.insert(0, r"D:\论文阅读问答助手")
from src.retrieval.embed_store import VectorStore
from src.retrieval.bm25_index import build_index, search as bm25_search
from src.retrieval.hybrid import hybrid_search

DATA = r"D:\论文阅读问答助手\data"
chunks = pickle.load(open(DATA + r"\chunks.pkl", "rb"))
store = VectorStore(persist_dir=DATA + r"\chroma_db_v2")
bm25 = build_index(chunks)
mapping = {c.chunk_id: c for c in chunks}

scope = "AGAN_Attentive_Raindrop_1711.10098.pdf"
q = "这些论文中哪些提出了新的数据集？"

res = store.search(q, k=16, where={"source": scope})
vids = [c for c in res["ids"][0] if c in mapping]
print("vector_ids(k=16):", vids)
print("  18 位置:", vids.index("AGAN_Attentive_Raindrop_1711.10098-18") if "AGAN_Attentive_Raindrop_1711.10098-18" in vids else "不在")

ref_noise = re.compile(
    r"(in proceedings|in european conference|in ieee|in international "
    r"conference|arxiv preprint|pages \d)", re.IGNORECASE)
def _is_ref_block(content):
    lines = [l.strip() for l in content.splitlines() if l.strip()]
    if len(lines) < 3:
        return False
    ref_hits = sum(1 for l in lines if ref_noise.search(l))
    return ref_hits >= max(3, int(len(lines) * 0.6))

c18 = [c for c in chunks if c.chunk_id == "AGAN_Attentive_Raindrop_1711.10098-18"][0]
print("AGAN-18 source:", repr(c18.source), "| is_ref_block:", _is_ref_block(c18.content))

hits = hybrid_search(store, bm25, chunks, q, bm25_query=q,
                     vector_k=16, bm25_k=16, final_k=10, scope=scope)
print("hybrid final top10:", [c.chunk_id for c, _ in hits])
