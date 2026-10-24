# -*- coding: utf-8 -*-
"""诊断错题检索命中：复刻 ask_one 的检索链路，看每道错题命中哪些块。"""
import sys, pickle, json
sys.path.insert(0, r"D:\论文阅读问答助手")
from src.retrieval.hybrid import hybrid_search
from src.retrieval.embed_store import VectorStore
from src.retrieval.bm25_index import load_or_build

DATA = r"D:\论文阅读问答助手\data"
chunks = pickle.load(open(DATA + r"\chunks.pkl", "rb"))
store = VectorStore(data_dir=DATA)
bm25 = load_or_build(DATA, chunks)

# 复刻 ask_one 的检索：无 scope 全库 top-K
qs = [
    ("rc-20", "RobotCar 立体数据集有什么缺陷？", None),
    ("ntire-15", "NTIRE 2025 排行榜中模型复杂度（参数量）的范围是多少？", None),
    ("all-17", "这些论文中哪些提出了新的数据集？", None),
    ("all-18", "NTIRE 报告里有哪些基于 Restormer 的方法？", None),
]
for qid, q, scope in qs:
    hits = hybrid_search(store, bm25, chunks, q, final_k=12, scope=scope)
    print("=" * 70)
    print(qid, "|", q)
    for c, sc in hits:
        print("   hit: %s rrf=%.4f | %s" % (c.chunk_id, sc, " ".join(c.content.split())[:85]))
