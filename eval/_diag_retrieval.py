# -*- coding: utf-8 -*-
"""诊断错题检索命中：看问题在 bm25+chroma 检索下命中了哪些块。"""
import sys, pickle, json
sys.path.insert(0, r"D:\论文阅读问答助手")
from src.retrieval.hybrid import HybridRetriever

qs = [
    ("rc-20", "RobotCar 立体数据集有什么缺陷？"),
    ("ntire-15", "NTIRE 2025 排行榜中模型复杂度（参数量）的范围是多少？"),
    ("all-17", "这些论文中哪些提出了新的数据集？"),
    ("all-18", "NTIRE 报告里有哪些基于 Restormer 的方法？"),
    ("all-01", "这个库里一共有几篇论文？"),
]
r = HybridRetriever(data_dir=r"D:\论文阅读问答助手\data")
chunks = pickle.load(open(r"D:\论文阅读问答助手\data\chunks.pkl", "rb"))
by_id = {c.chunk_id: c for c in chunks}

for qid, q in qs:
    hits = r.retrieve(q, top_k=8)
    print("=" * 70)
    print(qid, "|", q)
    for h in hits:
        if isinstance(h, tuple):
            c, score = h
        else:
            c, score = h, None
        print("   hit:", c.chunk_id, "score=%.3f" % score if score else "", "|", " ".join(c.content.split())[:90])
