# -*- coding: utf-8 -*-
"""验证修复方向：
1) 对比链路全部论文模式：子查询带主题词（去指代词）能否命中 all-17/all-18 目标块
2) 数值统计意图问题：表格块补召回能否命中 ntire-15 排行榜表
"""
import pickle, sys, re
sys.path.insert(0, r"D:\论文阅读问答助手")
from src.retrieval.hybrid import hybrid_search
from src.retrieval.embed_store import VectorStore
from src.retrieval.bm25_index import build_index, search as bm25_search
from src.qa.query_rewriter import expand_query

DATA = r"D:\论文阅读问答助手\data"
chunks = pickle.load(open(DATA + r"\chunks.pkl", "rb"))
store = VectorStore(persist_dir=DATA + r"\chroma_db_v2")
bm25 = build_index(chunks)

def is_table(c):
    return c.content.count("|") >= 4

def probe(q, scope, k=10):
    hits = hybrid_search(store, bm25, chunks, q, bm25_query=expand_query(q),
                         vector_k=16, bm25_k=16, final_k=k, scope=scope)
    return [(c.chunk_id, "TABLE" if is_table(c) else "text") for c, _ in hits]

# ---- 1) all-17 / all-18：每篇 scope 下用主题词查询 ----
print("### all-17 主题查询：'这些论文中哪些提出了新的数据集？'")
for src in ["AGAN_Attentive_Raindrop_1711.10098.pdf",
            "NightRain_Nighttime_Deraining_2401.00729.pdf"]:
    hits = probe("这些论文中哪些提出了新的数据集？", src, 10)
    print(src[:20], "->", hits[:6])

print()
print("### all-18 主题查询：'这些论文中哪些基于 Restormer 的方法？'")
hits = probe("这些论文中哪些基于 Restormer 的方法？",
             "NTIRE2025_DayNight_Raindrop_2504.12711.pdf", 12)
print("NTIRE ->", hits[:8])

# ---- 2) ntire-15：原查询 + 表格块补召回 ----
print()
print("### ntire-15 原查询（全库）")
hits = probe("NTIRE 2025 排行榜中模型复杂度（参数量）的范围是多少？", None, 16)
print("全库 top16 ->", hits)
