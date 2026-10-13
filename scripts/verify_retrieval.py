# -*- coding: utf-8 -*-
"""检索验证脚本：对已入库的论文提问，同时看向量检索和 BM25 各自返回什么。

用法（先跑过 ingest 之后）：
    python scripts/verify_retrieval.py

目的：确认"论文真的进库了 + 两路检索都能命中"，并直观感受
向量检索与 BM25 的差异（这正是 M2 做混合检索的动机）。
"""
import sys
from pathlib import Path

# 把项目根目录加进模块搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.bm25_index import load_index, search
from src.retrieval.embed_store import VectorStore


def main() -> None:
    # 按你实际论文内容修改这几个问题
    queries = [
        "这篇论文提出了什么方法？",
        "实验用了什么数据集？",
        "论文的主要贡献是什么？",
    ]

    # 重新加载已持久化的向量库（会再次加载 embedding 模型，较慢）
    store = VectorStore()
    print(f"[verify] 向量库现有 {store.count()} 个块\n")

    bm25, chunks = load_index()

    for q in queries:
        print("=" * 56)
        print(f"问题: {q}")

        # --- 向量检索 ---
        print("-- 向量检索 top3 --")
        res = store.search(q, k=3)
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            print(f"  [{meta.get('source')} p{meta.get('page')}] {doc[:70]}...")

        # --- BM25 检索 ---
        print("-- BM25 top3 --")
        for chunk, score in search(bm25, chunks, q, k=3):
            print(f"  [{chunk.source} p{chunk.page}] {chunk.content[:70]}...")


if __name__ == "__main__":
    main()
