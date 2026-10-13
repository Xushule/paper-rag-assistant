# -*- coding: utf-8 -*-
"""BM25 关键词索引：与向量检索互补。

为什么需要 BM25（面试常问）：
- 向量检索抓"语义相似"，但遇到专有名词（模型名 STAEformer、指标名 MAPE），
  语义相近的词可能覆盖掉精确命中；
- BM25 是经典关键词打分，对专有名词、公式名、缩写命中更准；
- M2 会把两路结果合并再 Rerank，这里先把索引建好。

M1 用 rank_bm25（轻量、纯 Python）；数据量大了再考虑 Elasticsearch。
"""
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from ..config import BM25_PATH, CHUNK_META_PATH


def _tokenize(text: str) -> list[str]:
    """简单分词：英文按单词切，中文按单字切。

    数字与字母分开切："248Quadruplets" -> ["248", "quadruplets"]，
    否则会被 [a-zA-Z0-9]+ 切成单个 token "248quadruplets"，
    查询 "Quadruplets" 就匹配不上（真实踩坑）。
    说明：中文按单字切只是 M1 的简化方案，效果一般；
    后续可换 jieba 分词提升中文召回（M2 优化项）。
    """
    return re.findall(r"[a-zA-Z]+|[0-9]+|[\u4e00-\u9fff]", text.lower())


def build_index(chunks: list) -> BM25Okapi:
    """用 Chunk 列表构建 BM25 索引，并连同块元数据一起缓存到磁盘。

    Returns:
        BM25Okapi 索引对象。
    """
    tokenized = [_tokenize(c.content) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)
    with open(CHUNK_META_PATH, "wb") as f:
        pickle.dump(chunks, f)
    return bm25


def load_index() -> tuple[BM25Okapi, list]:
    """从磁盘加载 BM25 索引与块元数据。"""
    with open(BM25_PATH, "rb") as f:
        bm25 = pickle.load(f)
    with open(CHUNK_META_PATH, "rb") as f:
        chunks = pickle.load(f)
    return bm25, chunks


def search(bm25: BM25Okapi, chunks: list, query: str, k: int = 5) -> list:
    """返回 [(Chunk, 得分), ...]，按得分降序，得分 0 的丢弃。"""
    scores = bm25.get_scores(_tokenize(query))
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [(chunks[i], scores[i]) for i in top if scores[i] > 0]
