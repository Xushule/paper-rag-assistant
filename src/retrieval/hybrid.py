# -*- coding: utf-8 -*-
"""混合检索：向量检索 × BM25 的 RRF 融合。

为什么混合（面试常问）：
- 向量检索擅长"语义相近"：问"夜间雨滴为什么难去除"，能召回讲 blur / night 的段落；
- BM25 擅长"精确关键词"：问"PSNR 是多少"，能精确命中含 PSNR 的段落；
- 单独用任何一路都会漏召回，RRF 把两路的"排名"融合，稳、快、无需调权重。

RRF（Reciprocal Rank Fusion）：
    score(chunk) = Σ_retriever  1 / (k + rank(chunk))
只依赖"排名"不依赖"分数"，所以两路分数尺度不同也没关系（k 一般取 60）。
"""
from .bm25_index import search as bm25_search

RRF_K = 60  # RRF 经典常数


def _id2chunk(chunks: list) -> dict:
    return {c.chunk_id: c for c in chunks}


def hybrid_search(store, bm25, chunks: list, query: str,
                  bm25_query: str | None = None,
                  vector_k: int = 10, bm25_k: int = 10,
                  final_k: int = 5, scope: str | None = None,
                  use_bm25: bool = True) -> list:
    """混合检索：返回 [(Chunk, rrf_score), ...] 按融合分降序。

    Args:
        store: VectorStore 实例（向量路）
        bm25:  BM25 索引对象（关键词路）
        chunks: 全部 Chunk 列表（与入库顺序一致）
        query: 用户原问题（向量路直接用，bge-m3 支持中英混合）
        bm25_query: 给 BM25 路用的改写后查询（默认与原问题相同）
        vector_k / bm25_k: 两路各自先取多少候选
        final_k: 融合后返回多少
        scope: 论文作用域（PDF 文件名）。不为 None 时，两路候选都
            只保留属于该论文的块，防止"多篇论文串台"。
        use_bm25: 消融开关。False 时跳过 BM25 路（RRF 退化为纯向量排名），
            用于"仅向量（无 RRF 混合）"对比组。
    """
    mapping = _id2chunk(chunks)

    # ---- 向量路：按排名给分（scope 时在向量库层做 metadata 过滤）----
    where = {"source": scope} if scope else None
    res = store.search(query, k=vector_k, where=where)
    vector_ids = res["ids"][0]

    # ---- BM25 路：按排名给分（scope 时检索后过滤非本论文的块）----
    bm25_hits: list = []
    if use_bm25:
        bm25_hits = bm25_search(bm25, chunks, bm25_query or query, k=bm25_k)
    if scope:
        vector_ids = [cid for cid in vector_ids
                      if cid in mapping and mapping[cid].source == scope]
        bm25_hits = [(c, s) for c, s in bm25_hits if c.source == scope]

    rrf: dict[str, float] = {}
    for rank, cid in enumerate(vector_ids, start=1):
        if cid in mapping:
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (RRF_K + rank)
    for rank, (chunk, _score) in enumerate(bm25_hits, start=1):
        rrf[chunk.chunk_id] = rrf.get(chunk.chunk_id, 0.0) + 1.0 / (RRF_K + rank)

    ranked = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)[:final_k]
    return [(mapping[cid], score) for cid, score in ranked]
