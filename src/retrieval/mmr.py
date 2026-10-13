# -*- coding: utf-8 -*-
"""MMR（Maximal Marginal Relevance）多样性选片段。

为什么需要（跨论文对比的多样性问题）：
- 全库模式下，RRF 融合 + 精排后的 top-k 常被块数多的/相关性强的少数论文垄断，
  其他论文的代表片段被挤出候选（实测问"这里有几篇论文"时 NightRain 0 块进 top12）。
- MMR 每选一个片段时同时考虑"与问题的相关度"和"与已选片段的冗余度"：

      score = λ · Rel(q, d) − (1−λ) · max_{d'∈S} Sim(d, d')

  同篇论文的块语义相近（Sim 大），后选的被自动压低，不同论文的块自然顶上；
  它不需要预先知道库里有哪几篇论文（软多样性，区别于"按论文硬配额"）。

- 出处：Carbonell & Goldstein (1998)，信息检索多样性的经典方法；
  LangChain 的 VectorStoreRetriever search_type="mmr" 即此思路。
  本实现：相关度用 RRF 融合分数（min-max 归一化），相似度用 bge-m3 向量余弦
  （向量从 Chroma 直接取，不重复编码）。
"""
import numpy as np


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """两个已归一化向量的余弦相似度（点积）。"""
    return float(np.dot(a, b))


def mmr_select(cand_ids: list[str], embeddings: list[list[float]],
               rel_scores: list[float], k: int,
               lambda_: float = 0.7) -> list[str]:
    """从候选中用 MMR 选 k 个多样化的块 id（保持原始顺序）。

    Args:
        cand_ids: 候选 chunk_id 列表（与 embeddings / rel_scores 同序）。
        embeddings: 候选向量（未归一化也行，内部处理）。
        rel_scores: 候选与查询的相关度（RRF 融合分数），内部 min-max 归一化。
        k: 要选多少个。
        lambda_: 相关度权重（0~1）。越大越重相关度，越小越重多样性。

    Returns:
        选中的 chunk_id 列表（按 MMR 得分从高到低）。
    """
    n = len(cand_ids)
    if n <= k:
        return cand_ids
    if n == 0:
        return []

    # 1) 向量归一化（余弦相似度需要）
    vecs = []
    for v in embeddings:
        arr = np.asarray(v, dtype=np.float64)
        norm = np.linalg.norm(arr)
        vecs.append(arr / norm if norm > 0 else arr)
    V = np.stack(vecs)  # (n, dim)

    # 2) 相关度 min-max 归一化到 [0,1]
    lo, hi = min(rel_scores), max(rel_scores)
    rel = [(s - lo) / (hi - lo) if hi > lo else 1.0 for s in rel_scores]

    # 3) 迭代挑选：每轮选 score 最大的候选
    selected: list[int] = []
    remaining = list(range(n))
    while len(selected) < k and remaining:
        best_i, best_score = -1, -1e18
        for i in remaining:
            score = lambda_ * rel[i]
            if selected:
                sim_max = max(_cosine_sim(V[i], V[j]) for j in selected)
                score -= (1 - lambda_) * sim_max
            if score > best_score:
                best_score, best_i = score, i
        selected.append(best_i)
        remaining.remove(best_i)

    return [cand_ids[i] for i in selected]
