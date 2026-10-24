# -*- coding: utf-8 -*-
"""诊断：模拟 nr-02 的完整精排链路，看 PRETRAIN 块为何被挤出"""
import sys, pickle
sys.path.insert(0, r'D:\论文阅读问答助手')
from src.retrieval.embed_store import VectorStore
from src.retrieval.hybrid import hybrid_search
from src.qa.query_rewriter import expand_query
from src.retrieval.reranker import rerank, MAX_CHARS

store = VectorStore()
bm25 = pickle.load(open(r'D:\论文阅读问答助手\data\bm25.pkl', 'rb'))
chunks = pickle.load(open(r'D:\论文阅读问答助手\data\chunks.pkl', 'rb'))

q = 'NightRain 由哪几个部分组成'
scope = 'NightRain_Nighttime_Deraining_2401.00729.pdf'
hits = hybrid_search(store, bm25, chunks, query=q, bm25_query=expand_query(q),
                     vector_k=16, bm25_k=16, final_k=20, scope=scope)
print('候选 20 块中带 PRETRAIN 的:')
for i, (c, _s) in enumerate(hits):
    low = c.content.lower()
    pos = low.find('pretrain')
    if pos != -1:
        print('  idx=%d p%d [%s] pretrain@%d (300字符内:%s)' % (i, c.page, (c.section or '')[:16], pos, pos < 300))
        print('    前300字: %s...' % c.content[:150].replace(chr(10), ' '))
        print('    pretrain 上下文: %s' % c.content[max(0, pos-60):pos+90].replace(chr(10), ' '))

docs = [c.content for c, _ in hits]
print('\nMAX_CHARS =', MAX_CHARS)
ranked = rerank(q, docs, top_k=12)
print('\n精排 top12:')
for rank, (idx, score) in enumerate(ranked, 1):
    c = hits[idx][0]
    low = c.content.lower()
    tags = []
    if 'pretrain' in low: tags.append('PRETRAIN')
    if 'synnightrain' in low: tags.append('SYNNIGHT')
    if 'adaptive-rain-removal' in low: tags.append('RAIN-REMOVAL')
    if 'adaptive-correction' in low: tags.append('CORRECTION')
    print('  %02d score=%4.1f idx=%02d p%d [%s] %s' % (rank, score, idx, c.page, (c.section or '')[:18], ' '.join(tags)))
