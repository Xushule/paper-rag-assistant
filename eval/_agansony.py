# -*- coding: utf-8 -*-
"""查 AGAN 分篇检索里 Sony 块的内容与排名"""
import sys, pickle
sys.path.insert(0, r'D:\论文阅读问答助手')
from src.retrieval.embed_store import VectorStore
from src.retrieval.hybrid import hybrid_search
from src.qa.query_rewriter import expand_query

store = VectorStore()
bm25 = pickle.load(open(r'D:\论文阅读问答助手\data\bm25.pkl', 'rb'))
chunks = pickle.load(open(r'D:\论文阅读问答助手\data\chunks.pkl', 'rb'))
q = '这些论文各自使用了什么数据集'
hits = hybrid_search(store, bm25, chunks, query=q, bm25_query=expand_query(q),
                     vector_k=8, bm25_k=8, final_k=8, scope='AGAN_Attentive_Raindrop_1711.10098.pdf')
out = []
for i, (c, _s) in enumerate(hits[:8], 1):
    low = c.content.lower()
    tag = ' <<<SONY' if ('sony' in low or 'a6000' in low or 'eos' in low) else ''
    out.append('--- %02d p%d [%s]%s ---' % (i, c.page, (c.section or '')[:20], tag))
    out.append(c.content[:300].replace('\n', ' '))
    out.append('')
open(r'D:\论文阅读问答助手\eval\_agansony.txt', 'w', encoding='utf-8').write('\n'.join(out))
print('done')
