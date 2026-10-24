# -*- coding: utf-8 -*-
"""当前代码下 all-06/all-04 的实际检索 hits（含每篇保底）"""
import sys
sys.path.insert(0, r'D:\论文阅读问答助手')
from src.retrieval.embed_store import VectorStore
from src.retrieval.bm25_index import load_index
from src.qa.answer import QASystem

store = VectorStore()
bm25, chunks = load_index()
qa = QASystem(store, bm25, chunks)

cases = [
    ('all-04', '哪些论文提出了方法，哪些是数据集？', None),
    ('all-06', '这些论文都用了什么评测指标？', None),
]
out = []
for sid, q, scope in cases:
    out.append('=== %s ===' % sid)
    hits = qa._retrieve(q, top_k=16, scope=scope)
    for i, (c, _s) in enumerate(hits, 1):
        low = c.content.lower()
        tags = []
        for kw in ('google vision', 'vision api', 'sony', 'a6000', 'eos', 'synnightrain',
                   'strrnet', 'miracle', 'baseline', 'id, rdiffusion', 'propose', 'dataset'):
            if kw.lower() in low:
                tags.append(kw)
        out.append('  %02d %s p%d [%s] %s' % (i, c.source[:12], c.page, (c.section or '')[:16], ' '.join(tags) if tags else '-'))
    out.append('')
open(r'D:\论文阅读问答助手\eval\_retrieve_probe.txt', 'w', encoding='utf-8').write('\n'.join(out))
print('done')
