# -*- coding: utf-8 -*-
"""确认 all-06 的 AGAN 保底块"""
import sys
sys.path.insert(0, r'D:\论文阅读问答助手')
from src.retrieval.embed_store import VectorStore
from src.retrieval.bm25_index import load_index
from src.qa.answer import QASystem

store = VectorStore()
bm25, chunks = load_index()
qa = QASystem(store, bm25, chunks)

hits = qa._retrieve('这些论文都用了什么评测指标？', top_k=16, scope=None)
out = []
for i, (c, _s) in enumerate(hits, 1):
    low = c.content.lower()
    tags = []
    for kw in ('google vision', 'vision api', 'recognition', 'conclusion',
               'sony', 'synnightrain', 'psnr', 'ssim', 'lpips'):
        if kw.lower() in low:
            tags.append(kw)
    out.append('  %02d %s p%d [%s] %s' % (i, c.source[:12], c.page, (c.section or '')[:18], ' '.join(tags) if tags else '-'))
    if 'google vision' in low:
        pos = low.find('google vision')
        out.append('      ...%s...' % c.content[max(0,pos-80):pos+120].replace('\n',' '))
open(r'D:\论文阅读问答助手\eval\_all06_probe.txt', 'w', encoding='utf-8').write('\n'.join(out))
print('done')
