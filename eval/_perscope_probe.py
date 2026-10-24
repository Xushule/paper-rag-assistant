# -*- coding: utf-8 -*-
"""查 AGAN/NTIRE/NightRain 分篇检索 top8 是否含错题所需块"""
import sys, pickle
sys.path.insert(0, r'D:\论文阅读问答助手')
from src.retrieval.embed_store import VectorStore
from src.retrieval.hybrid import hybrid_search
from src.qa.query_rewriter import expand_query

store = VectorStore()
bm25 = pickle.load(open(r'D:\论文阅读问答助手\data\bm25.pkl', 'rb'))
chunks = pickle.load(open(r'D:\论文阅读问答助手\data\chunks.pkl', 'rb'))

out = []
def probe(q, scope, kws, title):
    out.append('=== %s ===' % title)
    hits = hybrid_search(store, bm25, chunks, query=q, bm25_query=expand_query(q),
                         vector_k=8, bm25_k=8, final_k=8, scope=scope)
    for i, (c, _s) in enumerate(hits[:8], 1):
        low = c.content.lower()
        tags = [kw for kw in kws if kw.lower() in low]
        out.append('  %02d p%d [%s] %s' % (i, c.page, (c.section or '')[:16], ' '.join(tags) if tags else '-'))
    out.append('')

# all-06: AGAN Google Vision 块
probe('这些论文都用了什么评测指标', 'AGAN_Attentive_Raindrop_1711.10098.pdf',
      ['google vision', 'vision api', 'downstream', 'recognition', 'psnr', 'ssim'], 'all-06 AGAN 分篇')
# all-07: AGAN 数据集块
probe('这些论文各自使用了什么数据集', 'AGAN_Attentive_Raindrop_1711.10098.pdf',
      ['sony', 'a6000', 'eos', 'dataset', 'collected', 'captured'], 'all-07 AGAN 分篇')
# all-07: NightRain SynNightRain 块
probe('这些论文各自使用了什么数据集', 'NightRain_Nighttime_Deraining_2401.00729.pdf',
      ['synnightrain', 'synthetic', 'nighttime', 'dataset'], 'all-07 NightRain 分篇')
# all-05: NTIRE STRRNet 块
probe('这些论文里有哪些去雨方法', 'NTIRE2025_DayNight_Raindrop_2504.12711.pdf',
      ['strrnet', 'miracle', 'restormer', 'transweather', 'method', 'proposes'], 'all-05 NTIRE 分篇')
open(r'D:\论文阅读问答助手\eval\_perscope_probe.txt', 'w', encoding='utf-8').write('\n'.join(out))
print('done')
