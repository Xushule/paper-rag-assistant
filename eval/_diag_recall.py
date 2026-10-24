# -*- coding: utf-8 -*-
"""诊断：检查错题关键块是否在检索 top20 候选内（区分召回 vs 精排截断）"""
import sys, pickle
sys.path.insert(0, r'D:\论文阅读问答助手')
from src.retrieval.embed_store import VectorStore
from src.retrieval.hybrid import hybrid_search
from src.qa.query_rewriter import expand_query

store = VectorStore()
bm25 = pickle.load(open(r'D:\论文阅读问答助手\data\bm25.pkl', 'rb'))
chunks = pickle.load(open(r'D:\论文阅读问答助手\data\chunks.pkl', 'rb'))
print('chunks:', len(chunks))

def probe(q, scope=None, top=20):
    hits = hybrid_search(store, bm25, chunks, query=q, bm25_query=expand_query(q),
                         vector_k=30, bm25_k=30, final_k=top, scope=scope)
    print('=== Q:', q[:60], '| scope:', scope)
    for i, (c, _s) in enumerate(hits[:top], 1):
        low = c.content.lower()
        tag = ''
        if 'pretrain' in low: tag += ' <PRETRAIN>'
        if 'synnightrain' in low: tag += ' <SYNNIGHT>'
        if 'global spatial' in low or 'global temporal' in low: tag += ' <GLOBAL-ST>'
        if 'noise map' in low: tag += ' <NOISEMAP>'
        if 'clear video' in low or 'clear videos' in low: tag += ' <CLEARVIDEO>'
        if 'agreement' in low or 'generative adversarial' in low or 'attentive' in low: tag += ' <AGAN-METHOD>'
        if 'strrnet' in low: tag += ' <STRRNET>'
        if 'miracle' in low: tag += ' <MIRACLE>'
        print('  %02d [%s] p%-3d %-22s%s' % (i, c.source[:12], c.page, (c.section or '')[:22], tag))

NR = 'NightRain_Nighttime_Deraining_2401.00729.pdf'
probe('NightRain 由哪几个部分组成', scope=NR)
probe('NightRain 提出了什么任务和方法', scope=NR)
probe('NightRain 使用了什么数据集', scope=NR)
probe('这些论文里有哪些去雨方法', scope=None)
