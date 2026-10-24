# -*- coding: utf-8 -*-
"""查 AGAN p8 各块的 ref_noise 命中数，定位 Google Vision 块为何被过滤"""
import sys, pickle, re
sys.path.insert(0, r'D:\论文阅读问答助手')
chunks = pickle.load(open(r'D:\论文阅读问答助手\data\chunks.pkl', 'rb'))
ref_noise = re.compile(
    r"(in proceedings|in european conference|in ieee|in international "
    r"conference|arxiv preprint|pages \d)", re.IGNORECASE)
out = []
for c in chunks:
    if c.source != 'AGAN_Attentive_Raindrop_1711.10098.pdf':
        continue
    low = c.content.lower()
    n = len(ref_noise.findall(c.content))
    gv = 'GOOGLEVISION' if ('google vision' in low or 'vision api' in low) else ''
    out.append('p%d [%s] refs=%d %s' % (c.page, (c.section or '')[:22], n, gv))
open(r'D:\论文阅读问答助手\eval\_agan_refnoise.txt', 'w', encoding='utf-8').write('\n'.join(out))
print('done')
