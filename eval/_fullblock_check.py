# -*- coding: utf-8 -*-
"""按 chunk_id 查 4 条错题实际进入 prompt 的完整块内容"""
import json, pickle, sys
sys.path.insert(0, r'D:\论文阅读问答助手')
chunks = pickle.load(open(r'D:\论文阅读问答助手\data\chunks.pkl', 'rb'))
by_id = {c.chunk_id: c for c in chunks}
samples = json.loads(open(r'D:\论文阅读问答助手\eval\results\samples.json', encoding='utf-8').read())

checks = {
 'nr-01': ['global spatial', 'global temporal', 'spatio-temporal', 'noise map', 'noise', 'transformer'],
 'all-05': ['strrnet', 'miracle', 'polyrain', 'entrovision', 'entovision', 'restormer', 'transweather'],
 'all-06': ['google vision', 'vision api', 'downstream', 'recognition', 'api'],
 'all-07': ['sony', 'a6000', 'eos', 'synnightrain'],
}
out = []
for s in samples:
    if s['id'] not in checks:
        continue
    out.append('=== %s (n_cit=%d) ===' % (s['id'], s['n_citations']))
    for i, c in enumerate(s.get('citations', []) or [], 1):
        chunk = by_id.get(c.get('chunk_id'))
        if not chunk:
            out.append('  [%d] (chunk 未找到 %s)' % (i, c.get('chunk_id')))
            continue
        low = chunk.content.lower()
        tags = [kw for kw in checks[s['id']] if kw.lower() in low]
        out.append('  [%d] p%d [%s] %s' % (i, chunk.page, (chunk.section or '')[:16], ' '.join(tags) if tags else '-'))
        if tags:
            for kw in tags:
                pos = low.find(kw.lower())
                out.append('      %s @ %d: ...%s...' % (kw, pos, chunk.content[max(0, pos-40):pos+70].replace('\n', ' ')))
    out.append('')
open(r'D:\论文阅读问答助手\eval\_fullblock_check.txt', 'w', encoding='utf-8').write('\n'.join(out))
print('done')
