# -*- coding: utf-8 -*-
import pickle, sys
sys.path.insert(0, r'D:\论文阅读问答助手')
out = []
try:
    chunks = pickle.load(open(r'D:\论文阅读问答助手\data\chunks.pkl', 'rb'))
    out.append('chunks loaded: %d' % len(chunks))
    srcs = {}
    for c in chunks:
        srcs[c.source] = srcs.get(c.source, 0) + 1
    out.append('sources: %s' % srcs)
    nums = ['8,655', '8655', '9,744', '9744', '5,442', '5442', '1,089', '1089', '4,713', '4713', '14,139', '15,186', '15186', '729', '240', '731']
    for c in chunks:
        low = c.content.lower()
        hit = [n for n in nums if n in low]
        if hit:
            out.append('--- %s p%d [%s] %s ---' % (c.source[:16], c.page, c.section, hit))
            for line in c.content.split('\n'):
                if any(n in line.lower() for n in nums):
                    out.append('   | ' + line.strip()[:220])
            out.append('')
except Exception as e:
    out.append('ERROR: %s' % e)
open(r'D:\论文阅读问答助手\eval\_scope_out.txt', 'w', encoding='utf-8').write('\n'.join(out))
print('written', len(out))
