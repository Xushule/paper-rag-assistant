# -*- coding: utf-8 -*-
import pickle, re, sys
sys.path.insert(0, r'D:\论文阅读问答助手')
chunks = pickle.load(open(r'D:\论文阅读问答助手\data\chunks.pkl', 'rb'))
srcs = {}
for c in chunks:
    srcs[c.source] = srcs.get(c.source, 0) + 1
print('sources:', srcs)

# 全库搜数字
nums = ['8,655', '8655', '9,744', '9744', '5,442', '5442', '1,089', '1089', '4,713', '4713', '14,139', '15,186', '15186']
for c in chunks:
    low = c.content.lower()
    hit = [n for n in nums if n in low]
    if hit:
        print('--- %s p%d [%s] %s ---' % (c.source[:14], c.page, c.section, hit))
        for line in c.content.split('\n'):
            if any(n in line.lower() for n in nums):
                print('   |', line.strip()[:200])
        print()
