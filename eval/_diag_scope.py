# -*- coding: utf-8 -*-
"""查证 ntire-06 双口径的真实数字关系（挑战赛正文 vs RC 数据集论文）"""
import pickle, sys
sys.path.insert(0, r'D:\论文阅读问答助手')
chunks = pickle.load(open(r'D:\论文阅读问答助手\data\chunks.pkl', 'rb'))

def show(src_kw, kws, title):
    print('=' * 20, title, '=' * 20)
    seen = set()
    for c in chunks:
        if src_kw not in c.source.lower(): continue
        low = c.content.lower()
        if any(kw in low for kw in kws):
            key = c.chunk_id
            if key in seen: continue
            seen.add(key)
            print('--- %s p%d [%s] ---' % (c.source[:14], c.page, c.section))
            # 打印含关键数字的句子
            for line in c.content.split('\n'):
                ll = line.lower()
                if any(kw in ll for kw in kws):
                    print('   |', line.strip()[:220])
            print()

show('raindrop_clarity', ['8,655', '8655', '9,744', '9744', '5,442', '5442', '1,089', '1089', 'training set', '15,186', '15186'],
     'RC 数据集论文：白天/夜间构成')
show('ntire', ['4,713', '4713', '14,139', '240', '731', 'training set', 'exactly the same'],
     'NTIRE 挑战赛：发布划分')
