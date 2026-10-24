# -*- coding: utf-8 -*-
"""快速估算 faithfulness：复用 run_eval.run_ragas 对 5 条样本跑 ragas（约 2-3 分钟）"""
import json, sys
sys.path.insert(0, r'D:\论文阅读问答助手')
from eval.run_eval import SAMPLES_PATH, run_ragas

samples = json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))
want = ['rc-12', 'all-02', 'ntire-05', 'nr-07', 'all-05']
sub = [s for s in samples if s['id'] in want]
print('子集样本:', [s['id'] for s in sub], 'n_cit:', {s['id']: s['n_citations'] for s in sub})
res = run_ragas(sub)
print('faithfulness 子集:', res['scores'].get('faithfulness'), '| 跳过:', res['skipped_no_context'])
print('relevancy:', res['scores'].get('answer_relevancy'))
