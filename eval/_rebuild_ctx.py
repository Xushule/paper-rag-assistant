# -*- coding: utf-8 -*-
"""重建 retrieved_contexts 为完整块内容（修复 faithfulness 测量口径）。

背景：ask_one 只把 citations 的 snippet（chunk.content[:400]）存入
samples.json 的 retrieved_contexts；ragas 的 faithfulness 用该字段判定
"回答是否有上下文支持"，但模型生成时看到的是完整块（约 1800 字符），
400 字符截断会系统性压低 faithfulness（0.394/0.4586 均是这个口径）。

本脚本：用 snippet 前缀在 chunks.pkl 中唯一匹配 chunk，替换为 chunk.content 全文；
匹配失败的保留原 snippet 并计数。不改 ask 链路，直接改 samples.json。
"""
import json, pickle, sys
sys.path.insert(0, r"D:\论文阅读问答助手")
from collections import Counter

SAMPLES = r"D:\论文阅读问答助手\eval\results\samples.json"
CHUNKS = r"D:\论文阅读问答助手\data\chunks.pkl"

chunks = pickle.load(open(CHUNKS, "rb"))
# content 前缀 -> chunk_id（snippet 是 content 开头 400 字符，startswith 精确匹配）
by_prefix = {}
for c in chunks:
    pref = c.content[:120].strip()
    if pref not in by_prefix:
        by_prefix[pref] = c
    # 若同一前缀对应多个 chunk（极少），保留第一个即可；后续用更长前缀再校验

samples = json.loads(open(SAMPLES, encoding="utf-8").read())
stats = Counter()
out = []
for s in samples:
    new_ctx = []
    for snip in s.get("retrieved_contexts", []):
        pref = snip[:120].strip()
        c = by_prefix.get(pref)
        # snippet 是 chunk.content 的开头（旧 200 / 现 400 字符均成立），
        # 命中前缀且 chunk 内容确实以 snippet 开头 → 替换为完整块
        if c is not None and c.content.startswith(snip.strip()[:200]):
            new_ctx.append(c.content)          # 完整块
            stats["matched"] += 1
        else:
            new_ctx.append(snip)               # 保留原 snippet
            stats["unmatched"] += 1
    s["retrieved_contexts"] = new_ctx
    out.append(s)

json.dump(out, open(SAMPLES, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("统计:", dict(stats))
print("总片段:", sum(len(s["retrieved_contexts"]) for s in out))
