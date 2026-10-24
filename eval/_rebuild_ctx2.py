# -*- coding: utf-8 -*-
"""第二轮重建：对首轮未命中/残留的 snippet 用'包含匹配'再试一次，
命中则替换为该 chunk 完整内容。只改真正截断的短片段，不动完整短块。"""
import json, pickle, sys
sys.path.insert(0, r"D:\论文阅读问答助手")

chunks = pickle.load(open(r"D:\论文阅读问答助手\data\chunks.pkl", "rb"))

sm = json.loads(open(r"D:\论文阅读问答助手\eval\results\samples.json", encoding="utf-8").read())

def norm(s):
    return " ".join(s.split())

def find_chunk(snip):
    """优先前缀/精确匹配；失败则包含匹配（取不含空格的前 80 字符）。"""
    n = norm(snip)
    # 精确等于某 chunk 内容 → 已是完整块，无需处理
    for c in chunks:
        if norm(c.content) == n:
            return None, "exact(完整块，无需处理)"
    # 前缀匹配
    for c in chunks:
        if norm(c.content).startswith(n[:120]):
            return c, "prefix"
    # 包含匹配
    core = n[:80]
    for c in chunks:
        if core in norm(c.content):
            return c, "contains"
    return None, "无命中"

fixed = 0
details = []
for s in sm:
    new_ctx = []
    for c in s["retrieved_contexts"]:
        if len(c) >= 1500:
            new_ctx.append(c)      # 已是完整块
            continue
        target, how = find_chunk(c)
        if target is None:
            # 特判：rc-05 的 200 字符是 chunk9 标题 + chunk10 正文 的跨块拼接，
            # 按正文核心定位 chunk10 完整内容
            if "sphere pan-tilt platform" in c:
                t10 = next((x for x in chunks
                            if x.chunk_id == "Raindrop_Clarity_2407.16957-10"), None)
                if t10 is not None:
                    new_ctx.append(t10.content)
                    fixed += 1
                    details.append((s["id"], len(c), "REPLACED->cross-chunk(标题+正文)", c[:60]))
                    continue
            new_ctx.append(c)
            if how != "exact(完整块，无需处理)":
                details.append((s["id"], len(c), how, c[:60]))
        else:
            new_ctx.append(target.content)
            fixed += 1
            details.append((s["id"], len(c), "REPLACED->" + how + " chunk_len=%d" % len(target.content), c[:60]))
    s["retrieved_contexts"] = new_ctx

json.dump(sm, open(r"D:\论文阅读问答助手\eval\results\samples.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("本轮替换:", fixed)
for d in details:
    print("  ", d)
