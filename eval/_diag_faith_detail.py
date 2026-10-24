# -*- coding: utf-8 -*-
"""按句切分答案，用核心短语在上下文中做包含检查，定位无依据断言。"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_ID = sys.argv[1] if len(sys.argv) > 1 else "all-03"

samples = json.loads((ROOT / "eval/results/samples.json").read_text(encoding="utf-8"))
s = next(x for x in samples if x["id"] == SAMPLE_ID)
ctx_all = " ".join(s.get("retrieved_contexts", []))

# 去掉标记语法后按句拆
text = re.sub(r"\*\*|\[?\d+\]?", "", s["answer"])
sentences = [x.strip() for x in re.split(r"[。；\n]", text) if len(x.strip()) > 12]

# 每句取核心词（去停用词后的名词/数字短语）
STOP = set("这些那些什么如何并且以及或者但是因为所以对于通过使用其中我们他们其它的一个该在在上中下是了的不与和及").union(set("，。、；：！？"))
CORE_RE = re.compile(r"[A-Za-z][\w\-\.]{3,}|[\u4e00-\u9fff]{2,10}|\d[\d,\.]*[MKG]?")
def core_terms(sent: str, ctx: str) -> tuple[list[str], list[str]]:
    terms = []
    for m in CORE_RE.finditer(sent):
        t = m.group(0)
        if t.lower() in ("这些", "那些", "什么", "如何", "并且", "或者", "其中", "对于", "通过", "使用", "我们的", "他们的", "一个", "这个", "这样的"):
            continue
        terms.append(t)
    missed = [t for t in terms if t.lower() not in ctx.lower()]
    return terms, missed

print(f"### {SAMPLE_ID}: {s['question']}")
print(f"上下文总长 {len(ctx_all)} 字符\n")
for i, sent in enumerate(sentences, 1):
    terms, missed = core_terms(sent, ctx_all)
    flag = "OK" if not missed else f"MISS({len(missed)}): " + ",".join(missed[:6])
    print(f"[{i}] {flag}")
    print("    ", sent[:110])
