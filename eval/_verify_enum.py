# -*- coding: utf-8 -*-
"""验证 enum 五路 per_paper（问题4 + 其余2）是否含 AGAN-20 / NTIRE-17 / NightRain 数据集块。"""
import pickle, sys, re
sys.path.insert(0, r"D:\论文阅读问答助手")
from src.retrieval.embed_store import VectorStore
from src.retrieval.bm25_index import build_index
from src.retrieval.hybrid import hybrid_search

DATA = r"D:\论文阅读问答助手\data"
chunks = pickle.load(open(DATA + r"\chunks.pkl", "rb"))
store = VectorStore(persist_dir=DATA + r"\chroma_db_v2")
bm25 = build_index(chunks)

ref_noise = re.compile(
    r"(in proceedings|in european conference|in ieee|in international "
    r"conference|arxiv preprint|pages \d)", re.IGNORECASE)
inline_ref = re.compile(
    r"\[\d+\]\s+[A-Z][\w.]{2,}\s*,?\s*[A-Z]\.?"
    r"|(19|20)\d\d\.\s+[A-Z]"
    r"|in (visual media|intelligent robots|proceedings|international|"
    r"european|ieee|cvpr|iccv|eccv|neurips|acm|springer)\b",
    re.IGNORECASE)

def _is_ref_block(content: str) -> bool:
    lines = [l.strip() for l in content.splitlines() if l.strip()]
    if not lines:
        return False
    if len(lines) >= 3:
        ref_hits = sum(1 for l in lines if ref_noise.search(l))
        return ref_hits >= max(2, int(len(lines) * 0.6))
    return len(inline_ref.findall(lines[0])) >= 2

def enum_per_paper(question):
    probes = [
        question,
        "这篇论文提出的数据集、数据收集方式或评测基准是什么",
        "这篇论文提出的方法、网络结构或算法流程是什么",
        "这篇论文的实验设置、定量结果和主要结论是什么",
        "这篇论文的数据集是怎么收集或构建的，用了什么拍摄设备或合成方法",
    ]
    caps = [8, 2, 2, 2, 3]
    order = [0, 4, 1, 2, 3]
    out = {}
    for src in sorted({c.source for c in chunks}):
        lane = []
        for p in probes:
            ph = hybrid_search(store, bm25, chunks, p, bm25_query=p,
                               vector_k=12, bm25_k=12, final_k=12, scope=src)
            lane.append([(c, s) for c, s in ph if not _is_ref_block(c.content)])
        merged = []
        for pi in order:
            for i in range(caps[pi]):
                if i < len(lane[pi]):
                    c, s = lane[pi][i]
                    if not any(x[0].chunk_id == c.chunk_id for x in merged):
                        merged.append((c, s))
        out[src.split("_")[0]] = [c.chunk_id for c, _ in merged]
    return out

for q, tag, wants in [
    ("这些论文中哪些提出了新的数据集？", "all-17",
     ["AGAN_Attentive_Raindrop_1711.10098-20", "NightRain_Nighttime_Deraining_2401.00729-1"]),
    ("NTIRE 报告里有哪些基于 Restormer 的方法？", "all-18",
     ["NTIRE2025_DayNight_Raindrop_2504.12711-17", "NTIRE2025_DayNight_Raindrop_2504.12711-13"]),
]:
    print("###", tag)
    pp = enum_per_paper(q)
    for k, v in pp.items():
        mark = ""
        hits = [w.split("-")[-1] for w in wants if any(w in x for x in v)]
        if hits:
            mark = "  <== 命中 " + ",".join(hits)
        print("  ", k, "->", v[:10], mark)
