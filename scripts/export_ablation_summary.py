# -*- coding: utf-8 -*-
"""生成消融 + 无 RAG 对照的完整对比表 CSV（eval/results/ablation_summary.csv）。"""
import csv
from pathlib import Path

RESULTS = Path("D:/论文阅读问答助手/eval/results")
rows = []

def load_summary(tag):
    p = RESULTS / f"eval_summary_{tag}.json"
    if not p.exists():
        return None
    import json
    return json.loads(p.read_text(encoding="utf-8"))

# 消融四组
groups = [
    ("full", "完整版：RRF 混合 + MMR + 查询扩展", "全开"),
    ("no_mmr", "关 MMR（RRF → 直接精排）", "MMR_ENABLED=0"),
    ("no_expand", "关查询扩展（单路检索）", "QUERY_EXPAND=0"),
    ("vector_only", "仅向量（无 BM25，RRF 退化）", "HYBRID_USE_BM25=0"),
    ("no_rag", "无 RAG：诚实模式（不确定就说不确定）", "--no-rag"),
    ("no_rag_default", "无 RAG：普通提示词（大模型默认行为）", "--no-rag-mode default"),
    ("no_rag_fab", "无 RAG：强制编造模式", "--no-rag --fabricate"),
]

header = ["组", "配置", "开关", "答案准确率", "faithfulness", "answer_relevancy", "context_precision", "context_recall"]
rows.append(header)
for tag, desc, sw in groups:
    s = load_summary(tag)
    if s is None:
        rows.append([tag, desc, sw, "未生成", "", "", "", ""])
        continue
    rows.append([
        tag, desc, sw,
        f"{s.get('answer_accuracy_pct', '')}%",
        s.get("faithfulness") if s.get("faithfulness") is not None else "—",
        s.get("answer_relevancy") if s.get("answer_relevancy") is not None else "—",
        s.get("context_precision") if s.get("context_precision") is not None else "—",
        s.get("context_recall") if s.get("context_recall") is not None else "—",
    ])

out = RESULTS / "ablation_summary.csv"
with open(out, "w", newline="", encoding="utf-8-sig") as f:
    csv.writer(f).writerows(rows)
print("written", out)
for r in rows:
    print(" | ".join(str(x) for x in r))
