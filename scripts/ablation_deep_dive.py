# -*- coding: utf-8 -*-
"""消融结果深挖：分类准确率矩阵 + 证据样本（组件救回的题目）。

四组 samples 均已落盘（judge 完成），无需重跑评测。
输出：
1) 按问题类别（rc/agan/nr/ntire/all）的四组准确率矩阵
2) full 正确但 no_mmr / no_expand / vector_only 错误的样本清单（组件增量证据）
3) full 错误但某组正确的样本（噪声/回归证据，诚实呈现）
"""
import json
import sys
from pathlib import Path

RESULTS = Path("D:/论文阅读问答助手/eval/results")
OUT = Path("D:/论文阅读问答助手/eval/results/ablation_deep_dive.txt")
GROUPS = ["full", "no_mmr", "no_expand", "vector_only"]
CATS = ["rc", "agan", "nr", "ntire", "all"]
CAT_LABEL = {"rc": "数据集", "agan": "方法", "nr": "夜间去雨", "ntire": "挑战赛", "all": "全库对比/元"}


def load(tag):
    p = RESULTS / f"samples_{tag}.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


data = {t: load(t) for t in GROUPS}
by_id = {t: {s["id"]: s for s in data[t]} for t in GROUPS}

def emit(text):
    print(text)
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def main():
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("")
    emit("=" * 62)
    emit("① 分类准确率矩阵（组件价值按任务放大）")
    emit("=" * 62)
    emit(f"{'类别':<12}" + "".join(f"{t:>9}" for t in GROUPS))
    for c in CATS:
        row = []
        for t in GROUPS:
            s = by_id[t]
            items = {k: v for k, v in s.items() if k.startswith(c + "-")}
            corr = sum(1 for v in items.values() if v.get("judge", {}).get("correct"))
            row.append(f"{corr}/{len(items)}")
        emit(f"{CAT_LABEL[c]:<12}" + "".join(f"{r:>9}" for r in row))
    emit(f"{'总计':<12}" + "".join(
        f"{sum(1 for v in by_id[t].values() if v.get('judge',{}).get('correct'))}/100" for t in GROUPS))

    emit("")
    emit("=" * 62)
    emit("② 证据样本：full 对、但关掉组件后答错的题")
    emit("=" * 62)
    for comp_tag, comp_name in [("no_mmr", "关 MMR"), ("no_expand", "关查询扩展"), ("vector_only", "仅向量")]:
        evid = []
        for sid, s in by_id["full"].items():
            if not s.get("judge", {}).get("correct"):
                continue
            c = by_id[comp_tag].get(sid)
            if c and not c.get("judge", {}).get("correct"):
                evid.append(sid)
        emit(f"\n[{comp_name}] 被它救回的题 {len(evid)} 道: {evid[:12]}{'...' if len(evid) > 12 else ''}")

    emit("")
    emit("=" * 62)
    emit("③ 回归样本：full 错、但某组答对（诚实呈现 LLM 波动）")
    emit("=" * 62)
    for comp_tag, comp_name in [("no_mmr", "关 MMR"), ("no_expand", "关查询扩展"), ("vector_only", "仅向量")]:
        reg = []
        for sid, s in by_id["full"].items():
            if s.get("judge", {}).get("correct"):
                continue
            c = by_id[comp_tag].get(sid)
            if c and c.get("judge", {}).get("correct"):
                reg.append(sid)
        emit(f"[{comp_name}] full 错它对的题 {len(reg)} 道: {reg[:12]}{'...' if len(reg) > 12 else ''}")


if __name__ == "__main__":
    main()
