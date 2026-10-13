# -*- coding: utf-8 -*-
"""无边框表检测：基于单词坐标的"列对齐启发式"。

背景（真实踩坑）：
- pdfplumber 的 find_tables() 靠检测表格边框线工作；
- 学术论文的数值对比表（PSNR/SSIM、排名表等）几乎都是**无边框文本对齐表**，
  实测 4 篇论文 18 个区域只有 1 个带边框表被抽到，4 张关键数值表全部漏掉；
- 这里用启发式补漏：表格行的特征 = 行内单词被明显水平间隙分隔成多个"列簇"，
  且多行连续、行距规则。

关键细节（都踩过坑）：
1. **先按栏拆分再检测**：双栏论文同一 top 的左右栏行会被合并成一行，
   中间的栏间隙 >8pt 会被当成"列间隙"，导致整页正文误报成表格；
2. **跨栏宽表再合并**：宽表横跨中线被劈成左右两半，按行 top 对齐合并还原；
3. 行内"列间隙"阈值 >8pt：正文单词间隙均匀 <3pt，表列间隙通常 >8pt；
4. 连续表格行按行距分组（<30pt）；组内行数 ≥3 才判定（过滤公式/图注碎片）。
"""
from dataclasses import dataclass


@dataclass
class AlignedTable:
    """一个无边框表区域（bbox 坐标与页面相同）。"""
    bbox: tuple          # (x0, top, x1, bottom)
    lines: list[str]     # 按阅读顺序的行文本（列用两个空格分隔）
    tops: list[float]    # 每行对应的 top 坐标
    n_cols_hint: int     # 出现过的最大列簇数（供 LLM 参考）


def _group_lines(words: list[dict], line_tol: float) -> list[list[dict]]:
    """把单词按 top 容差分组，组内按 x0 排序，返回行列表。"""
    words = sorted(words, key=lambda w: (round(w["top"] / line_tol), w["x0"]))
    rows: list[list[dict]] = []
    for w in words:
        if rows and abs(w["top"] - rows[-1][0]["top"]) < line_tol:
            rows[-1].append(w)
        else:
            rows.append([w])
    return rows


def _row_clusters(row: list[dict], gap_tol: float) -> list[list[dict]]:
    """把行内单词按水平间隙切成簇（列）。"""
    row = sorted(row, key=lambda w: w["x0"])
    clusters: list[list[dict]] = []
    cur = [row[0]]
    for w in row[1:]:
        if w["x0"] - cur[-1]["x1"] > gap_tol:
            clusters.append(cur)
            cur = [w]
        else:
            cur.append(w)
    clusters.append(cur)
    return clusters


def _detect_in_region(words: list[dict],
                      line_tol: float, gap_tol: float,
                      row_gap_tol: float, min_rows: int) -> list[AlignedTable]:
    """在给定单词集合上检测无边框表（不跨栏）。"""
    if not words:
        return []
    rows = _group_lines(words, line_tol)
    table_rows: list[list[dict]] = []
    hints: list[int] = []
    for row in rows:
        if len(row) < 2:
            continue
        clusters = _row_clusters(row, gap_tol)
        if len(clusters) >= 2:
            table_rows.append(row)
            hints.append(len(clusters))
    if not table_rows:
        return []

    groups: list[list[list[dict]]] = []
    for row in table_rows:
        if groups and abs(row[0]["top"] - groups[-1][-1][0]["top"]) < row_gap_tol:
            groups[-1].append(row)
        else:
            groups.append([row])

    tables = []
    for g in groups:
        if len(g) < min_rows:
            continue
        top = min(r[0]["top"] for r in g)
        bottom = max(r[-1]["bottom"] for r in g)
        x0 = min(w["x0"] for r in g for w in r)
        x1 = max(w["x1"] for r in g for w in r)
        lines, tops = [], []
        max_cols = 1
        for r in g:
            clusters = _row_clusters(r, gap_tol)
            max_cols = max(max_cols, len(clusters))
            lines.append("  ".join(" ".join(w["text"] for w in cl) for cl in clusters))
            tops.append(r[0]["top"])
        tables.append(AlignedTable(bbox=(x0, top, x1, bottom), lines=lines,
                                   tops=tops, n_cols_hint=max_cols))
    return tables


def _merge_pair(left: AlignedTable, right: AlignedTable,
                top_tol: float = 6.0) -> AlignedTable | None:
    """把左右两栏的半张表按行 top 对齐合并成一张表；无法对齐返回 None。"""
    # 行数太少无法可靠对齐
    if min(len(left.lines), len(right.lines)) < 2:
        return None
    used: set[int] = set()
    merged_tops, merged_lines = [], []
    for i, (ta, la) in enumerate(zip(left.tops, left.lines)):
        best_j, best_d = -1, 1e9
        for j, tb in enumerate(right.tops):
            if j in used:
                continue
            d = abs(ta - tb)
            if d < best_d:
                best_d, best_j = d, j
        if best_j >= 0 and best_d < top_tol:
            used.add(best_j)
            merged_tops.append(ta)
            merged_lines.append(la + "  " + right.lines[best_j])
        else:
            merged_tops.append(ta)
            merged_lines.append(la)
    for j, (tb, lb) in enumerate(zip(right.tops, right.lines)):
        if j not in used:
            merged_tops.append(tb)
            merged_lines.append(lb)

    order = sorted(range(len(merged_tops)), key=lambda k: merged_tops[k])
    lines = [merged_lines[k] for k in order]
    tops = [merged_tops[k] for k in order]
    bbox = (min(left.bbox[0], right.bbox[0]), min(left.bbox[1], right.bbox[1]),
            max(left.bbox[2], right.bbox[2]), max(left.bbox[3], right.bbox[3]))
    return AlignedTable(bbox=bbox, lines=lines, tops=tops,
                        n_cols_hint=left.n_cols_hint + right.n_cols_hint)


def _merge_across_columns(tables: list[AlignedTable]) -> list[AlignedTable]:
    """合并左右栏中 top 范围重叠的半张表（跨栏宽表）。"""
    if len(tables) < 2:
        return tables
    merged: list[AlignedTable] = []
    used: set[int] = set()
    for i, t in enumerate(tables):
        if i in used:
            continue
        cur = t
        # 找与 cur top 范围重叠的其他表（按 x 排序靠右的）
        for j, t2 in enumerate(tables):
            if j == i or j in used or t2.bbox[2] <= cur.bbox[0]:
                continue
            if (abs(cur.bbox[1] - t2.bbox[1]) < 10
                    and abs(cur.bbox[3] - t2.bbox[3]) < 10):
                left_t, right_t = sorted([cur, t2], key=lambda x: x.bbox[0])
                joined = _merge_pair(left_t, right_t)
                if joined:
                    cur = joined
                    used.add(j)
        merged.append(cur)
    return sorted(merged, key=lambda t: t.bbox[1])


def detect_aligned_tables(words: list[dict], page_width: float,
                          line_tol: float = 8.0,
                          gap_tol: float = 8.0,
                          row_gap_tol: float = 30.0,
                          min_rows: int = 3) -> list[AlignedTable]:
    """在页面单词坐标上检测无边框对齐表（分栏检测 + 跨栏合并）。

    Args:
        words: page.extract_words() 返回的单词列表。
        page_width: 页面宽度（pt），用于分栏。
        line_tol / gap_tol / row_gap_tol / min_rows: 见模块 docstring。

    Returns:
        AlignedTable 列表（按 top 排序）。
    """
    if not words:
        return []
    mid_x = page_width / 2.0
    left = [w for w in words if (w["x0"] + w["x1"]) / 2.0 < mid_x]
    right = [w for w in words if (w["x0"] + w["x1"]) / 2.0 >= mid_x]

    # 单栏页（右栏内容极少）直接整体检测；双栏页分左右检测后合并跨栏宽表
    if len(left) > 10 and len(right) > 10:
        tables = _detect_in_region(left, line_tol, gap_tol, row_gap_tol, min_rows)
        tables += _detect_in_region(right, line_tol, gap_tol, row_gap_tol, min_rows)
        tables = _merge_across_columns(tables)
    else:
        tables = _detect_in_region(words, line_tol, gap_tol, row_gap_tol, min_rows)
    return tables
