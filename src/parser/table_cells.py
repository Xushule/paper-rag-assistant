# -*- coding: utf-8 -*-
"""无边框表坐标级切分：把候选区域内的单词按 top/x0 坐标切成「行 × 列」网格，
恢复单元格边界，输出 Markdown。

为什么写这个（面试可讲）：
- table_align.py 只做「行内分簇」，各行各自为政，没有「全表共用的列边界」——
  把行文本交给 LLM 猜结构时，相邻两行（如 NTIRE 排名表 EntroVision / Miracle）
  可能被猜成一行，数字全部挤进同一单元格，取数错位；
- 坐标切分用 PDF 本来就有的单词坐标（x0/top）恢复网格：top 相同 = 同行，
  跨行中心 x 聚类 = 同列，行列交叉 = 单元格。结构由坐标客观决定，不再依赖 LLM 猜。

算法（4 步）：
1. 行聚类：top 容差（3pt）内的单词归为一行——表格行距远小于正文，
   8pt 会把相邻行（如 NTIRE 排名表 Miracle/EntroVision 仅差 7.7pt）并成一行；
2. 行内簇检测：水平间隙 > 6pt 的单词切为簇（列候选；间隙阈值比 table_align
   的 8pt 更严——实测 PSNR/SSIM 列间隙恰好 8.0pt，8pt 判断会放过）；
3. 列边界对齐：把全部行簇的「单词区间 [x0,x1]」跨行做区间重叠合并 →
   全局列边界（同一列跨行区间必然重叠、不同列之间留有间隙；
   比中心聚类更抗同一列内文本宽度参差导致的中心偏移）；
4. 网格化：每个 (行, 列) 交叉取 center 最近的簇单词 → 单元格文本 → Markdown。

伪表过滤（3 道，实测踩坑）：
- 数据行 ≥2 行「实心」（≥2 个非空单元格）；
- 平均填充率 ≥0.5（挡住稀疏标签区）；
- 数据格含数字占比 ≥0.25 且数学符号密度 <0.3
  （真表以数字为主；架构图标签无数字；扩散方程公式含下标数字但符号密集）。

失败回退：网格不合格（伪表/切分失败）返回 None，调用方回退 LLM 方案。
"""
from dataclasses import dataclass


@dataclass
class CellGrid:
    """切分出的表格网格。"""
    headers: list[str]
    rows: list[list[str]]
    n_cols: int


def _group_rows(words: list[dict], line_tol: float = 3.0) -> list[list[dict]]:
    """按 top 容差把单词聚成行（行内按 x0 排序）。

    line_tol 用 3.0 而不是正文的 8.0：表格行距远小于正文（实测 NTIRE 排名表
    相邻行 top 仅差 7.7pt），8pt 会把相邻两行并成一行——这正是
    "EntroVision Miracle 两行并一行"的直接根源；表格内同行单词 top 完全一致。
    """
    words = sorted(words, key=lambda w: (round(w["top"] / line_tol), w["x0"]))
    rows: list[list[dict]] = []
    for w in words:
        if rows and abs(w["top"] - rows[-1][0]["top"]) < line_tol:
            rows[-1].append(w)
        else:
            rows.append([w])
    return rows


def _row_clusters(row: list[dict], gap_tol: float = 6.0) -> list[list[dict]]:
    """行内按水平间隙切成簇（列候选）。

    gap_tol 用 6.0 而非 table_align 的 8.0：实测 NTIRE 排名表 PSNR 列与 SSIM
    列的间隙恰好 8.0pt，`> 8.0` 判断为 False 会把两列并成一格。
    """
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


def _cluster_center(cluster: list[dict]) -> float:
    """簇的中心 x（列定位用）：取簇内单词 x0 的中间值，受右对齐/左对齐影响小。"""
    xs = sorted(w["x0"] for w in cluster)
    return xs[len(xs) // 2]


def _global_columns(rows_clusters: list[list[list[dict]]],
                    overlap_tol: float = 4.0) -> list[tuple[float, float]]:
    """跨行合并簇区间 → 全局列边界（每列一个 [lo, hi] 区间，按 x 升序）。

    用区间重叠判定而非中心聚类：同一列跨行的单词区间必然重叠
    （Miracle 队名 [113.8,133.8] 与 EntroVision [108.3,139.3]），
    而不同列之间留有间隙（PSNR [306,327] 与 SSIM [335,353] 不重叠），
    重叠合并、有间隙的保持独立——比「中心 x 聚类」更抗同一列内
    文本宽度参差（右对齐数字列、长短不一的队名）导致的中心偏移。
    """
    intervals = []
    for row in rows_clusters:
        for cl in row:
            xs = [w["x0"] for w in cl]
            xe = [w["x1"] for w in cl]
            intervals.append((min(xs), max(xe)))
    intervals.sort()
    cols: list[tuple[float, float]] = []
    for lo, hi in intervals:
        if cols and lo <= cols[-1][1] + overlap_tol:
            cols[-1] = (cols[-1][0], max(cols[-1][1], hi))
        else:
            cols.append((lo, hi))
    return cols


def _cell_text(words: list[dict]) -> str:
    """单元格内单词按 x0 排序拼接（ASCII 之间补空格，中文不补）。"""
    words = sorted(words, key=lambda w: w["x0"])
    parts = []
    for w in words:
        t = w["text"]
        if parts and parts[-1][-1].isascii() and t[:1].isascii():
            parts[-1] += " " + t
        else:
            parts.append(t)
    return "".join(parts).strip()


def cells_to_markdown(region_words: list[dict],
                      line_tol: float = 3.0,
                      gap_tol: float = 6.0,
                      overlap_tol: float = 4.0,
                      min_rows: int = 3) -> str | None:
    """把候选区域的单词坐标切分成交叉网格并转 Markdown。

    Args:
        region_words: 候选区域内 page.extract_words() 的单词列表。

    Returns:
        Markdown 表格文本；网格不合格（伪表/切分失败）返回 None。
    """
    rows = _group_rows(region_words, line_tol)
    if len(rows) < min_rows:
        return None

    rows_clusters = [_row_clusters(r, gap_tol) for r in rows]
    cols = _global_columns(rows_clusters, overlap_tol)
    if len(cols) < 2:
        return None

    # 网格化：每行每个簇归属最近的列区间；同一列可并多个簇（格内换行等）
    col_centers = [(lo + hi) / 2 for lo, hi in cols]
    grid: list[list[str]] = []
    for clusters in rows_clusters:
        cells: list[list[dict]] = [[] for _ in cols]
        for c in clusters:
            cc = _cluster_center(c)
            best = min(range(len(cols)), key=lambda j: abs(col_centers[j] - cc))
            cells[best].extend(c)
        grid.append([_cell_text(cell) for cell in cells])

    # 伪表过滤：数据行至少 2 行「实心」（≥2 个非空单元格）+
    # 平均填充率达标 + 数据格含数字占比达标。
    # 为什么两道：填充率只能挡住"每行只有 1~2 个词的稀疏标签"，
    # 挡不住架构图/流程图标区（"| - | EMA | - | - | EMA | - |" 6 列每行
    # 填 2~6 格，填充率仍 ≥0.5）。真表（PSNR 排名表/统计表）数据格绝大多数
    # 是数字（占比 ~0.9）；公式（√、α、ϵ）与架构图标签区几乎无数字，一票否决。
    data_rows = grid[1:]
    solid = sum(1 for row in data_rows if sum(1 for c in row if c) >= 2)
    if solid < 2:
        return None
    n_cols = len(grid[0])
    fills = [sum(1 for c in row if c) / n_cols for row in data_rows]
    if sum(fills) / len(fills) < 0.5:
        return None
    import re as _re
    digit_cells = sum(1 for row in data_rows for c in row if _re.search(r"\d", c))
    total_cells = sum(1 for row in data_rows for c in row if c)
    if total_cells and digit_cells / total_cells < 0.25:
        return None
    # 数学符号密度一票否决：公式区（如扩散模型方程
    # "x =α¯t√t w | t"、"t−1 t−1α¯"）含下标数字会骗过数字规则
    # （"t−1" 里的 1 也算数字），但其单元格以数学符号（√ α ϵ ¯ · −）为主；
    # 真表（PSNR 排名表/统计表）数据格没有这类符号。
    math_cells = sum(1 for row in data_rows for c in row
                     if _re.search(r"[√∑∫∈αβγϵ¯·×−]", c))
    if total_cells and math_cells / total_cells >= 0.3:
        return None

    header, data = grid[0], grid[1:]
    lines = ["| " + " | ".join(h if h else "-" for h in header) + " |"]
    lines.append("| " + " | ".join("---" for _ in cols) + " |")
    for row in data:
        lines.append("| " + " | ".join(c if c else "-" for c in row) + " |")
    return "\n".join(lines)
