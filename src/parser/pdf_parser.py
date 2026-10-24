# -*- coding: utf-8 -*-
"""PDF 文本解析：按物理列重排双栏论文，输出清洗后的文本页。

为什么用 pdfplumber + 单词坐标（经过真实论文踩坑）：
1) pypdf 的 extract_text 对双栏论文（ECCV/CVPR/arXiv 常见排版）会把左右两栏文字
   按行交错输出，导致正文顺序错乱、检索质量崩塌；
2) pdfplumber 的 extract_text_lines 在部分 PDF 上会丢失单词间空格
   （内容流里本就没有空格字符，例如 "learning-basedmethodshavebeenproposed"），
   所以这里改用 extract_words() 按坐标切词，再自行用空格拼接，保证词边界；
3) 拿到单词坐标后，按"x 坐标分左右栏、栏内按行排序"重组，恢复真实阅读顺序。

单栏论文（如普通期刊）也能正确识别：判断依据是"右侧栏是否真的从页面上部开始"，
不满足则按单栏处理，不影响现有单栏文档。
"""
import re
from dataclasses import dataclass
from pathlib import Path

import pdfplumber

from ..config import TABLE_LLM

# 一页里右侧栏起始线的最低 y 位置（单位：pt）。
# 若右侧栏第一行非常靠下（比如只有脚注），说明不是双栏正文，按单栏处理。
TWO_COL_RIGHT_TOP_LIMIT = 300.0
# 同一行内单词的 top 容差（pt）：top 相差小于该值的单词归为同一行
LINE_TOP_TOLERANCE = 8.0


@dataclass
class PageText:
    """一页论文文本。"""
    page_no: int
    text: str


def _is_ascii(s: str) -> bool:
    return bool(s) and s.isascii()


def _table_to_markdown(table) -> str | None:
    """把 pdfplumber Table 转成 Markdown 表格文本。

    过滤"伪表格"：结构图/架构图常被 find_tables() 误判成表格
    （只有 1~2 行标签碎片），行数 < 3 或列数 < 2 的一律跳过，
    让该区域文字继续走普通文本流，避免生成垃圾块。

    Returns:
        Markdown 表格文本；伪表格返回 None。
    """
    rows = table.extract()
    if not rows or len(rows) < 3 or len(rows[0]) < 2:
        return None

    def clean(cell):
        if cell is None:
            return ""
        return " ".join(cell.replace("\n", " ").split())

    clean_rows = [[clean(c) for c in row] for row in rows]
    # 数据行里至少要有 2 行是"像样"的（≥2 个非空单元格），否则视为碎片
    solid = sum(1 for row in clean_rows[1:] if sum(1 for c in row if c) >= 2)
    if solid < 2:
        return None

    lines = ["| " + " | ".join(clean_rows[0]) + " |"]
    lines.append("| " + " | ".join("---" for _ in clean_rows[0]) + " |")
    for row in clean_rows[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _word_in_table(word: dict, bbox: tuple) -> bool:
    """判断一个单词是否落在表格包围盒 bbox 内（带小容差）。"""
    tol = 2.0
    return (word["x0"] >= bbox[0] - tol and word["x1"] <= bbox[2] + tol
            and word["top"] >= bbox[1] - tol and word["bottom"] <= bbox[3] + tol)


def _bbox_overlap(a: tuple, b: tuple) -> bool:
    """判断两个 bbox 是否垂直高度大量重叠（用于避免与边框表重复检测）。"""
    ov_top = max(a[1], b[1])
    ov_bottom = min(a[3], b[3])
    if ov_bottom <= ov_top:
        return False
    ov_h = ov_bottom - ov_top
    min_h = min(a[3] - a[1], b[3] - b[1])
    return ov_h / min_h > 0.5


def _is_reference_block(lines: list[str]) -> bool:
    """粗筛：候选行过半以 [n] 开头（参考文献条目）则跳过。"""
    if not lines:
        return False
    n = sum(1 for ln in lines if re.match(r"^\s*\[\s*\d+\s*\]", ln))
    return n / len(lines) >= 0.5


def _words_to_lines(words: list[dict]) -> list[dict]:
    """把 pdfplumber 单词列表按坐标组装成行。

    返回 [{"top": float, "x0": float, "x1": float, "text": str}, ...]
    同一行内：ASCII 单词之间补一个空格；中文等非 ASCII 字符之间不加空格。
    """
    if not words:
        return []

    def sort_key(w):
        return (round(w["top"] / LINE_TOP_TOLERANCE), w["x0"])

    words = sorted(words, key=sort_key)

    lines: list[dict] = []
    for w in words:
        if lines and abs(w["top"] - lines[-1]["top"]) < LINE_TOP_TOLERANCE:
            text = lines[-1]["text"]
            if text and _is_ascii(text[-1]) and _is_ascii(w["text"][:1]):
                text += " "
            lines[-1]["text"] = text + w["text"]
            lines[-1]["x1"] = max(lines[-1]["x1"], w["x1"])
        else:
            lines.append({"top": w["top"], "x0": w["x0"], "x1": w["x1"], "text": w["text"]})
    return lines


def _clean_page_text(raw: str) -> str:
    """清理单页原始文本：去页码/页眉碎片、修复断行、压缩空白。"""
    lines = []
    for line in raw.splitlines():
        stripped = line.strip()
        if re.fullmatch(r"\d{1,4}", stripped):   # 孤立数字 = 页码
            continue
        if len(stripped) < 2:                     # 过短噪声行
            continue
        lines.append(stripped)

    text = "\n".join(lines)
    text = re.sub(r"-\s*\n\s*", "", text)          # 断行连字符修复
    text = re.sub(r"[ \t]+", " ", text)            # 压缩空白
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def _reorder_two_column(lines: list[dict], page_width: float) -> list[dict]:
    """把带坐标的行重排为阅读顺序。

    lines: _words_to_lines 返回的行列表（含 text/top/x0/x1）。
    规则：
    1) 以页面中线分左右两栏；
    2) 两栏都从页面顶部开始正文 → 判定为双栏，输出"左栏全 + 右栏全"；
    3) 否则按单栏（所有行按 top 排序）输出。

    Returns:
        按阅读顺序排列的行 dict 列表（含 text/top），供表格按位置插入。
    """
    if not lines:
        return []

    mid_x = page_width / 2.0
    left = [ln for ln in lines if (ln["x0"] + ln["x1"]) / 2.0 < mid_x]
    right = [ln for ln in lines if (ln["x0"] + ln["x1"]) / 2.0 >= mid_x]

    def by_top(items):
        return sorted(items, key=lambda ln: ln["top"])

    left, right = by_top(left), by_top(right)

    # 双栏判定：左右两栏都有足够内容，且右栏确实从页面上部开始
    is_two_col = (
        len(left) > 3 and len(right) > 3
        and right[0]["top"] < TWO_COL_RIGHT_TOP_LIMIT
    )

    if is_two_col:
        return left + right
    return by_top(lines)


def extract_pdf_text(pdf_path: str | Path) -> list[PageText]:
    """读取 PDF，返回按阅读顺序重排并清洗后的页面列表。

    表格处理（M2 增强）：
    - 用 find_tables() 定位**带边框表**，区域内单词不再走文本平铺
      （否则行列关系丢失，出现 "I1 I2 In I1 I2 In" 式碎片）；
      表格用 extract_table() 抽出行列，转成 Markdown 按 top 插回正文流；
    - **无边框表补漏（M2.8+）**：学术数值表（PSNR/SSIM、排名表）几乎都是
      无边框对齐表，find_tables() 检测不到。用列对齐启发式定位候选
      （src/parser/table_align.py）；**首选坐标级单元格切分**
      （src/parser/table_cells.py，按单词坐标切出行×列网格，离线确定性），
      切分不合格的候选再交给 LLM 判定 + 转 Markdown
      （src/parser/table_llm.py）。LLM 确认的真表区域同样剔除单词、插回正文流。
      误报（结构图/作者/公式/参考文献）由 LLM 兜底为 NOT_TABLE，走文本流。
    - TABLE_LLM 开关关闭时仅启用坐标切分补漏（无 API Key 环境仍可离线处理
      规整对齐表）；LLM 兜底路径需要 TABLE_LLM 开启。
    """
    pdf_path = Path(pdf_path)
    pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            tables = page.find_tables() or []
            # 只保留"真表格"（能转成 Markdown 的），伪表格区域照常走文本流
            real_tables = [(t, _table_to_markdown(t)) for t in tables]
            real_tables = [(t, md) for t, md in real_tables if md is not None]

            raw_words = page.extract_words(x_tolerance=1.5, y_tolerance=3) or []
            # 剔除真表格区域内的单词，避免行列平铺碎片
            if real_tables:
                raw_words = [w for w in raw_words
                             if not any(_word_in_table(w, t.bbox) for t, _ in real_tables)]

            # 无边框表补漏：候选定位 -> 粗筛 -> 坐标切分 -> LLM 兜底
            # 第一道用坐标级单元格切分（src/parser/table_cells.py，离线确定性）：
            #   行容差 3pt / 列区间重叠合并 / 间隙 6pt 切格，直接产出 Markdown，
            #   不依赖 LLM，且能处理"一行塞两队"这类 LLM 照单全收的错位；
            # 切分不合格（返回 None）或 LLM 开关开启时，回退原 LLM 判定路径。
            llm_tables = []  # [(bbox, markdown)]
            if raw_words:
                from .table_align import detect_aligned_tables
                from .table_cells import cells_to_markdown
                candidates = detect_aligned_tables(raw_words, page.width)
                kept = []
                for cand in candidates:
                    if _is_reference_block(cand.lines):
                        continue
                    if real_tables and any(_bbox_overlap(cand.bbox, t.bbox) for t, _ in real_tables):
                        continue
                    kept.append(cand)

                # 候选区域内单词（按 bbox 截取，含 2pt 容差）
                for cand in kept:
                    region = [w for w in raw_words
                              if w["x0"] >= cand.bbox[0] - 2 and w["x1"] <= cand.bbox[2] + 2
                              and w["top"] >= cand.bbox[1] - 2 and w["bottom"] <= cand.bbox[3] + 2]
                    md = cells_to_markdown(region)
                    if md:
                        llm_tables.append((cand.bbox, md))

                # 坐标切分失败/不合格的候选，交给 LLM 判定 + 结构化
                if TABLE_LLM:
                    from .table_llm import structure_candidates
                    llm_cands = [cand for cand in kept
                                 if not any(bb == cand.bbox for bb, _ in llm_tables)]
                    # 公式区不进 LLM 兜底：扩散模型方程（√ α ϵ 等符号密集）
                    # 常被 LLM 误判为 TABLE 污染检索；数学符号密度高的候选直接
                    # 走文本流，与 cells_to_markdown 的符号规则保持一致。
                    def _math_dense(cand) -> bool:
                        texts = [w["text"] for w in raw_words
                                 if cand.bbox[0] - 2 <= w["x0"] <= cand.bbox[2] + 2
                                 and cand.bbox[1] - 2 <= w["top"] <= cand.bbox[3] + 2]
                        if len(texts) < 4:
                            return False
                        sym = sum(1 for t in texts if re.search(r"[√∑∫∈αβγϵ¯·×]", t))
                        return sym / len(texts) >= 0.3
                    llm_cands = [c for c in llm_cands if not _math_dense(c)]
                    if llm_cands:
                        mds = structure_candidates([c.lines for c in llm_cands])
                        for cand, md in zip(llm_cands, mds):
                            if md:
                                llm_tables.append((cand.bbox, md))

                if llm_tables:
                    raw_words = [w for w in raw_words
                                 if not any(_word_in_table(w, bbox) for bbox, _ in llm_tables)]

            lines = _words_to_lines(raw_words)
            ordered = _reorder_two_column(lines, page.width)

            # 表格按 top 坐标插入正文流（表格通常位于某栏中部，
            # 按位置插回可保持"表在哪一段"的上下文）
            blocks = list(ordered)
            for t, md in real_tables:
                blocks.append({"top": t.bbox[1], "text": md})
            for bbox, md in llm_tables:
                blocks.append({"top": bbox[1], "text": md})
            blocks.sort(key=lambda b: b["top"])

            text = _clean_page_text("\n".join(b["text"] for b in blocks))
            if text:
                pages.append(PageText(page_no=i, text=text))
    return pages
