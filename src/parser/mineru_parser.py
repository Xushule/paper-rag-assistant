# -*- coding: utf-8 -*-
"""MinerU 解析产物适配：结构化 Markdown -> PageText，复用现有分块逻辑。

为什么这样设计（最小侵入，而不是重写一套分块器）：
1) MinerU 4.x 把 PDF 转成结构化 Markdown（标题 #、表格 |..|、公式 $..$、
   分页标记 <!-- page N of M -->），这些标记比 pdfplumber 的纯文本更可靠，
   尤其是无边框/合并单元格表格与 LaTeX 公式；
2) 但现有 chunker 的标题栈路径 / 父级标题补全 / 表格独立成块 / References
   防误判逻辑，都是为"纯文本行"设计的，功能已经过真实论文打磨，
   不值得为 MinerU 重写一遍；
3) 所以本适配器只做"标记语言 -> 行文本"的转换：
   - 去掉 `#` 前缀 -> 标题交给 chunker 的编号/白名单识别（层级由编号推断，
     与 pdfplumber 版行为一致）；
   - 保留 `|...|` 表格行 -> 表格打包逻辑原样复用（表头+分隔行+数据行）；
   - 跨行 `$$...$$` 公式块压成单行 `$latex$`，保证 LaTeX 可检索；
   - 图片块（表格被版面模型判成图片时）降级为"[图片块 第N页]"文本锚点，
     保留"此处有图/表"的语义位置；
   - 行内公式 / 正文行原样保留。
   转换后无缝走 split_into_sections，零重复代码、行为可预期。
"""
import re
from pathlib import Path

from .pdf_parser import PageText

# MinerU 分页标记："<!-- page 1 of 9 -->"
_PAGE_MARK = re.compile(r"^<!--\s*page\s+(\d+)\s+of\s+\d+\s*-->")
# 图片块行："![Image block](doc:<docid>/tier:.../page:6/block:3)"
_IMG_BLOCK = re.compile(r"^!\[[^\]]*\]\(doc:[^)]*\)")
# HTML 表格（MinerU 对带 rowspan/colspan 合并单元格的复杂表输出 HTML 而非 Markdown）
_HTML_TABLE = re.compile(r"<table[^>]*>.*?</table>", re.DOTALL)
_HTML_ROW = re.compile(r"<tr[^>]*>.*?</tr>", re.DOTALL)
_HTML_CELL = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.DOTALL)
_STRIP_TAG = re.compile(r"<[^>]+>")


def _html_table_to_md(html: str) -> str:
    """把 MinerU 的 HTML 表格转成 Markdown 表格行。

    rowspan/colspan 合并语义在检索视角不损失数值：每行按出现顺序输出
    各单元格内容（合并单元格的值保留在其首次出现的行）。表头行/分隔行
    由 chunker 的表格打包逻辑自行判断，这里只负责"单元格文本 -> | 行"。
    """
    out: list[str] = []
    for row in _HTML_ROW.findall(html):
        cells = [_STRIP_TAG.sub("", c).strip() for c in _HTML_CELL.findall(row)]
        if cells:
            out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def find_mineru_markdown(pdf_path: str | Path, out_dir: str | Path | None = None) -> Path | None:
    """按 PDF 文件名在 MinerU 产物目录里找对应 Markdown。

    查找顺序：精确 `{stem}.md` -> 前缀模糊 `{stem}*.md`
    （容忍 MinerU 输出文件名带额外后缀的情况）。
    """
    pdf_path = Path(pdf_path)
    if out_dir is None:
        from ..config import MINERU_OUT_DIR
        out_dir = MINERU_OUT_DIR
    out_dir = Path(out_dir)
    exact = out_dir / f"{pdf_path.stem}.md"
    if exact.is_file():
        return exact
    matches = sorted(out_dir.glob(f"{pdf_path.stem}*.md"))
    return matches[0] if matches else None


def _formula_block_to_latex(raw_lines: list[str], i: int) -> tuple[str, int]:
    """把跨行 $$...$$ 公式块合并成单行 LaTeX；返回 (latex行, 消费后的下标)。"""
    buf: list[str] = []
    first = raw_lines[i].strip()
    if len(first) > 2:  # "$$x$$" 单行公式
        return first.strip("$$").strip(), i + 1
    i += 1
    while i < len(raw_lines) and "$$" not in raw_lines[i]:
        buf.append(raw_lines[i].strip())
        i += 1
    if i < len(raw_lines):  # 收尾行（含 $$）
        buf.append(raw_lines[i].strip("$$").strip())
        i += 1
    latex = " ".join(x for x in buf if x)
    return latex, i


def _convert_page(text: str, page_no: int) -> str:
    """把一页 MinerU Markdown 转成 chunker 能直接消费的行文本。"""
    out: list[str] = []
    raw_lines = text.splitlines()
    i = 0
    html_buf: list[str] | None = None  # 跨行 HTML 表格收集器
    while i < len(raw_lines):
        line = raw_lines[i].strip()
        # 空行 / 注释（含分页标记，切页时已消费，这里兜底跳过）
        if not line or line.startswith("<!--"):
            i += 1
            continue
        # HTML 表格（MinerU 对合并单元格复杂表输出 HTML；可能跨行）-> Markdown 行
        if "<table" in line:
            if "</table>" in line:
                md = _html_table_to_md(line)
                if md:
                    out.extend(md.splitlines())
                i += 1
                continue
            html_buf = [line]
            i += 1
            while i < len(raw_lines) and "</table>" not in raw_lines[i]:
                html_buf.append(raw_lines[i])
                i += 1
            if i < len(raw_lines):
                html_buf.append(raw_lines[i])
                i += 1
            md = _html_table_to_md("\n".join(html_buf))
            if md:
                out.extend(md.splitlines())
            continue
        # 图片块 -> 语义锚点（保留"该处有图/表"的位置信息，去掉内部 doc id 噪声）
        if _IMG_BLOCK.match(line):
            out.append(f"[图片块 第{page_no}页]")
            i += 1
            continue
        # 公式块 $$...$$（可跨行）-> 单行 LaTeX
        if line.startswith("$$"):
            latex, i = _formula_block_to_latex(raw_lines, i)
            if latex:
                out.append(f"${latex}$")
            continue
        # 标题：# 前缀去掉，编号/白名单识别交给 chunker（层级由编号推断）
        if line.startswith("#"):
            t = line.lstrip("#").strip()
            if t:
                out.append(t)
            i += 1
            continue
        # 表格行 / 行内公式 / 正文：原样保留
        out.append(line)
        i += 1
    return "\n".join(out)


def mineru_markdown_to_pages(md_path: str | Path) -> list[PageText]:
    """读 MinerU Markdown，返回按页切分的 PageText 列表。

    页号取自 `<!-- page N of M -->` 标记（真实页码，比 pdfplumber 更准）。
    """
    md_path = Path(md_path)
    text = md_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    pages: list[PageText] = []
    cur_no: int | None = None
    buf: list[str] = []
    for line in lines:
        m = _PAGE_MARK.match(line.strip())
        if m:
            if cur_no is not None and any(s.strip() for s in buf):
                pages.append(PageText(page_no=cur_no, text=_convert_page("\n".join(buf), cur_no)))
            cur_no = int(m.group(1))
            buf = []
        else:
            buf.append(line)
    if cur_no is not None and any(s.strip() for s in buf):
        pages.append(PageText(page_no=cur_no, text=_convert_page("\n".join(buf), cur_no)))
    return pages
