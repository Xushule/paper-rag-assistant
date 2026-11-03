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

【markdown-aware 专属分块（M2.19，2026-11-03）】
上面"转行文本流复用 split_into_sections"是降级方案——把 MinerU 最值钱的
结构化信息（显式标题层级 / HTML 复杂表列名 / 段落边界）扔掉了：
  - `#` 数量本来就给出标题级别，却剥掉后让 chunker"猜"层级（对无编号标题
    如 Abstract/References，markdown 版正确定级、猜版可能猜错）；
  - `<table>` 复杂表（rowspan/colspan）列名行+数值行完整，压平成文本后
    "哪列哪个值"关系丢失，表格数字题（哪个方法 PSNR 最高）命中率下降。
因此新增 mineru_markdown_to_sections：直接按 markdown 标记切分，
返回与 split_into_sections 同构的 [(章节路径, 文本, 起始页)]，
chunk_pdf 的后续切块/入库逻辑原样复用；pdfplumber 路径保持不变，
两者仍可公平 A/B。
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
# 表格说明行锚点（"Table 1. ..." / "表 1 ..."），并入表格块首部供检索命中
_TABLE_CAP = re.compile(r"(Table|表)\s*\d")


def _strip_inline_tags(s: str) -> str:
    """剥掉行内 HTML 标签（<sup>/<small>/<sub> 等），保留文本。"""
    return _STRIP_TAG.sub("", s).strip()


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


# ---------------------------------------------------------------------------
# markdown-aware 专属分块（M2.19）：直接利用 MinerU 的结构化标记切分
# ---------------------------------------------------------------------------

# Markdown 标题："### 3.2 Method" -> level=3（显式层级，无需猜）
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
# 纯数字行（MinerU 产物里的页码噪声 "3"）跳过
_PAGE_NUM_ONLY = re.compile(r"^\d{1,3}$")


def _html_table_rows(html: str) -> list[list[str]]:
    """提取 HTML 表格的单元格矩阵。

    rowspan/colspan 合并值保留在其首次出现的行；表头行（第一行）的
    列名与数据行同矩阵输出，检索"列名-数值"关系不丢失。
    """
    rows: list[list[str]] = []
    for row in _HTML_ROW.findall(html):
        cells = [_STRIP_TAG.sub("", c).strip() for c in _HTML_CELL.findall(row)]
        if cells:
            rows.append(cells)
    return rows


def _md_table_text(rows: list[list[str]]) -> str:
    """单元格矩阵 -> Markdown 表格文本（列名行 + 数据行）。"""
    return "\n".join("| " + " | ".join(cells) + " |" for cells in rows)


def mineru_markdown_to_sections(md_path: str | Path) -> list[tuple[str, str, int]]:
    """MinerU Markdown 专属分块：返回 [(章节路径, 章节文本, 起始页)]。

    与 split_into_sections（为纯文本行设计）的差异——直接用 markdown 标记：
      1. 标题层级 = `#` 数量（1~6），无需按编号点号数/白名单猜；无编号标题
         （Abstract / References）由 `#` 正确定级，不会误挂父级；
      2. HTML 复杂表（rowspan/colspan）保留列名行 + 数值行独立成 section
         （"…（表）"），并把表格前最近的 "Table N" 说明行并入块首作检索锚点
         （对齐 pdfplumber 侧"表格块带 Table 锚点"的行为）；
      3. 段落边界保留（\n 连接），超长 section 由 _split_section 优先按段落
         断开，而不是纯字数硬切；
      4. 分页标记 -> 起始页元数据；图片块 -> "[图片块 第N页]" 锚点；
         $$..$$ 公式块压单行；行内 <sup>/<small> 等标签剥除。
    References 章节天然安全：条目行是普通段落（无 #），不会像文本流方案
    那样被"数字开头行"误判成标题而分裂。
    """
    md_path = Path(md_path)
    text = md_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    sections: list[tuple[str, str, int]] = []
    heading_stack: list[tuple[int, str]] = []  # [(level, title), ...]
    body: list[str] = []                        # 当前 section 的文本行
    cur_page = 1
    sec_page: int | None = None                 # 当前 section 起始页

    def section_path() -> str:
        return " > ".join(t for _, t in heading_stack) or "前言/未分区"

    def flush_section() -> None:
        nonlocal body, sec_page
        joined = "\n".join(x for x in body if x and x.strip())
        if joined.strip():
            sections.append((section_path(), joined, sec_page if sec_page is not None else cur_page))
        body = []
        sec_page = None

    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()
        # 分页标记：更新当前页码
        m = _PAGE_MARK.match(stripped)
        if m:
            cur_page = int(m.group(1))
            i += 1
            continue
        # 空行 = 段落边界（正文以换行保留段落结构，供超长 section 段落级断开）
        if not stripped:
            i += 1
            continue
        # 纯数字页码行噪声
        if _PAGE_NUM_ONLY.match(stripped):
            i += 1
            continue
        # HTML 表格（可能跨行）-> 独立 section，带 "Table N" 锚点
        if "<table" in raw:
            if "</table>" in raw:  # 单行完整表格（MinerU 复杂表常见形态）
                html = raw
                i += 1
            else:
                html_buf = [raw]
                i += 1
                while i < len(lines) and "</table>" not in lines[i]:
                    html_buf.append(lines[i])
                    i += 1
                if i < len(lines):
                    html_buf.append(lines[i])
                    i += 1
                html = "\n".join(html_buf)
            rows = _html_table_rows(html)
            if len(rows) >= 2:  # 表头 + 至少一行数据
                cap_rows: list[str] = []
                j = len(body) - 1
                while j >= 0 and len(cap_rows) < 3:
                    if _TABLE_CAP.search(body[j]):
                        cap_rows = body[j:]
                        break
                    j -= 1
                if cap_rows:
                    body = body[:j]
                tbl_text = "\n".join(cap_rows + [_md_table_text(rows)])
                flush_section()
                sections.append((section_path() + "（表）", tbl_text,
                                sec_page if sec_page is not None else cur_page))
                sec_page = None
            else:  # 表格碎片归正文
                body.append(_md_table_text(rows))
            continue
        # Markdown 表格行（|...|，含分隔行）-> 独立表格 section（带 Table 锚点）
        if stripped.startswith("|") and stripped.count("|") >= 2:
            tbl_rows: list[str] = []
            while i < len(lines) and lines[i].strip().startswith("|") \
                    and lines[i].strip().count("|") >= 2:
                tbl_rows.append(lines[i].rstrip())
                i += 1
            if len(tbl_rows) >= 2:  # 表头 + 分隔/数据行
                cap_rows = []
                j = len(body) - 1
                while j >= 0 and len(cap_rows) < 3:
                    if _TABLE_CAP.search(body[j]):
                        cap_rows = body[j:]
                        break
                    j -= 1
                if cap_rows:
                    body = body[:j]
                tbl_text = "\n".join(cap_rows + tbl_rows)
                flush_section()
                sections.append((section_path() + "（表）", tbl_text,
                                sec_page if sec_page is not None else cur_page))
                sec_page = None
            else:  # 零星表格碎片归正文
                body.extend(tbl_rows)
            continue
        # 标题：# 数量 = 层级
        hm = _MD_HEADING.match(stripped)
        if hm and len(stripped) < 80:
            title = _strip_inline_tags(hm.group(2))
            if title:
                flush_section()
                level = len(hm.group(1))
                # markdown 层级显式：弹出同级或更深，保留父级
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, title))
                sec_page = cur_page
            i += 1
            continue
        # 图片块 -> 语义锚点
        if _IMG_BLOCK.match(stripped):
            body.append(f"[图片块 第{cur_page}页]")
            if sec_page is None:
                sec_page = cur_page
            i += 1
            continue
        # 公式块 $$...$$（可跨行）-> 单行 LaTeX
        if stripped.startswith("$$"):
            latex, i = _formula_block_to_latex(lines, i)
            if latex:
                body.append(f"${latex}$")
            continue
        # 正文行：剥行内 html 标签后保留
        body.append(_strip_inline_tags(raw))
        if sec_page is None:
            sec_page = cur_page
        i += 1
    flush_section()
    return sections
