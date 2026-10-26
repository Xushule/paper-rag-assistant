# -*- coding: utf-8 -*-
"""结构感知分块：论文文本 -> 按章节切分 -> 每章切成带重叠的块。

为什么不用 LangChain 默认的固定长度分块：
1) 按章节切能保住语义边界（Method 和 Conclusion 不会被混进同一个块）；
2) 块上保存"章节名、页码、来源"元数据，这是后面"引用溯源"的基础；
3) 面试时你能讲清楚"为什么 500~800 token、为什么重叠"。

标题识别策略（经过两轮真实论文验证）：
- 只认两类标题，避免把图注/作者名/机构行/页眉误判成章节：
  ① 带编号的标题："3.2 Method"、"4 Experiments"、"一、引言"；
  ② 以常见章节关键词开头的无编号标题："Introduction"、"Related Work"（要求词边界）；
- 命中页眉/机构/图注特征词（et al.、University、@ 等）的一律排除。
"""
import re
from dataclasses import dataclass
from pathlib import Path

from .pdf_parser import PageText, extract_pdf_text

# 带编号的英文标题："3.2 Method" / "4 Experiments"
NUM_HEADING = re.compile(r"^\s*\d+(\.\d+)*\s*[\.\)]?\s+[A-Z][A-Za-z\u4e00-\u9fff].{1,60}$")
# 带编号的中文标题："一、引言" / "1.1 相关工作"
CN_NUM_HEADING = re.compile(r"^\s*[一二三四五六七八九十]+[、.]\s*\S{1,40}$")

# 无编号标题白名单：以这些词开头才当作标题（小写匹配，要求词边界）
KNOWN_HEADING_WORDS = [
    "abstract", "introduction", "related work", "background", "preliminary",
    "method", "methods", "approach", "proposed method", "methodology",
    "framework", "experiments", "experimental results", "evaluation", "results",
    "discussion", "conclusion", "conclusions", "references",
    "acknowledg", "appendix", "supplementary", "limitations", "social impact",
    "引言", "相关工作", "背景", "方法", "实验", "结果", "结论",
    "参考文献", "致谢", "摘要", "附录",
]

# 命中这些特征的"标题"是误判（页眉/机构行/图注），直接排除
HEADING_EXCLUDE_WORDS = [
    "et al.", "university", "institute", "school of", "college",
    "laboratory", "@",
]

MAX_HEADING_LEN = 60  # 超过 60 字符不当作标题


@dataclass
class Chunk:
    """一个检索单元（块）。"""
    chunk_id: str
    content: str
    source: str
    section: str
    page: int
    chunk_index: int


def _looks_like_heading(line: str) -> bool:
    """判断一行是否像章节标题。"""
    line = line.strip()
    if not line or len(line) > MAX_HEADING_LEN:
        return False
    low = line.lower()
    if any(w in low for w in HEADING_EXCLUDE_WORDS):
        return False
    if NUM_HEADING.match(line) or CN_NUM_HEADING.match(line):
        return True
    # 无编号标题：整行精确等于白名单词（允许尾部标点），避免
    # "Background-focused" 误命中 "background"、"approach leads to effective
    # deraining while simultaneously" 这类以白名单词开头的正文句子被误判为标题
    # （句子 40~60 字符、以句号结尾，整行匹配自然排除）
    return any(
        re.match(rf"^{re.escape(kw)}[.:：]?$", low)
        for kw in KNOWN_HEADING_WORDS
    )


# 表格说明行："...sets. Table 1. Bg-focus = ..."（可能嵌在行中间，用 search 匹配，
# 并入表格块作为检索锚点）
TABLE_CAPTION = re.compile(r"(Table|表)\s*\d")


def _heading_level(title: str) -> int:
    """从编号推断标题层级：'3.2 Method' → 2 级；'3.2.1 X' → 3 级；无编号 → 1 级。

    用于维护标题栈：深层标题切出的块要带完整父级路径
    （'2 Related Work > 2.1 Data > 2.1.3 X > 2.1.3.2 Y'）。
    """
    m = re.match(r"^\s*(\d+(\.\d+)*)", title)
    if m:
        return m.group(1).count(".") + 1
    return 1


def _is_numbered_ancestor(prev_title: str, cur_title: str) -> bool:
    """prev_title 是否为 cur_title 的编号祖先（'4' ⊃ '4.2'；'3.2' ⊃ '3.2.1'）。

    防止错挂：识别不到中间层标题时（如 "3 Dataset" 没被识别），
    "3.2 Characteristics" 不能挂到无关的 "2 Related Work" 下面——
    宁可不挂（块只有自身标题），也不挂错（引用时显示错误父级）。
    """
    m1 = re.match(r"^(\d+(\.\d+)*)", prev_title)
    m2 = re.match(r"^(\d+(\.\d+)*)", cur_title)
    if not m1 or not m2:
        return False
    return cur_title.startswith(m1.group(1) + ".")


def split_into_sections(pages: list[PageText]) -> list[tuple[str, str, int]]:
    """把多页文本按标题切分成 [(章节名, 章节文本, 起始页)]。

    表格单独成段：连续"|"开头的 Markdown 表格行被打包成独立片段，
    不与正文混在一起——否则表格块被正文稀释，检索"表 1 的数值"
    时表格块排名靠后、答不准（真实踩坑）。
    表格前面的说明行（"Table 1. ..."）并入表格块，作为检索锚点。

    标题层级（标题栈）：章节名存"一级 > 二级 > ... > 当前级"完整路径。
    四级标题切出的块能看到它的一/二/三级父标题（检索片段自包含、
    引用溯源精确到章）。
    """
    def is_table_row(line: str) -> bool:
        s = line.strip()
        return s.startswith("|") and s.count("|") >= 2

    sections = []
    heading_stack: list[tuple[int, str]] = []  # [(level, title), ...]
    current_title = "前言/未分区"
    current_lines = []  # [(page_no, line), ...]

    def flush_body():
        nonlocal current_lines
        if current_lines:
            sections.append((
                current_title,
                " ".join(text for _, text in current_lines),
                current_lines[0][0],
            ))
            current_lines = []

    for page in pages:
        lines = page.text.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i]
            # 连续表格行打包成独立片段
            if is_table_row(line):
                tbl = []
                # 表格说明行（含 "Table 1" 锚点）并入表格块首部：
                # 说明可能占 1~2 行（"Summary ... Table 1." + "Bg-focus = ..."），
                # 从 current_lines 尾部往回最多找 3 行，遇到含 "Table N" 即收
                caption_rows = []
                j = len(current_lines) - 1
                while j >= 0 and len(caption_rows) < 3:
                    caption_rows.insert(0, current_lines[j])
                    if TABLE_CAPTION.search(current_lines[j][1]):
                        break
                    j -= 1
                if caption_rows and TABLE_CAPTION.search(caption_rows[0][1]):
                    for _ in caption_rows:
                        current_lines.pop()
                    tbl.extend(caption_rows)
                while i < len(lines) and is_table_row(lines[i]):
                    tbl.append((page.page_no, lines[i]))
                    i += 1
                if len(tbl) >= 3:  # 表头 + 分隔行 + 至少一行数据
                    flush_body()
                    sections.append((
                        current_title + "（表）",
                        "\n".join(text for _, text in tbl),
                        tbl[0][0],
                    ))
                else:  # 零星表格碎片归回正文流
                    current_lines.extend(tbl)
                continue
            # References 章节内数字开头的行（参考文献条目 "14 In ECCV, ..."、
            # "31. Wang, ..."）不做标题——否则 References 被条目行分裂成多个
            # "伪章节"，后续块 section 丢失 reference 标识、校验全文逻辑失效
            in_refs = "reference" in current_title.lower()
            if _looks_like_heading(line) and not (in_refs and re.match(r"^\s*\d", line)):
                flush_body()
                title = line.strip()
                level = _heading_level(title)
                # 弹掉同级或更深的兄弟标题，保留父级："3.2" 出现时清掉 "3.1.x"；
                # 新一级标题（如 Introduction）出现时清空整栈
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                # 编号子标题要求栈顶是它的编号祖先（"4.2" 的父级必须是 "4"），
                # 识别不到中间层标题时宁可让块只有自身标题，也不挂错父级
                while heading_stack and _is_numbered_ancestor(
                        heading_stack[-1][1], title) is False \
                        and heading_stack[-1][0] < level:
                    heading_stack.pop()
                heading_stack.append((level, title))
                current_title = " > ".join(t for _, t in heading_stack)
                current_lines = [(page.page_no, line)]
            else:
                current_lines.append((page.page_no, line))
            i += 1
    flush_body()
    return sections


def _split_section(section_text: str, target: int, overlap: int) -> list[str]:
    """把一段长文本切成若干块（目标 ~500-600 token，带重叠，句子边界断开）。

    注意：切到最后一段（end 到达文末）时必须结束，否则重叠会让 start 每次
    只前进 1 个字符，产生一串 1~180 字符的碎块（真实踩过的 bug）。
    """
    if len(section_text) <= target:
        return [section_text] if section_text.strip() else []

    chunks = []
    start = 0
    while start < len(section_text):
        end = min(start + target, len(section_text))
        if end >= len(section_text):
            chunk = section_text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break
        for sep in ["。", "？", "！", ". ", "? ", "! ", "\n"]:
            idx = section_text.rfind(sep, start + target // 2, end)
            if idx != -1:
                end = idx + len(sep)
                break
        chunk = section_text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = max(end - overlap, start + 1)
    return chunks


def chunk_pdf(pdf_path: str | Path) -> list[Chunk]:
    """完整分块流程：解析 PDF -> 分章节 -> 切块 -> 返回 Chunk 列表。"""
    pdf_path = Path(pdf_path)
    pages = extract_pdf_text(pdf_path)
    sections = split_into_sections(pages)

    chunks = []
    for section_title, section_text, start_page in sections:
        pieces = _split_section(section_text, 1800, 180)  # ≈500~600 token
        for idx, piece in enumerate(pieces):
            chunks.append(Chunk(
                chunk_id=f"{pdf_path.stem}-{len(chunks)}",
                content=piece,
                source=pdf_path.name,
                section=section_title,
                page=start_page,
                chunk_index=idx,
            ))
    return chunks
