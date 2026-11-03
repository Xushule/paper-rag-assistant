# -*- coding: utf-8 -*-
"""验证 markdown-aware 分块（M2.19）：5 篇 section 统计 + RC 表格块抽查"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.parser.mineru_parser import mineru_markdown_to_sections

OUT = Path(r"D:\mineru_papers_out_v2")
out_lines = []
for md in sorted(OUT.glob("*.md")):
    if md.name.endswith(".log"):
        continue
    sections = mineru_markdown_to_sections(md)
    tbl_cnt = sum(1 for title, _, _ in sections if "（表）" in title)
    # 总字符与块数预估（1800 字符/块）
    total_chars = sum(len(t) for _, t, _ in sections)
    est_chunks = 0
    for _, t, _ in sections:
        est_chunks += max(1, (len(t) + 1799) // 1800)
    out_lines.append(f"{md.name}: sections={len(sections)} 表格块={tbl_cnt} chars={total_chars} 预估块数≈{est_chunks}")
    # 标题栈抽查（前 8 个 section 路径）
    for title, t, pg in sections[:8]:
        first = t.splitlines()[0][:60] if t else ""
        out_lines.append(f"    [{pg}] {title}  |  {first}")
    out_lines.append("")

# RC 篇表格块内容抽查
rc = [md for md in OUT.glob("Raindrop_Clarity*.md")][0]
sections = mineru_markdown_to_sections(rc)
out_lines.append("=== RC 表格块抽查（含 'Table 2' 锚点）===")
for title, t, pg in sections:
    if "（表）" in title and "Table 2" in t[:400]:
        out_lines.append(f"[{pg}] {title}")
        out_lines.append(t[:700])
        break

text = "\n".join(out_lines)
Path(r"D:\论文阅读问答助手\eval\results\_v2md_check.txt").write_text(text, encoding="utf-8")
print(text)
