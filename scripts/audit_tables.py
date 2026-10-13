# -*- coding: utf-8 -*-
"""表格信息全面审计：4 篇论文所有表格区域的抽取结果检验。

两部分：
A. 边框表：逐页 find_tables()，按与 pdf_parser 相同的判定分类
   （真表格转 Markdown / 伪表格跳过及原因）；
B. 无边框表：列对齐启发式候选（src/parser/table_align.py）——输出候选
   区域，最终是否入库由 LLM 判定（src/parser/table_llm.py，TABLE_LLM 开关）；
再核对 chunks.pkl 中实际入库的表格块。
"""
import sys, pickle
from pathlib import Path

sys.path.insert(0, r"D:\论文阅读问答助手")

import pdfplumber
from src.parser.pdf_parser import _table_to_markdown, _words_to_lines, _word_in_table
from src.parser.chunker import chunk_pdf
from src.parser.table_align import detect_aligned_tables

PAPERS_DIR = Path(r"D:\论文阅读问答助手\data\papers")

print("=" * 70)
print("【第 1 部分】PDF 逐页表格区域检测与分类")
print("=" * 70)

grand = {"pdf": 0, "region": 0, "real": 0, "fake": 0, "real_total_rows": 0}
for pdf_path in sorted(PAPERS_DIR.glob("*.pdf")):
    grand["pdf"] += 1
    print(f"\n### {pdf_path.name}")
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            tables = page.find_tables() or []
            if not tables:
                continue
            for t in tables:
                grand["region"] += 1
                rows = t.extract()
                n_rows = len(rows) if rows else 0
                n_cols = len(rows[0]) if rows else 0
                md = _table_to_markdown(t)
                if md is None:
                    grand["fake"] += 1
                    reason = []
                    if n_rows < 3:
                        reason.append(f"行数<3({n_rows})")
                    if n_cols < 2:
                        reason.append(f"列数<2({n_cols})")
                    if not reason:
                        reason.append("实心行<2")
                    print(f"  [第{i}页] 伪表格跳过 bbox=({t.bbox[0]:.0f},{t.bbox[1]:.0f})-({t.bbox[2]:.0f},{t.bbox[3]:.0f}) "
                          f"行x列={n_rows}x{n_cols} 原因:{','.join(reason)}")
                else:
                    grand["real"] += 1
                    grand["real_total_rows"] += n_rows
                    md_lines = md.splitlines()
                    print(f"  [第{i}页] 真表格 bbox=({t.bbox[0]:.0f},{t.bbox[1]:.0f})-({t.bbox[2]:.0f},{t.bbox[3]:.0f}) "
                          f"行x列={n_rows}x{n_cols}")
                    for ln in md_lines[:4]:
                        print(f"      {ln[:100]}")
                    if len(md_lines) > 4:
                        print(f"      ...（共 {len(md_lines)} 行）")

print("\n" + "=" * 70)
print(f"汇总：{grand['pdf']} 篇 PDF，检测到 {grand['region']} 个表格区域，"
      f"真表格 {grand['real']} 个，伪表格跳过 {grand['fake']} 个，真表格总行数 {grand['real_total_rows']}")
print("=" * 70)

print("\n【第 1.5 部分】无边框表候选（列对齐启发式；最终入库由 LLM 判定）")
align_total = 0
for pdf_path in sorted(PAPERS_DIR.glob("*.pdf")):
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            words = page.extract_words(x_tolerance=1.5, y_tolerance=3) or []
            for t in detect_aligned_tables(words, page.width):
                align_total += 1
                print(f"  {pdf_path.name} [第{i}页] bbox=({t.bbox[0]:.0f},{t.bbox[1]:.0f})-({t.bbox[2]:.0f},{t.bbox[3]:.0f}) "
                      f"行数={len(t.lines)} 列数={t.n_cols_hint}")
                print(f"      首行: {t.lines[0][:80]}")
print(f"无边框表候选总数: {align_total}（LLM 判定后入库）\n")

print("\n【第 2 部分】chunks.pkl 中实际入库的表格块（含『（表）』标记或 Markdown 特征）")
with open(r"D:\论文阅读问答助手\data\chunks.pkl", "rb") as f:
    chunks = pickle.load(f)
table_chunks = [
    c for c in chunks
    if "（表）" in c.section or "\n| " in c.content or c.content.strip().startswith("| ")
]
print(f"表格块数量: {len(table_chunks)}")
for c in sorted(table_chunks, key=lambda c: c.chunk_id):
    print(f"\n  [{c.chunk_id}] 章节={c.section} 第{c.page}页")
    lines = c.content.splitlines()
    for ln in lines[:8]:
        print(f"      {ln[:100]}")
    if len(lines) > 8:
        print(f"      ...（共 {len(lines)} 行）")
