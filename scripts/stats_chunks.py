# -*- coding: utf-8 -*-
"""分块质量诊断：统计一篇论文的块数、块长度分布、章节分布。

用法:
    python scripts/stats_chunks.py [pdf路径]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.parser.chunker import chunk_pdf, split_into_sections
from src.parser.pdf_parser import extract_pdf_text


def main(pdf_path: str) -> None:
    pdf_path = Path(pdf_path)
    pages = extract_pdf_text(pdf_path)
    print(f"页数: {len(pages)}")

    sections = split_into_sections(pages)
    print(f"章节数: {len(sections)}")
    for title, text, page in sections[:40]:
        print(f"  [p{page:>3}] {title[:40]:<44} {len(text):>8} 字符")

    chunks = chunk_pdf(pdf_path)
    lens = sorted(len(c.content) for c in chunks)
    n = len(lens)
    if n == 0:
        print("没有分块")
        return
    avg = sum(lens) // n
    print(f"\n总块数: {n}, 平均 {avg} 字符, 最短 {lens[0]}, 最长 {lens[-1]}, 中位 {lens[n//2]}")
    tiny = sum(1 for l in lens if l < 300)
    print(f"小于 300 字符的块: {tiny} ({tiny / n:.1%})")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else r"D:\论文阅读问答助手\data\papers\Raindrop_Clarity_2407.16957.pdf"
    main(arg)
