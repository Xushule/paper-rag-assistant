# -*- coding: utf-8 -*-
"""验证 table_cells 坐标切分：NTIRE p4 排名表，Miracle/EntroVision 是否两行。"""
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pdfplumber

from src.parser.table_align import detect_aligned_tables
from src.parser.table_cells import cells_to_markdown

PDF = ROOT / "data" / "papers" / "NTIRE2025_DayNight_Raindrop_2504.12711.pdf"

with pdfplumber.open(str(PDF)) as pdf:
    page = pdf.pages[3]  # p4（0-indexed）
    words = page.extract_words(x_tolerance=1.5, y_tolerance=3) or []
    print(f"p4 单词数: {len(words)}")
    cands = detect_aligned_tables(words, page.width)
    print(f"候选表区域: {len(cands)}")

    for i, c in enumerate(cands):
        region = [w for w in words
                  if w["x0"] >= c.bbox[0] - 2 and w["x1"] <= c.bbox[2] + 2
                  and w["top"] >= c.bbox[1] - 2 and w["bottom"] <= c.bbox[3] + 2]
        joined = " ".join(w["text"] for w in region)
        if "Miracle" in joined or "EntroVision" in joined:
            print(f"\n候选 {i}: bbox={c.bbox}  n_cols_hint={c.n_cols_hint}")
            print("--- 切分结果（table_cells）---")
            md = cells_to_markdown(region)
            print(md if md else "切分失败 -> 返回 None")
            # 验证两行是否分开
            if md:
                rows = [ln for ln in md.splitlines() if ln.startswith("|")][2:]
                miracle_line = [ln for ln in rows if "Miracle" in ln]
                entro_line = [ln for ln in rows if "EntroVision" in ln]
                print("--- 验证 ---")
                print("Miracle 独立行:", bool(miracle_line))
                print("EntroVision 独立行:", bool(entro_line))
                if miracle_line and entro_line:
                    print("✅ 两行已分开（行合并问题解决）")
                else:
                    print("❌ 仍在一行/未分开")
