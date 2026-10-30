import sys, io, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, r"D:\论文阅读问答助手")
from src.parser.mineru_parser import _html_table_to_md, _convert_page

t = open(r"D:\mineru_papers_out\Raindrop_Clarity_2407.16957.md", encoding="utf-8").read()
tables = re.findall(r"<table[^>]*>.*?</table>", t, re.DOTALL)
print("找到 table 数量:", len(tables))
for i, tb in enumerate(tables):
    print("--- table", i, "长度:", len(tb))
    md = _html_table_to_md(tb)
    print("转换结果 len:", len(md) if md else 0)
    if md:
        print(md[:300])
    else:
        print("!! 转换返回空")

# 走 _convert_page 全流程看是否产出 | 行
pages = []
for page in re.split(r"<!-- page (\d+) of \d+ -->", t)[::2]:
    pass
print("\n=== _convert_page 全流程测试 ===")
import src.parser.mineru_parser as mp
# 找分页标记
segs = re.split(r"<!--\s*page \d+ of \d+\s*-->", t)
print("分段数:", len(segs))
out_all = []
for seg in segs:
    out_all.extend(mp._convert_page(seg, 1, 1, None))
pipe_lines = [l for l in out_all if "|" in l]
print("全流程 | 行数:", len(pipe_lines))
for l in pipe_lines[:5]:
    print("  ", l[:120])
