import re, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

md = open(r"D:\mineru_papers_out\Raindrop_Clarity_2407.16957.md", encoding="utf-8").read()
print("=== MinerU md 中搜训练/测试划分数字 ===")
for num in ("4,713", "4713", "729", "8,655", "8655", "1,089", "1089", "train", "test split", "training"):
    hits = [l for l in md.splitlines() if num.lower() in l.lower()]
    if hits:
        print(f"[{num}] {len(hits)} 行")
        for h in hits[:2]:
            print("   ", h[:160])
    else:
        print(f"[{num}] 无")

# pdfplumber 原文也查一下（论文 PDF 原始文本）
print("\n=== 原 PDF（pdfplumber 快速查）===")
try:
    import pdfplumber
    with pdfplumber.open(r"D:\论文阅读问答助手\data\papers\Raindrop_Clarity_2407.16957.pdf") as pdf:
        full = "\n".join((p.extract_text() or "") for p in pdf.pages)
    for num in ("4,713", "4713", "729", "8,655", "8655", "1,089", "1089"):
        hits = [l for l in full.splitlines() if num in l]
        if hits:
            print(f"[{num}] PDF 原文 {len(hits)} 行")
            for h in hits[:2]:
                print("   ", h[:160])
        else:
            print(f"[{num}] PDF 原文无")
except Exception as e:
    print("pdfplumber 查询失败:", e)
