$log = "D:\论文阅读问答助手\eval\results\build3.log"
New-Item -ItemType File -Force $log | Out-Null
$py = "D:\anaconda\envs\paper_rag\python.exe"
cd "D:\论文阅读问答助手"

# 1) pdfplumber baseline 库
"=== 建库1 pdfplumber ===" | Out-File $log -Append -Encoding utf8
$env:CHROMA_DIR = "D:\论文阅读问答助手\data\chroma_pdfplumber"
$env:BM25_PATH = "D:\论文阅读问答助手\data\bm25_pdfplumber.pkl"
$env:CHUNK_META_PATH = "D:\论文阅读问答助手\data\chunks_pdfplumber.pkl"
$env:MINERU_OUT_DIR = "D:\mineru_papers_out_v2"
& $py -X utf8 src/ingest.py --parser pdfplumber 2>&1 | Out-File $log -Append -Encoding utf8

# 2) mineru v1 库（截断版产物）
"=== 建库2 mineru_v1 ===" | Out-File $log -Append -Encoding utf8
$env:CHROMA_DIR = "D:\论文阅读问答助手\data\chroma_mineru_v1"
$env:BM25_PATH = "D:\论文阅读问答助手\data\bm25_mineru_v1.pkl"
$env:CHUNK_META_PATH = "D:\论文阅读问答助手\data\chunks_mineru_v1.pkl"
$env:MINERU_OUT_DIR = "D:\mineru_papers_out"
& $py -X utf8 src/ingest.py --parser mineru 2>&1 | Out-File $log -Append -Encoding utf8

# 3) mineru v2 库（--pages all 全量产物）
"=== 建库3 mineru_v2 ===" | Out-File $log -Append -Encoding utf8
$env:CHROMA_DIR = "D:\论文阅读问答助手\data\chroma_mineru_v2"
$env:BM25_PATH = "D:\论文阅读问答助手\data\bm25_mineru_v2.pkl"
$env:CHUNK_META_PATH = "D:\论文阅读问答助手\data\chunks_mineru_v2.pkl"
$env:MINERU_OUT_DIR = "D:\mineru_papers_out_v2"
& $py -X utf8 src/ingest.py --parser mineru 2>&1 | Out-File $log -Append -Encoding utf8

"DONE" | Out-File $log -Append -Encoding utf8
