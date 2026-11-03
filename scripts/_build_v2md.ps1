# 建 mineru_v2md 库（markdown-aware 分块，env 隔离）
# 兼容 Windows PowerShell 5.1（后台宿主）：New-Item 只用 -Path
$lg = 'D:\论文阅读问答助手\eval\results\build_v2md.log'
New-Item -ItemType File -Force -Path $lg | Out-Null
$py = 'D:\anaconda\envs\paper_rag\python.exe'
Set-Location -LiteralPath 'D:\论文阅读问答助手'

$env:CHROMA_DIR = 'D:\论文阅读问答助手\data\chroma_mineru_v2md'
$env:BM25_PATH = 'D:\论文阅读问答助手\data\bm25_mineru_v2md.pkl'
$env:CHUNK_META_PATH = 'D:\论文阅读问答助手\data\chunks_mineru_v2md.pkl'
$env:MINERU_OUT_DIR = 'D:\mineru_papers_out_v2'
& $py -X utf8 src/ingest.py --parser mineru 2>&1 | Out-File -FilePath $lg -Append -Encoding utf8

'DONE' | Out-File -FilePath $lg -Append -Encoding utf8
