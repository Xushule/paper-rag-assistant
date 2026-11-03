# 评测 mineru_v2md_5p（markdown-aware 分块，full 链路全开）
$log = "D:\论文阅读问答助手\eval\results\eval_v2md.log"
New-Item -ItemType File -Force $log | Out-Null
$py = "D:\anaconda\envs\paper_rag\python.exe"
cd "D:\论文阅读问答助手"

$env:CHROMA_DIR = "D:\论文阅读问答助手\data\chroma_mineru_v2md"
$env:BM25_PATH = "D:\论文阅读问答助手\data\bm25_mineru_v2md.pkl"
$env:CHUNK_META_PATH = "D:\论文阅读问答助手\data\chunks_mineru_v2md.pkl"
Remove-Item env:MMR_ENABLED,env:QUERY_EXPAND,env:HYBRID_USE_BM25 -ErrorAction SilentlyContinue

$t0 = Get-Date
& $py -X utf8 eval/run_eval.py --skip-ragas --tag mineru_v2md_5p 2>&1 | Out-File $log -Append -Encoding utf8
$sec = [math]::Round(((Get-Date) - $t0).TotalSeconds)
"完成 mineru_v2md_5p  用时 ${sec}s" | Out-File $log -Append -Encoding utf8
"DONE" | Out-File $log -Append -Encoding utf8
