$log = "D:\论文阅读问答助手\eval\results\eval_5p.log"
New-Item -ItemType File -Force $log | Out-Null
$py = "D:\anaconda\envs\paper_rag\python.exe"
cd "D:\论文阅读问答助手"

# 库绑定（每组用独立库，互不污染）
$pdf = "D:\论文阅读问答助手\data\chroma_pdfplumber"
$pdf_bm25 = "D:\论文阅读问答助手\data\bm25_pdfplumber.pkl"
$pdf_chunks = "D:\论文阅读问答助手\data\chunks_pdfplumber.pkl"
$v1 = "D:\论文阅读问答助手\data\chroma_mineru_v1"
$v1_bm25 = "D:\论文阅读问答助手\data\bm25_mineru_v1.pkl"
$v1_chunks = "D:\论文阅读问答助手\data\chunks_mineru_v1.pkl"
$v2 = "D:\论文阅读问答助手\data\chroma_mineru_v2"
$v2_bm25 = "D:\论文阅读问答助手\data\bm25_mineru_v2.pkl"
$v2_chunks = "D:\论文阅读问答助手\data\chunks_mineru_v2.pkl"

function Set-Lib([string]$chroma, [string]$bm25, [string]$chunks) {
  $env:CHROMA_DIR = $chroma
  $env:BM25_PATH = $bm25
  $env:CHUNK_META_PATH = $chunks
}

function Run-Group([string]$tag, [string]$chroma, [string]$bm25, [string]$chunks, [string]$extra) {
  "=== 评测 $tag ===" | Out-File $log -Append -Encoding utf8
  $t0 = Get-Date
  Set-Lib $chroma $bm25 $chunks
  Remove-Item env:MMR_ENABLED,env:QUERY_EXPAND,env:HYBRID_USE_BM25 -ErrorAction SilentlyContinue
  if ($extra -ne "") { Invoke-Expression $extra }
  & $py -X utf8 eval/run_eval.py --skip-ragas --tag $tag 2>&1 | Out-File $log -Append -Encoding utf8
  $sec = [math]::Round(((Get-Date) - $t0).TotalSeconds)
  "完成 $tag  用时 ${sec}s" | Out-File $log -Append -Encoding utf8
}

Run-Group "full_5p"      $pdf $pdf_bm25 $pdf_chunks ""
Run-Group "no_mmr_5p"    $pdf $pdf_bm25 $pdf_chunks '$env:MMR_ENABLED="0"'
Run-Group "no_expand_5p" $pdf $pdf_bm25 $pdf_chunks '$env:QUERY_EXPAND="0"'
Run-Group "vector_only_5p" $pdf $pdf_bm25 $pdf_chunks '$env:HYBRID_USE_BM25="0"'
Run-Group "mineru_ab_5p" $v1 $v1_bm25 $v1_chunks ""
Run-Group "mineru_v2_5p" $v2 $v2_bm25 $v2_chunks ""
"DONE" | Out-File $log -Append -Encoding utf8
