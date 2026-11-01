$log = "D:\论文阅读问答助手\eval\results\judge_5p.log"
New-Item -ItemType File -Force $log | Out-Null
$tags = @("full_5p","no_mmr_5p","no_expand_5p","vector_only_5p","mineru_ab_5p","mineru_v2_5p")
foreach ($t in $tags) {
  "=== 重判 $t ===" | Out-File $log -Append -Encoding utf8
  $t0 = Get-Date
  & "D:\anaconda\envs\paper_rag\python.exe" -X utf8 eval/run_eval.py --skip-ask --skip-ragas --tag $t 2>&1 | Out-File $log -Append -Encoding utf8
  $sec = [math]::Round(((Get-Date) - $t0).TotalSeconds)
  "完成 $t  用时 ${sec}s" | Out-File $log -Append -Encoding utf8
}
"DONE" | Out-File $log -Append -Encoding utf8
