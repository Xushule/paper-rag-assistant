# -*- coding: utf-8 -*-
"""消融评测驱动：四组配置串行全量跑 run_eval.py（每组 100 样本 + ragas）。

组别（老师要求）：
    full         完整版（RRF 混合 + MMR + 查询扩展，全开）
    no_mmr       关 MMR（RRF -> 直接精排）
    no_expand    关查询扩展（单路检索，无 LLM 子查询）
    vector_only  仅向量（无 BM25，RRF 退化为纯向量排名）

每组输出隔离：eval/results/{samples,eval_summary,eval_results,ragas_scores,run_log}_{tag}.*
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
EVAL = ROOT / "eval" / "run_eval.py"
LOG = ROOT / "data" / "_ablation_run.log"

GROUPS = [
    ("full",         {}),
    ("no_mmr",       {"MMR_ENABLED": "0"}),
    ("no_expand",    {"QUERY_EXPAND": "0"}),
    ("vector_only",  {"HYBRID_USE_BM25": "0"}),
]


def log(msg: str) -> None:
    line = time.strftime("%H:%M:%S") + " " + msg
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> None:
    results = {}
    for tag, env_extra in GROUPS:
        env = os.environ.copy()
        env.update(env_extra)
        log(f"=== {tag} 开始（env: {env_extra or '默认全开'}）===")
        t0 = time.time()
        r = subprocess.run([PY, str(EVAL), "--tag", tag], env=env, cwd=ROOT)
        dt = round((time.time() - t0) / 60, 1)
        results[tag] = r.returncode
        log(f"=== {tag} 结束 exit={r.returncode} 耗时={dt} 分钟 ===")
    log("全部完成: " + str(results))


if __name__ == "__main__":
    main()
