# -*- coding: utf-8 -*-
"""消融续跑：vector_only 组 ask+judge（拿准确率）→ 补跑两组 ragas 指标。

背景：no_expand 组 21:41 卡死在 ragas（OpenAI 客户端无 timeout），ask+judge
已完成 100 样本；vector_only 组尚未启动。先保证四组准确率齐全，ragas 后补；
ragas 阶段已加 timeout=120 防止再次无限挂起。
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

STEPS = [
    ("vector_only ask+judge", [PY, str(EVAL), "--tag", "vector_only", "--skip-ragas"],
     {"HYBRID_USE_BM25": "0"}),
    ("no_expand ragas 补跑", [PY, str(EVAL), "--tag", "no_expand", "--skip-ask"], {}),
    ("vector_only ragas 补跑", [PY, str(EVAL), "--tag", "vector_only", "--skip-ask"],
     {"HYBRID_USE_BM25": "0"}),
]


def log(msg: str) -> None:
    line = time.strftime("%m-%d %H:%M:%S") + " " + msg
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> None:
    for name, cmd, env_extra in STEPS:
        env = os.environ.copy()
        env.update(env_extra)
        log(f"=== {name} 开始 ===")
        t0 = time.time()
        r = subprocess.run(cmd, env=env, cwd=ROOT)
        dt = round((time.time() - t0) / 60, 1)
        log(f"=== {name} 结束 exit={r.returncode} 耗时={dt} 分钟 ===")
    log("续跑全部完成")


if __name__ == "__main__":
    main()
