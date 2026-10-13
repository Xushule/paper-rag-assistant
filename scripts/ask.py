# -*- coding: utf-8 -*-
"""CLI 问答入口：命令行直接提问，端到端验证检索 + 生成 + 引用。

用法：
    python scripts/ask.py "夜间雨滴去除的难点是什么？"
    python scripts/ask.py --paper "Raindrop Clarity" "数据集包含哪些条件？"
    python scripts/ask.py                    # 进入交互模式，输入 q 退出

交互模式额外命令：
    /papers            列出库内论文
    /paper <关键词>     切换当前论文（只在这篇论文里检索，防止串台）
    /all               切回全库检索
    /reset             清空对话历史（多轮记忆）
    q                  退出
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.qa.answer import QASystem
from src.retrieval.bm25_index import load_index
from src.retrieval.embed_store import VectorStore


def _match_paper(qa: QASystem, keyword: str) -> str | None:
    """按关键词模糊匹配论文文件名；唯一匹配返回文件名，否则 None。

    文件名用下划线（如 Raindrop_Clarity_2407.16957.pdf），用户可能输入
    空格/大小写差异，匹配前把 _ - 空格 统一归一化。
    """
    def _norm(x: str) -> str:
        return x.lower().replace("_", " ").replace("-", " ")

    kw = _norm(keyword.strip())
    matched = [s for s in qa.papers() if kw in _norm(s)]
    if len(matched) == 1:
        return matched[0]
    if len(matched) > 1:
        print(f"[ask] 匹配到多篇，请更精确：{matched}")
        return None
    print(f"[ask] 没找到包含 '{keyword}' 的论文，可用 /papers 查看列表")
    return None


def main(question: str | None = None, paper: str | None = None) -> None:
    # 加载向量库 + BM25 + 分块元数据（每次启动一次性加载）
    print("[ask] 加载向量库与 BM25 索引 ...")
    store = VectorStore()
    bm25, chunks = load_index()
    qa = QASystem(store, bm25, chunks)
    print(f"[ask] 就绪，库内共 {store.count()} 个块\n")

    # 一次性问答：--paper 指定作用域
    if question:
        scope = None
        if paper:
            scope = _match_paper(qa, paper)
            if scope is None:
                return
            print(f"[ask] 作用域：{scope}")
        _ask_once(qa, question, scope)
        return

    # 交互模式：默认锁定第一篇论文（符合"针对某一篇论文"的阅读形态，
    # 防止多篇论文串台）；/paper 切换，/all 切全库
    all_papers = qa.papers()
    if not all_papers:
        print("[ask] 库内没有论文，请先运行 python src/ingest.py 入库")
        return
    scope = all_papers[0]
    print("进入交互模式（输入 q 退出；/papers 看论文；/paper 关键词 切换论文）")
    print(f"[ask] 当前论文：{scope}（默认锁定，/all 可切全库）")
    while True:
        try:
            q = input("\n问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[ask] 已退出")
            break
        if q.lower() == "q":
            break
        if not q:
            print("（输入为空，请直接输入问题；输入 q 退出）")
            continue
        if q.startswith("/"):
            cmd, _, arg = q.partition(" ")
            if cmd == "/papers":
                for i, s in enumerate(qa.papers(), start=1):
                    mark = "  <-- 当前" if s == scope else ""
                    print(f"  {i}. {s}{mark}")
            elif cmd == "/paper":
                if not arg.strip():
                    print("用法：/paper 关键词，例如 /paper Raindrop")
                else:
                    new_scope = _match_paper(qa, arg.strip())
                    if new_scope:
                        scope = new_scope
                        print(f"[ask] 已切换：{scope}")
            elif cmd == "/all":
                scope = None
                print("[ask] 已切回全库检索（未限定论文）")
            elif cmd == "/reset":
                qa.reset_history()
                print("[ask] 对话历史已清空")
            else:
                print("未知命令。可用：/papers /paper 关键词 /all /reset / q")
            continue
        _ask_once(qa, q, scope)


def _ask_once(qa: QASystem, question: str, scope: str | None = None) -> None:
    result = qa.ask(question, scope=scope)
    if scope:
        print(f"\n（当前论文：{scope}）")
    print("\n" + "=" * 56)
    print(f"问题: {result['question']}")
    print("-" * 56)
    print(result["answer"])
    print("-" * 56)
    for i, c in enumerate(result["citations"], start=1):
        verdict = c.get("verdict", "")
        mark = {"supported": "✓", "partial": "~", "unsupported": "✗"}.get(verdict, "")
        print(f"[{i}]{mark} {c['source']} | {c['section']} | 第{c['page']}页")
        print(f"    {c['snippet'][:80].replace(chr(10), ' ')}...")
    if result.get("verification"):
        print("-" * 56)
        print("引用校验（回答中每个 [n] 是否被片段支持）：")
        for v in result["verification"]:
            name = {"supported": "支持", "partial": "部分支持", "unsupported": "不支持"}.get(
                v["verdict"], v["verdict"])
            print(f"  [{v['cite']}] {name}：{v['reason']}")


if __name__ == "__main__":
    # 支持两种写法：python ask.py "问题"  或  python ask.py --paper 关键词 "问题"
    args = sys.argv[1:]
    paper = None
    if args and args[0] == "--paper" and len(args) >= 2:
        paper = args[1]
        args = args[2:]
    question = args[0] if args else None
    main(question, paper)
