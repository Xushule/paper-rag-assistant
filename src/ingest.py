# -*- coding: utf-8 -*-
"""数据管道入口（M1 里程碑）。

用法：
    python src/ingest.py                    # 处理 data/papers/ 下所有 PDF
    python src/ingest.py D:\某个文件夹      # 处理指定目录

流程：
    PDF 解析 -> 结构感知分块 -> Embedding 入库（Chroma）-> 构建 BM25 索引

跑通后验证：
    python scripts/verify_retrieval.py
"""
import sys
from pathlib import Path

# 把项目根目录加进模块搜索路径，保证「python src/ingest.py」能 import src.*
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PAPERS_DIR
from src.parser.chunker import chunk_pdf
from src.retrieval.bm25_index import build_index
from src.retrieval.embed_store import VectorStore


def main(papers_dir: Path = PAPERS_DIR, parser: str = "auto") -> None:
    papers_dir.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(papers_dir.glob("*.pdf"))
    if not pdfs:
        print(f"[ingest] 没有找到 PDF，请把论文放到: {papers_dir}")
        return

    # 1) 逐篇解析 + 分块
    all_chunks = []
    for pdf in pdfs:
        print(f"[ingest] 处理: {pdf.name}")
        chunks = chunk_pdf(pdf, parser=parser)
        print(f"         分块数: {len(chunks)}")
        all_chunks.extend(chunks)

    if not all_chunks:
        print("[ingest] 所有 PDF 都没解析出文本，请检查是否为扫描版（无文字层）")
        return

    # 2) Embedding 入库（会下载模型，首次较慢）
    store = VectorStore()
    # 重建索引语义 = 清空旧向量再写入；否则上次入库中"已不存在"的块
    # （如被伪表过滤拒绝的块）会以旧 chunk_id 残留在 Chroma 里，
    # 检索时仍会被命中，污染结果（Chroma add 对旧 id 是 upsert，不删除缺失 id）。
    try:
        store.reset()
    except Exception as e:
        print(f"[ingest] 清空旧向量失败（忽略，继续追加）: {e}")
    store.add_chunks(all_chunks)
    print(f"[ingest] Chroma 向量库现有 {store.count()} 个块")

    # 3) 构建 BM25 索引（存到 data/bm25.pkl）
    build_index(all_chunks)
    print(f"[ingest] BM25 索引已构建（{len(all_chunks)} 个块）")

    print("\n[ingest] 完成。下一步：python scripts/verify_retrieval.py")


if __name__ == "__main__":
    args = sys.argv[1:]
    parser_mode = "auto"
    dir_arg = None
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--parser" and i + 1 < len(args):
            parser_mode = args[i + 1]
            i += 2
            continue
        if a.startswith("--parser="):
            parser_mode = a.split("=", 1)[1]
            i += 1
            continue
        if not a.startswith("-"):
            dir_arg = a
        i += 1
    main(Path(dir_arg) if dir_arg else PAPERS_DIR, parser=parser_mode)
