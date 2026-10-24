# -*- coding: utf-8 -*-
"""向量库封装：Embedding 生成 + Chroma 存储 + 相似度检索。

Embedding 两种来源（用 .env 的 EMBED_PROVIDER 切换）：
1. ollama（默认）：调用本地 Ollama 的 /api/embed 接口
   —— 适合 bge-m3 这类 GGUF 格式模型（Ollama 部署），零下载、不联网
2. sentence_transformers：加载 HuggingFace 格式模型（本地目录或自动下载）
   —— 需要更细控制或没有 Ollama 时使用

为什么用专门向量库（面试常问）：
- Chroma 用 ANN 索引（HNSW），几万条数据也能毫秒级检索；
  自己用 numpy 算余弦相似度是 O(N) 全扫描，规模一大就慢；
- 索引数据落盘（persist），重启不用重新入库。
"""
import logging
from pathlib import Path

import chromadb
import requests

from ..config import CHROMA_DIR, EMBED_MODEL, EMBED_PROVIDER, OLLAMA_BASE_URL

# chroma 0.6.3 的遥测单例与新版 posthog 不兼容，每次调用都会刷错误日志；
# 直接禁用 posthog + 压掉 telemetry 日志，既清静也不会上报任何数据。
try:
    import posthog
    posthog.disabled = True
except ImportError:
    pass

logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
# chroma 0.6.3 打开已存在 collection 时，vector segment 会用 logger.warning
# 重放一批 "Add of existing embedding ID"（无害的一致性提示，每次启动出现）；
# 调到 ERROR 级，保持命令行输出干净。
logging.getLogger("chromadb.segment").setLevel(logging.ERROR)

BATCH_SIZE = 64  # 每批 embedding 的文本数，避免单次请求过大


class VectorStore:
    """向量库封装类。"""

    def __init__(self, embed_provider: str | None = None,
                 embed_model: str | None = None,
                 persist_dir: str | Path | None = None):
        self.provider = embed_provider or EMBED_PROVIDER
        self.embed_model = embed_model or EMBED_MODEL
        self.persist_dir = Path(persist_dir) if persist_dir else CHROMA_DIR
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # 按 provider 初始化 embedding 通道
        if self.provider == "ollama":
            # 不加载模型，直接调本地 Ollama API（前提：Ollama 服务已启动）
            print(f"[embed] 使用 Ollama 模型: {self.embed_model} @ {OLLAMA_BASE_URL}")
        elif self.provider == "sentence_transformers":
            from sentence_transformers import SentenceTransformer
            print(f"[embed] 加载模型: {self.embed_model} ...")
            self.encoder = SentenceTransformer(self.embed_model)
        else:
            raise ValueError(f"未知 EMBED_PROVIDER: {self.provider}，可选 ollama / sentence_transformers")

        # PersistentClient：数据持久化到磁盘目录
        # anonymized_telemetry=False：关闭遥测上报，避免每次调用刷错误日志
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name="papers",
            metadata={"hnsw:space": "cosine"},  # 余弦相似度衡量语义相近
        )

    def _encode(self, texts: list[str]) -> list[list[float]]:
        """把文本列表批量转成向量列表。"""
        if self.provider == "ollama":
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/embed",
                json={"model": self.embed_model, "input": texts},
                timeout=300,  # 首次调用会加载模型到内存，可能较慢
            )
            resp.raise_for_status()
            return resp.json()["embeddings"]
        # sentence_transformers 分支
        return self.encoder.encode(texts, normalize_embeddings=True).tolist()

    def add_chunks(self, chunks: list) -> None:
        """批量写入 Chunk 列表。

        每个 Chunk 需要有: chunk_id / content / source / section / page。
        """
        ids = [c.chunk_id for c in chunks]
        contents = [c.content for c in chunks]
        metadatas = [
            {"source": c.source, "section": c.section, "page": c.page}
            for c in chunks
        ]
        # 分批 embedding + 写入，防止单次请求过大
        for i in range(0, len(chunks), BATCH_SIZE):
            sub_contents = contents[i:i + BATCH_SIZE]
            vectors = self._encode(sub_contents)
            self.collection.add(
                ids=ids[i:i + BATCH_SIZE],
                documents=sub_contents,
                embeddings=vectors,
                metadatas=metadatas[i:i + BATCH_SIZE],
            )
        print(f"[embed] 已入库 {len(chunks)} 个块")

    def search(self, query: str, k: int = 5,
               where: dict | None = None) -> dict:
        """检索与 query 最相似的 k 个块。

        Args:
            query: 查询文本。
            k: 返回候选数。
            where: Chroma metadata 过滤条件，如 {"source": "xxx.pdf"}，
                用于只检索某一篇论文（防止跨论文串台）。

        Returns:
            Chroma 返回的 dict，包含 ids / documents / metadatas / distances。
        """
        q_vec = self._encode([query])
        kwargs = {"query_embeddings": q_vec, "n_results": k}
        if where:
            kwargs["where"] = where
        return self.collection.query(**kwargs)

    def reset(self) -> None:
        """清空整个向量库（删除并重建 collection）。

        为什么不用逐 id delete：Chroma 0.6.x 在大批量 delete 后重写
        HNSW 持久化文件偶发 "Cannot open header file" 崩溃（实测复现），
        删除重建 collection 是物理级清理，最干净也最稳。
        """
        try:
            self.client.delete_collection("papers")
        except Exception:
            pass
        self.collection = self.client.create_collection(
            name="papers",
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        """当前库里的块总数。"""
        return self.collection.count()
