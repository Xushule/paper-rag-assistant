# -*- coding: utf-8 -*-
"""FastAPI 问答接口 + 语料库管理（Web 界面后端）。

启动（项目根目录下）：
    & D:\anaconda\envs\paper_rag\python.exe -m uvicorn api.main:app --port 8000

接口：
    GET    /                    Web 界面
    GET    /health              健康检查（返回库内块数）
    GET    /api/papers          论文列表（供作用域下拉）
    POST   /api/qa              问答 {"question": "...", "paper": "xxx.pdf"}
    POST   /api/papers/upload   multipart 上传 PDF（增量入库）
    DELETE /api/papers/{name}   删除论文（从文件 + 向量库 + 索引移除）

增量入库/删除设计：
- 复用已加载 QASystem 的同一个 Chroma 客户端做增删（同一进程内多客户端
  打开同一目录会互相锁文件，这是历史踩过的坑），完成后原地更新 bm25/chunks；
- build_index() 同时重写 bm25.pkl + chunks.pkl，供下次启动/其他进程读取；
- 上传失败自动回滚（删除已写入的块 + 删除文件），不留脏状态。
"""
import re
import sys
import threading
from pathlib import Path

# 保证能 import src.*（无论从哪个目录启动）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.config import PAPERS_DIR
from src.parser.chunker import chunk_pdf
from src.qa.answer import QASystem
from src.retrieval.bm25_index import build_index, load_index
from src.retrieval.embed_store import VectorStore

app = FastAPI(title="论文阅读问答助手 API", version="0.4.0")

# 文件名白名单：只允许常规 PDF 文件名，杜绝路径穿越（../、绝对路径等）。
# 允许点号（arXiv 编号 "Restormer_2111.09839.pdf" 自带点号）；点号前后必须有
# 合法字符（"[A-Za-z0-9_...]+\.[...]+"），".."、"..pdf"、"../x" 均无法匹配。
_SAFE_NAME = re.compile(
    r"^[A-Za-z0-9_\-\u4e00-\u9fff]+(\.[A-Za-z0-9_\-\u4e00-\u9fff]+)*\.pdf$")


class QARequest(BaseModel):
    question: str
    paper: str | None = None  # 论文作用域（PDF 文件名，见 /api/papers）；None=全库
    session_id: str | None = None  # 会话标识；同一会话共享多轮历史，不同会话隔离


class QAResponse(BaseModel):
    question: str
    answer: str
    citations: list[dict]
    retrieved: int
    verification: list[dict] | None = None  # 引用校验（posthoc attribution）结果
    revised: bool = False                   # 是否发生过校验反馈再检索（self-correction）


# 全局只加载一次（向量库 + BM25 + 分块元数据），避免每个请求都重新加载
_qa: QASystem | None = None
# 写锁：上传/删除期间互斥，避免并发写库；问答读旧状态不受影响
_write_lock = threading.Lock()


def _get_qa() -> QASystem:
    global _qa
    if _qa is None:
        store = VectorStore()
        bm25, chunks = load_index()
        _qa = QASystem(store, bm25, chunks)
    return _qa


def _refresh_qa_after_write() -> None:
    """写库完成后，从磁盘重读最新 bm25/chunks 原地替换（复用同一 chroma 实例）。"""
    global _qa
    if _qa is None:
        return
    bm25, chunks = load_index()
    _qa.bm25 = bm25
    _qa.chunks = chunks
    _qa.reset_history()


def _safe_pdf_name(name: str | None) -> str:
    """校验并规范化上传/删除的文件名，非法则 400。"""
    if not name:
        raise HTTPException(status_code=400, detail="缺少文件名")
    pure = Path(name).name  # 去路径，防穿越
    if not _SAFE_NAME.match(pure):
        raise HTTPException(status_code=400, detail=f"非法文件名：{name}")
    return pure


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Web 界面（浏览器打开 http://localhost:8000 即可使用）。"""
    html = Path(__file__).resolve().parent.parent / "web" / "index.html"
    return FileResponse(html)


@app.get("/health")
def health() -> dict:
    qa = _get_qa()
    return {"status": "ok", "chunks": qa.store.count()}


@app.get("/api/papers")
def papers() -> dict:
    """论文列表（供前端作用域下拉 + 语料库管理），附每篇块数。"""
    qa = _get_qa()
    names = sorted({c.source for c in qa.chunks})
    counts = {}
    for c in qa.chunks:
        counts[c.source] = counts.get(c.source, 0) + 1
    return {"papers": names, "counts": counts}


@app.post("/api/qa", response_model=QAResponse)
def qa_endpoint(req: QARequest) -> QAResponse:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question 不能为空")
    try:
        # session_id 缺省为 "default"：兼容 CLI/旧客户端（单会话共享历史）
        result = _get_qa().ask(req.question, scope=req.paper,
                               session=req.session_id or "default")
    except Exception as e:  # 网络/超时等统一返回 502，便于前端识别
        raise HTTPException(status_code=502, detail=f"问答服务出错: {e}")
    return QAResponse(**result)


@app.post("/api/papers/upload")
async def upload_paper(file: UploadFile = File(...)) -> dict:
    """上传新论文 PDF：保存 -> 解析分块 -> 增量入库 -> 重建 BM25。"""
    name = _safe_pdf_name(file.filename)
    dest = PAPERS_DIR / name
    if dest.exists():
        raise HTTPException(status_code=400, detail=f"论文已存在：{name}（如需替换请先删除）")

    with _write_lock:
        content = await file.read()
        dest.write_bytes(content)
        chunks = None
        try:
            chunks = chunk_pdf(dest)          # 解析 + 分块（含无边框表 LLM 结构化）
            if not chunks:
                raise HTTPException(status_code=422,
                                    detail="未能从 PDF 解析出文本（可能是扫描版，无文字层）")
            qa = _get_qa()
            qa.store.add_chunks(chunks)       # 复用现有 chroma 客户端增量写
            all_chunks = qa.chunks + chunks
            build_index(all_chunks)           # 重写 bm25.pkl + chunks.pkl
            _refresh_qa_after_write()
        except HTTPException:
            raise
        except Exception as e:
            # 回滚：删除已写入的块（若 embedding 中途失败）与文件
            try:
                if chunks:
                    _get_qa().store.collection.delete(ids=[c.chunk_id for c in chunks])
            except Exception:
                pass
            if dest.exists():
                dest.unlink()
            raise HTTPException(status_code=500, detail=f"入库失败：{e}")
    return {"name": name, "added": len(chunks), "total": len(all_chunks)}


@app.delete("/api/papers/{filename}")
def delete_paper(filename: str) -> dict:
    """删除论文：向量库移除该 source 的块 + 重建索引 + 删除文件。"""
    name = _safe_pdf_name(filename)
    dest = PAPERS_DIR / name
    if not dest.exists():
        raise HTTPException(status_code=404, detail=f"论文不存在：{name}")

    with _write_lock:
        try:
            qa = _get_qa()
            qa.store.collection.delete(where={"source": name})
            keep = [c for c in qa.chunks if c.source != name]
            build_index(keep)                 # 重写 bm25.pkl + chunks.pkl
            dest.unlink()
            _refresh_qa_after_write()
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"删除失败：{e}")
    return {"deleted": name, "total": len(keep)}
