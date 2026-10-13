# -*- coding: utf-8 -*-
"""全局配置：集中管理路径、模型选择。

设计说明：所有"会变的东西"（路径、模型名、API Key）都从这里读，
代码里不写死任何路径。这样换机器、换模型只需要改 .env。
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录 = src 的上一级（D:\论文阅读问答助手）
ROOT_DIR = Path(__file__).resolve().parent.parent

# 加载 .env（文件不存在也不会报错，只是读不到自定义配置）
load_dotenv(ROOT_DIR / ".env")

# ---------- 路径 ----------
DATA_DIR = ROOT_DIR / "data"
PAPERS_DIR = DATA_DIR / "papers"        # 放论文 PDF
# 向量库持久化目录。2026-10-08 表格结构化改造后旧库（chroma_db，混排版）
# 与新版分块不兼容；为避免删除操作，新库改用 chroma_db_v2，旧目录原样保留，
# 确认新库稳定后可手动删除 data/chroma_db。
CHROMA_DIR = DATA_DIR / "chroma_db_v2"
BM25_PATH = DATA_DIR / "bm25.pkl"       # BM25 索引缓存（重建时覆盖写）
CHUNK_META_PATH = DATA_DIR / "chunks.pkl"  # 分块元数据缓存（重建时覆盖写）

# ---------- Embedding ----------
# 来源二选一：
#   ollama（默认）          : 调用本地 Ollama 的 /api/embed 接口，适合 bge-m3 这类 GGUF 模型
#   sentence_transformers   : 加载 HuggingFace 格式模型（本地目录或自动下载）
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "ollama")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# 精排模型（cross-encoder）：与查询+文档联合编码打分，比双塔 embedding 精确
RERANK_MODEL = os.getenv("RERANK_MODEL", "awenleven/bge-reranker-v2-m3")  # Ollama 社区版（BAAI 原模型转 GGUF），官方库无此模型
# 精排方式：llm（默认，DeepSeek 打分，paper-qa 同款）/ ollama（cross-encoder，Ollama 官方无 /api/rerank 端点，暂不可用）
RERANK_MODE = os.getenv("RERANK_MODE", "llm")
# 无边框表补漏开关：解析时用 LLM 判定/结构化无边框数值表（需要 LLM API Key）；
# 关闭后离线解析，无边框表内容仍走文本流（结构丢失）
TABLE_LLM = os.getenv("TABLE_LLM", "1") == "1"
# 查询扩展开关：问答检索前用 LLM 把问题扩成多个子查询、多路检索后跨路 RRF 融合，
# 提升"专有名词/不同表述"的召回；关闭后只用原问题单路检索
QUERY_EXPAND = os.getenv("QUERY_EXPAND", "1") == "1"
# MMR 多样性选择：RRF 融合后的候选先用 MMR（相关度 − 与已选片段冗余度）挑一批
# 多样化的，再交给 LLM 精排。解决全库对比时候选被少数论文垄断（实测问"这里有几篇
# 论文"时 NightRain 0 块进候选）。关闭后走原链路（RRF -> 直接精排）
MMR_ENABLED = os.getenv("MMR_ENABLED", "1") == "1"
MMR_TOP_K = int(os.getenv("MMR_TOP_K", "8"))   # MMR 从 RRF top12 里选几个进精排
MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", "0.7"))  # 相关度权重；越小越重多样性

# ---------- LLM（M2 问答接口才用，先预留） ----------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek")  # deepseek / local(Ollama)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
# 本地 Ollama 模式示例（与 LLM_BASE_URL=http://localhost:11434/v1 配套）：
# LLM_PROVIDER=local, LLM_MODEL=qwen3:8b


def _resolve_api_key() -> str:
    """解析 LLM API Key：优先 .env 的 LLM_API_KEY；无效（占位/中文）时回退系统环境变量。"""
    val = os.getenv("LLM_API_KEY", "").strip()
    if val and "你的Key" not in val and all(ord(ch) < 128 for ch in val):
        return val
    # 本机已配置在系统环境变量 Deepseek_Key（Machine 级，随会话继承）
    return os.getenv("Deepseek_Key", "").strip()


LLM_API_KEY = _resolve_api_key()

# ---------- 分块参数 ----------
# 近似 token 估算：英文约 4 字符/token，中文约 1.5 字符/token，取折中 3
CHARS_PER_TOKEN = 3.0
TARGET_CHARS = 1800     # ≈ 500~600 token
OVERLAP_CHARS = 180     # ≈ 50~60 token 重叠
