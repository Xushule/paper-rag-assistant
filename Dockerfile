# 论文阅读问答助手 — Docker 镜像
# 用法（项目根目录）：docker compose up --build -d
# 依赖的外部服务：
#   - Ollama（宿主机 11434，提供 bge-m3 embedding）→ 容器内经 host.docker.internal 访问
#   - DeepSeek API（LLM 生成/精排/校验）
FROM python:3.10-slim

# 时区 + 常用工具（pdfplumber 需要 libxml/libxslt；python-docx 无系统依赖）
ENV TZ=Asia/Shanghai \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ChromaDB 0.6.3 需要 sqlite 较新版本，python:3.10-slim（Debian bookworm）自带满足。
# 如遇 sqlite 版本过旧，可 uncomment 下一行安装 libsqlite3-0 新版：
# RUN apt-get update && apt-get install -y --no-install-recommends libsqlite3-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先装依赖（利用层缓存，源码改动不触发重装）。
# 运行时用精简清单（requirements-runtime.txt），评测依赖（ragas/datasets）留在本地环境。
COPY requirements-runtime.txt requirements.txt .
RUN pip install -r requirements-runtime.txt

# 拷贝项目（data/ 由 compose 挂载覆盖，.dockerignore 已排除）
COPY . .

# 容器内默认从环境变量读取配置（LLM_API_KEY / OLLAMA_BASE_URL 由 compose 注入）
EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
