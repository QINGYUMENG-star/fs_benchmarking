# pipeline/Dockerfile
FROM python:3.11-slim

# 基础构建依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# 代码路径（仅存放代码 & 依赖）
WORKDIR /app

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# 先装依赖（利用缓存）
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# === 可选：把「示例数据」打进镜像（如不需要就删除这两行）===
# 建议在 .dockerignore 里写上 `data/`，避免被下面的 COPY . 覆盖/重复
# COPY data/ /app/data/

# 拷贝代码
COPY . /app/

# 运行时工作目录切到 /work（可写，要求用户绑定）
RUN mkdir -p /work
WORKDIR /work
VOLUME ["/work"]

# 入口：main.py（用绝对路径，Apptainer 也稳）
ENTRYPOINT ["python", "/app/main.py"]
CMD ["--help"]