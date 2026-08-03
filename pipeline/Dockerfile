# pipeline/Dockerfile
FROM python:3.11-slim

# Base build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Code path (only code & dependencies stored)
WORKDIR /app

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install dependencies first (use cache)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# === Optional: include example data in the image (remove these two lines if not needed) ===
# It's recommended to add `data/` to .dockerignore to avoid COPY . overwriting/duplicating it
# COPY data/ /app/data/

# Copy code
COPY . /app/

# Runtime working directory to /work (writable, mount by user)
RUN mkdir -p /work
WORKDIR /work
VOLUME ["/work"]

# Entrypoint: main.py (use absolute path, also works with Apptainer)
ENTRYPOINT ["python", "/app/main.py"]
CMD ["--help"]