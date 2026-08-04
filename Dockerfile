FROM python:3.11-slim

# Build-time options.
# CPU build example:
#   --build-arg IMAGE_VARIANT=cpu \
#   --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
#
# NVIDIA GPU build example:
#   --build-arg IMAGE_VARIANT=gpu \
#   --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
ARG IMAGE_VARIANT=cpu
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_VERSION=

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIPELINE_IMAGE_VARIANT=${IMAGE_VARIANT}

# Copy the dependency list first so Docker can cache dependency installation.
COPY pipeline/requirements.txt /app/requirements.txt

# Install the requested CPU or CUDA build of PyTorch first.
# The standalone `torch` entry is then removed from the general requirements
# file so that pip does not replace it with a different PyTorch build.
RUN python -m pip install --upgrade pip \
 && if [ -n "$TORCH_VERSION" ]; then \
      python -m pip install "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX_URL}"; \
    else \
      python -m pip install torch --index-url "${TORCH_INDEX_URL}"; \
    fi \
 && sed -E '/^[[:space:]]*torch([[:space:]]*([<>=!~].*)?)?[[:space:]]*$/d' \
      /app/requirements.txt > /app/requirements-without-torch.txt \
 && python -m pip install -r /app/requirements-without-torch.txt \
 && python -c "import torch; print('Installed torch:', torch.__version__); print('CUDA build:', torch.version.cuda)"

# Copy the standalone analysis pipeline, including the bundled example dataset,
# into the image. The example dataset will be available at:
# /app/data/ALLAML_10.npz
COPY pipeline/ /app/

# Fail the build if the bundled example dataset was not copied into the image.
RUN test -f /app/data/ALLAML_10.npz

# Optional mount points for user-provided input data and persistent results.
RUN mkdir -p /data /results

# Run the command-line pipeline by default.
ENTRYPOINT ["python", "/app/main.py"]
CMD ["--help"]