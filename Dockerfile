FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install dependencies required by the standalone pipeline.
COPY pipeline/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

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