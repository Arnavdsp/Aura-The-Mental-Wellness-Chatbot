# syntax=docker/dockerfile:1
#
# Default build: the full service on the echo engine — small, CPU-only, and
# useful immediately. For Gemma 3n inference, build with:
#     docker build --build-arg EXTRAS='[ml]' -t aura:gpu .
# and run on a CUDA host with `--gpus all`.

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ffmpeg lets the server decode the WebM/Opus that browsers record.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so edits to source don't invalidate the install layer.
ARG EXTRAS=""
COPY pyproject.toml README.md ./
COPY src/aura/__init__.py src/aura/config.py ./src/aura/
RUN pip install --no-cache-dir -e ".${EXTRAS}"

COPY src/ ./src/
COPY web/ ./web/

# Run unprivileged.
RUN useradd --create-home --uid 10001 aura && chown -R aura:aura /app
USER aura

ENV AURA_HOST=0.0.0.0 \
    AURA_PORT=8000 \
    AURA_ENVIRONMENT=production \
    AURA_LOG_JSON=true \
    HF_HOME=/home/aura/.cache/huggingface

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "aura.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
