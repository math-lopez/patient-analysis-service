FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/tmp/huggingface
ENV HF_HUB_CACHE=/tmp/huggingface/hub
ENV XDG_CACHE_HOME=/tmp

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/models

ARG MODEL_REPO="bartowski/Qwen2.5-3B-Instruct-GGUF"
ARG MODEL_FILE="Qwen2.5-3B-Instruct-Q4_K_M.gguf"

RUN curl -L "https://huggingface.co/${MODEL_REPO}/resolve/main/${MODEL_FILE}" \
    -o /app/models/model.gguf

COPY . .

ENV MODEL_PATH=/app/models/model.gguf

CMD ["sh", "-c", "python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]