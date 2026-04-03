# ─── CropAI Dockerfile ────────────────────────────────────────────────────────
# Multi-stage build for FastAPI + Streamlit

FROM python:3.10-slim AS base

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Stage 1: Install Python deps ─────────────────────────────────────────────
FROM base AS deps
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Stage 2: Application ─────────────────────────────────────────────────────
FROM deps AS app

COPY . .

# Create required directories
RUN mkdir -p data models/saved results logs

# Expose ports: FastAPI (8000) + Streamlit (8501)
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default: start FastAPI
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
