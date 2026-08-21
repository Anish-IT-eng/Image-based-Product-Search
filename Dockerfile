# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — VisualFind Backend (FastAPI + ResNet50 + FAISS)
# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 (builder): install heavy Python deps in a clean layer
# Stage 2 (runtime): minimal image with only what's needed to run the API
#
# Build:   docker build -t visualfind-backend .
# Run:     docker run -p 8000:8000 -v $(pwd)/backend/data:/app/data visualfind-backend
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# System deps required by torch / faiss-cpu / Pillow / OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy and install Python dependencies into an isolated prefix
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="VisualFind" \
      description="Image-based Product Search API — ResNet50 + FAISS" \
      version="1.0.0"

# Minimal runtime system libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy backend source
COPY backend/ .

# backend/data/ will be mounted as a volume at runtime (contains index.faiss + catalog.json)
# We create the directory so the container starts cleanly even without a mount.
RUN mkdir -p /app/data

# Non-root user for security
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --no-create-home appuser && \
    chown -R appuser:appgroup /app
USER appuser

# ── Environment defaults (override via docker-compose or -e flags) ────────────
ENV API_HOST=0.0.0.0 \
    API_PORT=8000 \
    LOG_LEVEL=info \
    MAX_IMAGES=1600 \
    EMBEDDING_BATCH_SIZE=16 \
    CORS_ORIGINS=http://localhost:8080,http://localhost:3000

EXPOSE 8000

# Health check — Docker will restart unhealthy containers
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# ── Entrypoint ────────────────────────────────────────────────────────────────
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--log-level", "info"]
