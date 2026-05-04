# ── Build stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build deps
RUN pip install --no-cache-dir hatchling

# Copy package metadata and source
COPY pyproject.toml .
COPY src/ src/

# Build wheel (core + gRPC extras only; embeddings are optional)
RUN pip wheel --no-cache-dir --wheel-dir /wheels ".[grpc]" .

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

LABEL org.opencontainers.image.title="IntelliL3"
LABEL org.opencontainers.image.description="Intelligent three-tier KV-cache / RAG object store for LLM inference"
LABEL org.opencontainers.image.source="https://github.com/tmaxell/intellil3"

WORKDIR /app

# Install awscli-style client for bucket init + runtime wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links /wheels \
        l3-llm-store \
    && rm -rf /wheels

# Copy application config and entrypoint
COPY configs/ configs/
COPY deploy/docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# gRPC port
EXPOSE 50051

# Allow overriding every setting via environment variables:
#   L3_ENDPOINT_URL, L3_ACCESS_KEY, L3_SECRET_KEY, L3_BUCKET,
#   L3_HOST, L3_PORT, L3_WORKERS, L3_LOG_LEVEL
ENV L3_CONFIG=/app/configs/docker.yaml \
    L3_HOST="[::]" \
    L3_PORT=50051 \
    L3_WORKERS=10 \
    L3_LOG_LEVEL=INFO

HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=5 \
    CMD python - <<'EOF'
import grpc, sys
from l3store.api.proto import l3_service_pb2 as pb, l3_service_pb2_grpc as svc
import os
ch = grpc.insecure_channel(f"localhost:{os.environ.get('L3_PORT','50051')}")
stub = svc.L3ServiceStub(ch)
try:
    stub.GetStats(pb.GetStatsRequest(), timeout=3)
    sys.exit(0)
except Exception:
    sys.exit(1)
EOF

ENTRYPOINT ["/entrypoint.sh"]
