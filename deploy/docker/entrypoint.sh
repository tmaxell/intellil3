#!/usr/bin/env sh
# Entrypoint for the IntelliL3 gRPC service container.
# Waits for MinIO to be ready, ensures the target bucket exists, then starts
# the server. All settings can be overridden via environment variables.

set -e

L3_ENDPOINT_URL="${L3_ENDPOINT_URL:-http://minio:9000}"
L3_ACCESS_KEY="${L3_ACCESS_KEY:-minioadmin}"
L3_SECRET_KEY="${L3_SECRET_KEY:-minioadmin}"
L3_BUCKET="${L3_BUCKET:-l3-llm-store}"
L3_CONFIG="${L3_CONFIG:-/app/configs/docker.yaml}"
L3_HOST="${L3_HOST:-[::]}"
L3_PORT="${L3_PORT:-50051}"
L3_WORKERS="${L3_WORKERS:-10}"
L3_LOG_LEVEL="${L3_LOG_LEVEL:-INFO}"

# ── 1. Wait for MinIO ─────────────────────────────────────────────────────────
echo "[entrypoint] Waiting for MinIO at ${L3_ENDPOINT_URL} ..."
MAX_TRIES=30
i=0
until curl -sf "${L3_ENDPOINT_URL}/minio/health/live" > /dev/null 2>&1; do
    i=$((i + 1))
    if [ "$i" -ge "$MAX_TRIES" ]; then
        echo "[entrypoint] ERROR: MinIO did not become ready in time." >&2
        exit 1
    fi
    sleep 2
done
echo "[entrypoint] MinIO is ready."

# ── 2. Create bucket if it does not exist ─────────────────────────────────────
python - <<EOF
import boto3, os
from botocore.exceptions import ClientError

s3 = boto3.client(
    "s3",
    endpoint_url="${L3_ENDPOINT_URL}",
    aws_access_key_id="${L3_ACCESS_KEY}",
    aws_secret_access_key="${L3_SECRET_KEY}",
    region_name="us-east-1",
)
bucket = "${L3_BUCKET}"
try:
    s3.head_bucket(Bucket=bucket)
    print(f"[entrypoint] Bucket '{bucket}' already exists.")
except ClientError as e:
    if e.response["Error"]["Code"] in ("404", "NoSuchBucket"):
        s3.create_bucket(Bucket=bucket)
        print(f"[entrypoint] Bucket '{bucket}' created.")
    else:
        raise
EOF

# ── 3. Write a runtime config that picks up env-var overrides ─────────────────
python - <<EOF
import yaml, os, pathlib

cfg_path = pathlib.Path("${L3_CONFIG}")
cfg = yaml.safe_load(cfg_path.read_text())

cfg["storage"]["endpoint_url"]  = os.environ.get("L3_ENDPOINT_URL", cfg["storage"]["endpoint_url"])
cfg["storage"]["access_key"]    = os.environ.get("L3_ACCESS_KEY",   cfg["storage"]["access_key"])
cfg["storage"]["secret_key"]    = os.environ.get("L3_SECRET_KEY",   cfg["storage"]["secret_key"])
cfg["storage"]["bucket"]        = os.environ.get("L3_BUCKET",       cfg["storage"]["bucket"])

runtime_cfg = pathlib.Path("/tmp/l3_runtime.yaml")
runtime_cfg.write_text(yaml.dump(cfg))
print(f"[entrypoint] Runtime config written to {runtime_cfg}")
EOF

# ── 4. Start the gRPC server ──────────────────────────────────────────────────
echo "[entrypoint] Starting IntelliL3 gRPC server on port ${L3_PORT} ..."
exec l3-store serve \
    --config /tmp/l3_runtime.yaml \
    --host  "${L3_HOST}" \
    --port  "${L3_PORT}" \
    --max-workers "${L3_WORKERS}" \
    --log-level   "${L3_LOG_LEVEL}"
