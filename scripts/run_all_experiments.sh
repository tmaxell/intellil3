#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON:-$ROOT_DIR/.venv/bin/python}"
REPETITIONS="${REPETITIONS:-5}"
CONFIG_GLOB="${CONFIG_GLOB:-benchmarks/configs/exp*.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-benchmarks/results}"

"$PYTHON_BIN" -m benchmarks.measurement.run_all_comparisons \
  --config-glob "$CONFIG_GLOB" \
  --output-dir "$OUTPUT_DIR" \
  --repetitions "$REPETITIONS"
