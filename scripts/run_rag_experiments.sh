#!/usr/bin/env bash
# Run realistic RAG experiments (exp10+).
#
# Usage:
#   bash scripts/run_rag_experiments.sh
#   REPETITIONS=3 bash scripts/run_rag_experiments.sh
#   OUTPUT_DIR=my/dir bash scripts/run_rag_experiments.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON:-$ROOT_DIR/.venv/bin/python}"
REPETITIONS="${REPETITIONS:-5}"
CONFIG_GLOB="${CONFIG_GLOB:-benchmarks/configs/exp1[0-9]_*.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-benchmarks/results/rag}"

"$PYTHON_BIN" -m benchmarks.measurement.run_all_comparisons \
  --config-glob "$CONFIG_GLOB" \
  --output-dir "$OUTPUT_DIR" \
  --repetitions "$REPETITIONS"
