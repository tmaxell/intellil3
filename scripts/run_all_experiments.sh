#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON:-$ROOT_DIR/.venv/bin/python}"
REPETITIONS="${REPETITIONS:-5}"

"$PYTHON_BIN" -m benchmarks.measurement.run_all_comparisons \
  --config-glob "benchmarks/configs/exp*.yaml" \
  --output-dir benchmarks/results \
  --repetitions "$REPETITIONS"
