#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python -m benchmarks.measurement.run_measurement \
  --config benchmarks/configs/smoke_measurement.yaml \
  --output-dir benchmarks/results/smoke_measurement \
  --repetitions 3 \
  --noop
