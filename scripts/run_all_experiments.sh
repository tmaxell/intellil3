#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

for config in benchmarks/configs/exp*.yaml; do
  name="$(basename "$config" .yaml)"
  output="benchmarks/results/${name}/results.json"
  report="benchmarks/results/${name}/report.md"
  python -m benchmarks.runners.benchmark_runner \
    --config "$config" \
    --output "$output" \
    --noop
  python -m benchmarks.analysis.report \
    --results "$output" \
    --output "$report"
done
