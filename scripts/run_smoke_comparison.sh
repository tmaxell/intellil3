#!/usr/bin/env bash
set -euo pipefail

python -m benchmarks.measurement.run_comparison \
  --config benchmarks/configs/smoke_comparison.yaml \
  --output-dir benchmarks/results/smoke_comparison \
  --repetitions 5
