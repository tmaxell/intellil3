#!/usr/bin/env bash
# Run the optional LLM sanity-check for workload patterns.
#
# Usage:
#   bash scripts/run_llm_sanity_check.sh [dry_run|ollama] [output_dir] [model]
#
# Examples:
#   bash scripts/run_llm_sanity_check.sh
#   bash scripts/run_llm_sanity_check.sh dry_run benchmarks/results/sanity
#   bash scripts/run_llm_sanity_check.sh ollama benchmarks/results/sanity llama3.2:1b

set -euo pipefail

MODE="${1:-dry_run}"
OUTPUT_DIR="${2:-benchmarks/results/sanity}"
MODEL="${3:-llama3.2:1b}"

echo "=== LLM Sanity Check ==="
echo "Mode:       ${MODE}"
echo "Output:     ${OUTPUT_DIR}"
if [ "${MODE}" = "ollama" ]; then
    echo "Model:      ${MODEL}"
fi
echo ""

python -m benchmarks.sanity \
    --mode "${MODE}" \
    --output-dir "${OUTPUT_DIR}" \
    --model "${MODEL}"
