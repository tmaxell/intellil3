"""Entry point for ``python -m benchmarks.sanity``."""
from __future__ import annotations

import argparse

from benchmarks.sanity.llm_sanity_check import (
    LLMSanityChecker,
    SanityCheckConfig,
    _ALL_PATTERNS,
    _PATTERN_GENERATORS,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM sanity-check for IntelliL3 workload patterns."
    )
    parser.add_argument(
        "--mode",
        choices=["dry_run", "ollama"],
        default="dry_run",
        help="dry_run: analyse prompts only. ollama: call a local Ollama instance.",
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        choices=list(_PATTERN_GENERATORS),
        default=list(_ALL_PATTERNS),
        help="Patterns to check (default: all three).",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=5,
        help="Number of prompts per pattern.",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmarks/results/sanity",
        help="Directory for JSON and Markdown reports.",
    )
    parser.add_argument(
        "--model",
        default="llama3.2:1b",
        help="Ollama model name (ignored in dry_run mode).",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama base URL.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = SanityCheckConfig(
        mode=args.mode,
        patterns=args.patterns,
        num_prompts=args.num_prompts,
        output_dir=args.output_dir,
        ollama_model=args.model,
        ollama_base_url=args.ollama_url,
        seed=args.seed,
    )
    report = LLMSanityChecker(config).run()

    print(f"\n{'Pattern':<20} {'Cache reuse':<14} {'Prefix overlap':>14}")
    print("-" * 52)
    for pr in report.patterns:
        print(
            f"{pr.pattern:<20} {pr.cache_reuse_potential:<14} "
            f"{pr.prefix_overlap_ratio:>13.1%}"
        )
    print(f"\nReports saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
