from __future__ import annotations

import argparse
import csv
import glob
from io import StringIO
from pathlib import Path

from benchmarks.measurement.comparison import SystemComparison
from benchmarks.measurement.run_comparison import _parse_targets, run_comparison


CORE_KEY_METRICS = [
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_p99_ms",
    "throughput_req_s",
    "cache_hit_rate",
    "prefetch_use_rate",
]

AGENTIC_KEY_METRICS = [
    "workflow_cache_hit_rate",
    "tool_cache_hit_rate",
    "plan_cache_hit_rate",
    "prefetch_precision",
    "prefetch_recall",
    "kv_reload_count",
    "kv_recompute_count",
    "recompute_avoided",
    "tool_wait_hidden_ms",
    "latency_saved_ms",
    "wasted_prefetch_bytes",
    "l2_eviction_count",
    "l3_read_count",
    "l3_write_count",
]

KEY_METRICS = CORE_KEY_METRICS + AGENTIC_KEY_METRICS


def run_all_comparisons(
    config_glob: str,
    output_dir: str | Path,
    repetitions: int = 5,
    targets: dict[str, float] | None = None,
) -> list[SystemComparison]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    config_paths = [Path(path) for path in sorted(glob.glob(config_glob))]
    if not config_paths:
        raise ValueError(f"No benchmark configs matched: {config_glob}")

    comparisons = []
    for config_path in config_paths:
        experiment_output = output_path / config_path.stem
        comparisons.append(
            run_comparison(
                config_path=config_path,
                output_dir=experiment_output,
                repetitions=repetitions,
                targets=targets,
            )
        )

    (output_path / "comparison_index.csv").write_text(
        render_comparison_index_csv(comparisons),
        encoding="utf-8",
    )
    (output_path / "comparison_index.md").write_text(
        render_comparison_index_markdown(comparisons),
        encoding="utf-8",
    )
    return comparisons


def render_comparison_index_csv(comparisons: list[SystemComparison]) -> str:
    buffer = StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=[
            "experiment_name",
            "metric",
            "metric_group",
            "baseline_mean",
            "candidate_mean",
            "improvement_percent",
            "target_percent",
            "target_mode",
            "meets_target",
        ],
    )
    writer.writeheader()
    for comparison in comparisons:
        for metric in comparison.metrics:
            if metric.metric not in KEY_METRICS:
                continue
            writer.writerow(
                {
                    "experiment_name": comparison.experiment_name,
                    "metric": metric.metric,
                    "metric_group": metric.metric_group,
                    "baseline_mean": metric.baseline_mean,
                    "candidate_mean": metric.candidate_mean,
                    "improvement_percent": metric.improvement_percent,
                    "target_percent": metric.target_percent,
                    "target_mode": metric.target_mode,
                    "meets_target": metric.meets_target,
                }
            )
    return buffer.getvalue()


def render_comparison_index_markdown(comparisons: list[SystemComparison]) -> str:
    lines = [
        "# Benchmark Comparison Index",
        "",
        "| Experiment | Group | Metric | Baseline | Candidate | Improvement | Target | Status |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | :---: |",
    ]
    for comparison in comparisons:
        for metric in comparison.metrics:
            if metric.metric not in KEY_METRICS:
                continue
            lines.append(
                "| "
                f"{comparison.experiment_name} | "
                f"{metric.metric_group} | "
                f"{metric.metric} | "
                f"{metric.baseline_mean:.6f} | "
                f"{metric.candidate_mean:.6f} | "
                f"{_format_percent(metric.improvement_percent)} | "
                f"{_format_target(metric.target_percent, metric.target_mode)} | "
                f"{_format_status(metric.meets_target)} |"
            )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-glob", default="benchmarks/configs/exp*.yaml")
    parser.add_argument("--output-dir", default="benchmarks/results")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        metavar="METRIC=PERCENT",
        help="Override target for a metric, e.g. --target latency_p95_ms=35.",
    )
    args = parser.parse_args(argv)

    run_all_comparisons(
        config_glob=args.config_glob,
        output_dir=args.output_dir,
        repetitions=args.repetitions,
        targets=_parse_targets(args.target),
    )


def _format_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}%"


def _format_target(value: float | None, mode: str | None) -> str:
    if value is None:
        return "n/a"
    if mode == "absolute_percent":
        return f">= {value:.2f}%"
    return f"+{value:.2f}%"


def _format_status(value: bool | None) -> str:
    if value is None:
        return "n/a"
    return "pass" if value else "fail"


if __name__ == "__main__":
    main()
