from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.measurement.comparison import (
    compare_systems,
    render_comparison_csv,
    render_comparison_report,
)
from benchmarks.measurement.run_measurement import MeasurementSuite
from benchmarks.measurement.run_summary import build_from_comparison, write_run_summary


def run_comparison(
    config_path: str | Path,
    output_dir: str | Path,
    repetitions: int = 5,
    baseline_system: str | None = None,
    candidate_system: str | None = None,
    targets: dict[str, float] | None = None,
):
    output_path = Path(output_dir)
    suite = MeasurementSuite(
        config_path=config_path,
        output_dir=output_path,
        repetitions=repetitions,
    )
    summary = suite.run()
    baseline_system, candidate_system = _resolve_comparison_systems(
        summary.systems,
        baseline_system,
        candidate_system,
    )
    comparison = compare_systems(
        summary,
        baseline_system,
        candidate_system,
        targets=targets,
    )

    (output_path / "comparison.json").write_text(
        json.dumps(comparison.as_dict(), indent=2),
        encoding="utf-8",
    )
    (output_path / "comparison_report.md").write_text(
        render_comparison_report(comparison),
        encoding="utf-8",
    )
    (output_path / "comparison.csv").write_text(
        render_comparison_csv(comparison),
        encoding="utf-8",
    )
    write_run_summary(build_from_comparison(comparison), output_path)
    return comparison


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--baseline-system")
    parser.add_argument("--candidate-system")
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        metavar="METRIC=PERCENT",
        help=(
            "Override target improvement for a metric, for example "
            "--target latency_p95_ms=35 --target throughput_req_s=60."
        ),
    )
    args = parser.parse_args(argv)

    run_comparison(
        config_path=args.config,
        output_dir=args.output_dir,
        repetitions=args.repetitions,
        baseline_system=args.baseline_system,
        candidate_system=args.candidate_system,
        targets=_parse_targets(args.target),
    )


def _parse_targets(values: list[str]) -> dict[str, float]:
    targets = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid target override: {value!r}")
        metric, percent = value.split("=", 1)
        metric = metric.strip()
        if not metric:
            raise ValueError(f"Invalid target metric in override: {value!r}")
        targets[metric] = float(percent)
    return targets


def _resolve_comparison_systems(
    systems: dict[str, object],
    baseline_system: str | None,
    candidate_system: str | None,
) -> tuple[str, str]:
    system_names = list(systems)
    if baseline_system is None:
        if not system_names:
            raise ValueError("Cannot infer baseline system: no systems were measured")
        baseline_system = system_names[0]
    if candidate_system is None:
        if len(system_names) < 2:
            raise ValueError("Cannot infer candidate system: fewer than two systems")
        candidate_system = next(
            (name for name in reversed(system_names) if name != baseline_system),
            system_names[-1],
        )
    return baseline_system, candidate_system


if __name__ == "__main__":
    main()
