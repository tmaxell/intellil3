from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from benchmarks.baselines.base import BenchmarkSystem
from benchmarks.measurement.environment import collect_environment
from benchmarks.runners.benchmark_runner import BenchmarkRunner, NoOpSystem


@dataclass(frozen=True)
class MeasurementSummary:
    """Aggregated metrics over repeated benchmark runs."""

    experiment_name: str
    repetitions: int
    systems: dict[str, dict[str, dict[str, float]]]
    environment: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "repetitions": self.repetitions,
            "systems": self.systems,
            "environment": self.environment,
        }


class MeasurementSuite:
    """Run benchmark configs repeatedly and persist ВКР-friendly artifacts."""

    def __init__(
        self,
        config_path: str | Path,
        output_dir: str | Path,
        repetitions: int = 5,
        noop: bool = False,
        systems: list[BenchmarkSystem] | None = None,
        system_factory: Callable[[], list[BenchmarkSystem]] | None = None,
    ):
        if repetitions <= 0:
            raise ValueError("repetitions must be positive")
        if systems is not None and system_factory is not None:
            raise ValueError("systems and system_factory are mutually exclusive")

        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.repetitions = repetitions
        self.noop = noop
        self.systems = systems
        self.system_factory = system_factory

    def run(self) -> MeasurementSummary:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        raw_runs = []

        for repetition in range(self.repetitions):
            runner = BenchmarkRunner.from_yaml(
                self.config_path,
                systems=self._build_systems(),
            )
            results = [result.as_dict() for result in runner.run()]
            raw_runs.append(
                {
                    "repetition": repetition,
                    "results": results,
                }
            )

        environment = collect_environment(self.config_path)
        summary = self._summarize(raw_runs, environment)
        self._write_artifacts(raw_runs, summary)
        return summary

    def _summarize(
        self,
        raw_runs: list[dict[str, Any]],
        environment: dict[str, Any],
    ) -> MeasurementSummary:
        systems: dict[str, dict[str, list[float]]] = {}
        experiment_name = "benchmark"

        for run in raw_runs:
            for result in run["results"]:
                experiment_name = result["experiment_name"]
                system_name = result["system_name"]
                systems.setdefault(system_name, {})
                for metric_name, metric_value in result["metrics"].items():
                    if isinstance(metric_value, int | float):
                        systems[system_name].setdefault(metric_name, []).append(
                            float(metric_value)
                        )

        aggregated = {
            system_name: {
                metric_name: _aggregate(values)
                for metric_name, values in metrics.items()
            }
            for system_name, metrics in systems.items()
        }
        return MeasurementSummary(
            experiment_name=experiment_name,
            repetitions=self.repetitions,
            systems=aggregated,
            environment=environment,
        )

    def _write_artifacts(
        self,
        raw_runs: list[dict[str, Any]],
        summary: MeasurementSummary,
    ) -> None:
        raw_path = self.output_dir / "raw_runs.json"
        summary_path = self.output_dir / "summary.json"
        report_path = self.output_dir / "measurement_report.md"

        raw_path.write_text(json.dumps(raw_runs, indent=2), encoding="utf-8")
        summary_path.write_text(
            json.dumps(summary.as_dict(), indent=2),
            encoding="utf-8",
        )
        report_path.write_text(render_measurement_report(summary), encoding="utf-8")

    def _build_systems(self) -> list[BenchmarkSystem]:
        if self.system_factory is not None:
            return self.system_factory()
        if self.systems is not None:
            return self.systems
        return [NoOpSystem()] if self.noop else []


def render_measurement_report(summary: MeasurementSummary) -> str:
    lines = [
        f"# Measurement Report: {summary.experiment_name}",
        "",
        f"- Repetitions: {summary.repetitions}",
        f"- Git commit: `{summary.environment.get('git_commit', '')}`",
        f"- Git branch: `{summary.environment.get('git_branch', '')}`",
        f"- Config SHA256: `{summary.environment.get('config_sha256', '')}`",
        "",
    ]

    for system_name, metrics in summary.systems.items():
        lines.extend([f"## {system_name}", ""])
        for metric_name in sorted(metrics):
            values = metrics[metric_name]
            lines.append(
                "- "
                f"{metric_name}: mean={values['mean']:.6f}, "
                f"std={values['std']:.6f}, "
                f"min={values['min']:.6f}, "
                f"max={values['max']:.6f}"
            )
        lines.append("")

    return "\n".join(lines)


def _aggregate(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--noop",
        action="store_true",
        help="Run with synthetic NoOpSystem for measurement pipeline smoke tests.",
    )
    args = parser.parse_args(argv)

    suite = MeasurementSuite(
        config_path=args.config,
        output_dir=args.output_dir,
        repetitions=args.repetitions,
        noop=args.noop,
    )
    suite.run()


if __name__ == "__main__":
    main()
