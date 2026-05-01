from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.measurement.comparison import compare_systems, render_comparison_report
from benchmarks.measurement.run_measurement import MeasurementSuite
from benchmarks.measurement.synthetic_systems import NoSystemBaseline, SyntheticL3System


def run_comparison(
    config_path: str | Path,
    output_dir: str | Path,
    repetitions: int = 5,
    baseline_system: str = NoSystemBaseline.name,
    candidate_system: str = SyntheticL3System.name,
):
    output_path = Path(output_dir)
    suite = MeasurementSuite(
        config_path=config_path,
        output_dir=output_path,
        repetitions=repetitions,
        system_factory=lambda: [NoSystemBaseline(), SyntheticL3System()],
    )
    summary = suite.run()
    comparison = compare_systems(summary, baseline_system, candidate_system)

    (output_path / "comparison.json").write_text(
        json.dumps(comparison.as_dict(), indent=2),
        encoding="utf-8",
    )
    (output_path / "comparison_report.md").write_text(
        render_comparison_report(comparison),
        encoding="utf-8",
    )
    return comparison


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--baseline-system", default=NoSystemBaseline.name)
    parser.add_argument("--candidate-system", default=SyntheticL3System.name)
    args = parser.parse_args(argv)

    run_comparison(
        config_path=args.config,
        output_dir=args.output_dir,
        repetitions=args.repetitions,
        baseline_system=args.baseline_system,
        candidate_system=args.candidate_system,
    )


if __name__ == "__main__":
    main()
