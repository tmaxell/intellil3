from __future__ import annotations

import json

from benchmarks.measurement.run_comparison import run_comparison
from benchmarks.measurement.run_measurement import MeasurementSuite
from benchmarks.runners.benchmark_runner import BenchmarkRunner


def test_benchmark_runner_executes_retail_workflow_system_pair() -> None:
    config = {
        "experiment": {"name": "Retail workflow runner smoke"},
        "workload": {
            "type": "retail_support_workflow",
            "num_workflows": 4,
            "turns_per_workflow": 4,
            "seed": 52,
        },
        "systems": [
            {"name": "baseline_lru", "type": "lru_l3"},
            {"name": "agentic_active_l3", "type": "agentic_active_l3"},
        ],
    }

    results = BenchmarkRunner(config).run()

    assert len(results) == 2
    assert [result.system_name for result in results] == [
        "baseline_lru",
        "agentic_active_l3",
    ]
    assert all(result.workload_type == "retail_support_workflow" for result in results)
    assert all(result.metrics.total_requests == 16 for result in results)
    for result in results:
        exported = result.metrics.as_dict()
        assert "workflow_correctness_rate" in exported
        assert "policy_violation_rate" in exported
        assert "tool_use_accuracy" in exported
        assert "fallback_success_rate" in exported
        assert "milestone_pass_rate" in exported


def test_measurement_suite_writes_retail_workflow_artifacts(tmp_path) -> None:
    config_path = tmp_path / "retail_measurement.yaml"
    config_path.write_text(
        """
experiment:
  name: "Retail workflow measurement smoke"
workload:
  type: "retail_support_workflow"
  num_workflows: 5
  turns_per_workflow: 4
  seed: 53
systems:
  - name: "baseline_lru"
    type: "lru_l3"
  - name: "agentic_active_l3"
    type: "agentic_active_l3"
""",
        encoding="utf-8",
    )
    output_dir = tmp_path / "measurement_results"
    summary = MeasurementSuite(
        config_path=config_path,
        output_dir=output_dir,
        repetitions=1,
    ).run()

    assert (output_dir / "raw_runs.json").exists()
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "measurement_report.md").exists()
    assert set(summary.systems.keys()) == {"baseline_lru", "agentic_active_l3"}

    summary_json = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert "workflow_correctness_rate" in summary_json["systems"]["baseline_lru"]
    assert "policy_violation_rate" in summary_json["systems"]["agentic_active_l3"]


def test_run_comparison_emits_retail_workflow_comparison_artifacts(tmp_path) -> None:
    config_path = tmp_path / "retail_comparison.yaml"
    config_path.write_text(
        """
experiment:
  name: "Retail workflow comparison smoke"
workload:
  type: "retail_support_workflow"
  num_workflows: 4
  turns_per_workflow: 4
  seed: 54
systems:
  - name: "baseline_lru"
    type: "lru_l3"
  - name: "agentic_active_l3"
    type: "agentic_active_l3"
""",
        encoding="utf-8",
    )
    output_dir = tmp_path / "comparison_results"
    comparison = run_comparison(
        config_path=config_path,
        output_dir=output_dir,
        repetitions=1,
        baseline_system="baseline_lru",
        candidate_system="agentic_active_l3",
    )

    assert comparison.experiment_name == "Retail workflow comparison smoke"
    assert (output_dir / "comparison.json").exists()
    assert (output_dir / "comparison.csv").exists()
    assert (output_dir / "comparison_report.md").exists()

    comparison_json = json.loads((output_dir / "comparison.json").read_text(encoding="utf-8"))
    metric_names = {item["metric"] for item in comparison_json["metrics"]}
    assert "workflow_correctness_rate" in metric_names
    assert "policy_violation_rate" in metric_names
    assert "milestone_pass_rate" in metric_names
