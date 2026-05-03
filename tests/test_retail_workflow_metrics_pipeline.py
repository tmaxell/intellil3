from __future__ import annotations

import json

from benchmarks.measurement.run_comparison import run_comparison
from benchmarks.measurement.run_measurement import MeasurementSuite


def test_retail_workflow_metrics_are_aggregated_in_measurement(tmp_path) -> None:
    config_path = tmp_path / "retail_workflow.yaml"
    config_path.write_text(
        """
experiment:
  name: "Retail workflow metrics smoke"
workload:
  type: "retail_support_workflow"
  num_workflows: 4
  turns_per_workflow: 4
  seed: 21
systems:
  - name: "baseline_lru"
    type: "lru_l3"
  - name: "agentic_active_l3"
    type: "agentic_active_l3"
""",
        encoding="utf-8",
    )
    output_dir = tmp_path / "results"
    summary = MeasurementSuite(
        config_path=config_path,
        output_dir=output_dir,
        repetitions=1,
    ).run()

    baseline = summary.systems["baseline_lru"]
    candidate = summary.systems["agentic_active_l3"]
    for metrics in (baseline, candidate):
        assert "workflow_correctness_rate" in metrics
        assert "policy_violation_rate" in metrics
        assert "tool_use_accuracy" in metrics
        assert "fallback_success_rate" in metrics
        assert "milestone_pass_rate" in metrics
        assert "tool_arg_error_count" in metrics

    summary_json = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert "workflow_correctness_rate" in summary_json["systems"]["baseline_lru"]


def test_retail_workflow_metrics_flow_into_comparison(tmp_path) -> None:
    config_path = tmp_path / "retail_workflow_compare.yaml"
    config_path.write_text(
        """
experiment:
  name: "Retail workflow comparison smoke"
workload:
  type: "retail_support_workflow"
  num_workflows: 3
  turns_per_workflow: 4
  seed: 22
systems:
  - name: "baseline_lru"
    type: "lru_l3"
  - name: "agentic_active_l3"
    type: "agentic_active_l3"
""",
        encoding="utf-8",
    )
    output_dir = tmp_path / "comparison_results"
    run_comparison(
        config_path=config_path,
        output_dir=output_dir,
        repetitions=1,
        baseline_system="baseline_lru",
        candidate_system="agentic_active_l3",
    )

    comparison_json = json.loads(
        (output_dir / "comparison.json").read_text(encoding="utf-8")
    )
    metric_names = {row["metric"] for row in comparison_json["metrics"]}
    assert "workflow_correctness_rate" in metric_names
    assert "policy_violation_rate" in metric_names
    assert "milestone_pass_rate" in metric_names
