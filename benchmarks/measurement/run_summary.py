"""Per-run summary layer for the IntelliL3 benchmark pipeline.

After every ``run_comparison``, ``MeasurementSuite.run``, or LLM sanity-check
this module writes two compact artifacts to the same output directory:

  run_summary.json  — machine-readable summary with key metrics and verdict
  run_summary.md    — human-readable summary suitable for thesis transfer

The summary reads from the already-computed in-memory objects (SystemComparison,
MeasurementSummary, SanityCheckReport) — it does not re-run any benchmarks.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmarks.measurement.metric_catalog import KEY_METRICS, direction_for, metric_group_for

if TYPE_CHECKING:
    from benchmarks.measurement.comparison import SystemComparison
    from benchmarks.measurement.run_measurement import MeasurementSummary
    from benchmarks.sanity.llm_sanity_check import SanityCheckReport

# Minimum absolute delta to call a change "improved" or "regressed"
_EPSILON = 1e-9


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------


@dataclass
class MetricEntry:
    """Summary row for one metric."""

    metric: str
    group: str          # core | agentic | rag
    direction: str      # higher_is_better | lower_is_better | neutral
    values: dict[str, float]   # system_name → mean value
    delta: float | None = None              # candidate − baseline
    improvement_percent: float | None = None
    verdict: str = "n/a"  # improved | regressed | unchanged | n/a

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "group": self.group,
            "direction": self.direction,
            "values": self.values,
            "delta": self.delta,
            "improvement_percent": self.improvement_percent,
            "verdict": self.verdict,
        }


@dataclass
class RunSummary:
    """Compact summary for one benchmark run."""

    experiment_name: str
    mode: str               # comparison | measurement | sanity
    systems: list[str]
    baseline_system: str | None
    candidate_system: str | None
    timestamp: str
    key_metrics: list[MetricEntry]
    interpretation: str     # 2–4 sentences for thesis copy-paste

    def as_dict(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "mode": self.mode,
            "systems": self.systems,
            "baseline_system": self.baseline_system,
            "candidate_system": self.candidate_system,
            "timestamp": self.timestamp,
            "key_metrics": [m.as_dict() for m in self.key_metrics],
            "interpretation": self.interpretation,
        }

    def to_markdown(self) -> str:
        return _render_markdown(self)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def build_from_comparison(comparison: "SystemComparison") -> RunSummary:
    """Build a RunSummary from a finished SystemComparison."""
    key_entries: list[MetricEntry] = []
    for mc in comparison.metrics:
        if mc.metric not in KEY_METRICS:
            continue
        verdict = _verdict(mc.delta, mc.direction)
        key_entries.append(
            MetricEntry(
                metric=mc.metric,
                group=mc.metric_group,
                direction=mc.direction,
                values={
                    comparison.baseline_system: mc.baseline_mean,
                    comparison.candidate_system: mc.candidate_mean,
                },
                delta=mc.delta,
                improvement_percent=mc.improvement_percent,
                verdict=verdict,
            )
        )

    interpretation = _interpret_comparison(comparison, key_entries)
    return RunSummary(
        experiment_name=comparison.experiment_name,
        mode="comparison",
        systems=[comparison.baseline_system, comparison.candidate_system],
        baseline_system=comparison.baseline_system,
        candidate_system=comparison.candidate_system,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        key_metrics=key_entries,
        interpretation=interpretation,
    )


def build_from_measurement(summary: "MeasurementSummary") -> RunSummary:
    """Build a RunSummary from a MeasurementSummary (no comparison pair)."""
    system_names = list(summary.systems)

    # Collect mean values for each key metric across all systems
    key_entries: list[MetricEntry] = []
    all_metric_names = set()
    for metrics in summary.systems.values():
        all_metric_names.update(metrics)

    for metric_name in sorted(all_metric_names):
        if metric_name not in KEY_METRICS:
            continue
        values: dict[str, float] = {}
        for sys_name, metrics in summary.systems.items():
            if metric_name in metrics:
                values[sys_name] = metrics[metric_name]["mean"]
        if not values:
            continue
        key_entries.append(
            MetricEntry(
                metric=metric_name,
                group=metric_group_for(metric_name),
                direction=direction_for(metric_name),
                values=values,
            )
        )

    interpretation = _interpret_measurement(summary, key_entries)
    return RunSummary(
        experiment_name=summary.experiment_name,
        mode="measurement",
        systems=system_names,
        baseline_system=None,
        candidate_system=None,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        key_metrics=key_entries,
        interpretation=interpretation,
    )


def build_from_sanity(report: "SanityCheckReport") -> RunSummary:
    """Build a RunSummary from an LLM sanity-check report."""
    key_entries: list[MetricEntry] = []
    for pr in report.patterns:
        key_entries.append(
            MetricEntry(
                metric=f"{pr.pattern}_prefix_overlap",
                group="sanity",
                direction="higher_is_better",
                values={pr.pattern: round(pr.prefix_overlap_ratio, 4)},
                verdict=pr.cache_reuse_potential,
            )
        )

    interpretation = _interpret_sanity(report)
    return RunSummary(
        experiment_name="LLM Sanity Check",
        mode="sanity",
        systems=[pr.pattern for pr in report.patterns],
        baseline_system=None,
        candidate_system=None,
        timestamp=report.timestamp,
        key_metrics=key_entries,
        interpretation=interpretation,
    )


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def write_run_summary(summary: RunSummary, output_dir: Path | str) -> None:
    """Persist run_summary.json and run_summary.md inside *output_dir*."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_summary.json").write_text(
        json.dumps(summary.as_dict(), indent=2), encoding="utf-8"
    )
    (out / "run_summary.md").write_text(summary.to_markdown(), encoding="utf-8")


# ---------------------------------------------------------------------------
# Interpretation text generators
# ---------------------------------------------------------------------------


def _interpret_comparison(
    comparison: "SystemComparison",
    entries: list[MetricEntry],
) -> str:
    improved = [e for e in entries if e.verdict == "improved"]
    regressed = [e for e in entries if e.verdict == "regressed"]
    total = len(entries)

    if total == 0:
        return (
            f"Experiment '{comparison.experiment_name}': "
            f"no key metrics were measured for comparison."
        )

    parts: list[str] = [
        f"Experiment '{comparison.experiment_name}': "
        f"candidate '{comparison.candidate_system}' vs "
        f"baseline '{comparison.baseline_system}' — "
        f"{len(improved)} of {total} key metric(s) improved."
    ]

    # Best improvement
    if improved:
        best = max(
            improved,
            key=lambda e: abs(e.improvement_percent or 0),
        )
        pct = f"{best.improvement_percent:+.1f}%" if best.improvement_percent is not None else ""
        parts.append(
            f"Largest gain: {best.metric} {pct} "
            f"({best.values.get(comparison.baseline_system, 0):.4f} → "
            f"{best.values.get(comparison.candidate_system, 0):.4f})."
        )

    # Notable regression
    if regressed:
        worst = max(regressed, key=lambda e: abs(e.improvement_percent or 0))
        pct = f"{worst.improvement_percent:+.1f}%" if worst.improvement_percent is not None else ""
        parts.append(f"Notable regression: {worst.metric} {pct}.")

    # Target status
    meets = [
        mc for mc in comparison.metrics
        if mc.metric in KEY_METRICS and mc.meets_target is True
    ]
    misses = [
        mc for mc in comparison.metrics
        if mc.metric in KEY_METRICS and mc.meets_target is False
    ]
    if meets or misses:
        parts.append(
            f"Targets met: {len(meets)}, missed: {len(misses)}."
        )

    return " ".join(parts)


def _interpret_measurement(
    summary: "MeasurementSummary",
    entries: list[MetricEntry],
) -> str:
    n_sys = len(summary.systems)
    parts: list[str] = [
        f"Experiment '{summary.experiment_name}': "
        f"{n_sys} system(s) measured over {summary.repetitions} repetition(s)."
    ]

    if not entries:
        return " ".join(parts)

    # Find standout metric per group
    group_highlights: dict[str, str] = {}
    for entry in entries:
        if entry.group in group_highlights:
            continue
        best_sys = _best_system_for(entry)
        if best_sys:
            val = entry.values[best_sys]
            group_highlights[entry.group] = (
                f"{entry.metric}={val:.4f} ({best_sys})"
            )

    if group_highlights:
        highlights = "; ".join(group_highlights.values())
        parts.append(f"Key metric highlights: {highlights}.")

    return " ".join(parts)


def _interpret_sanity(report: "SanityCheckReport") -> str:
    potentials = {pr.pattern: pr.cache_reuse_potential for pr in report.patterns}
    high = [p for p, v in potentials.items() if v == "high"]
    medium = [p for p, v in potentials.items() if v == "medium"]
    low = [p for p, v in potentials.items() if v == "low"]

    parts: list[str] = [
        f"LLM sanity-check ({report.mode} mode): "
        f"{len(report.patterns)} workload pattern(s) analysed."
    ]
    if high:
        parts.append(
            f"High KV-cache reuse potential: {', '.join(high)} — "
            "these patterns are strong candidates for prefix-caching experiments."
        )
    if medium:
        parts.append(
            f"Medium cache reuse potential: {', '.join(medium)} — "
            "benefits from session-level prefix accumulation."
        )
    if low:
        parts.append(
            f"Low cache reuse potential: {', '.join(low)} — "
            "suitable as a cold-start / no-reuse baseline."
        )
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------


def _render_markdown(summary: RunSummary) -> str:
    lines: list[str] = [
        f"# Run Summary — {summary.experiment_name}",
        "",
        f"**Mode:** {summary.mode}  ",
        f"**Timestamp:** {summary.timestamp}  ",
    ]

    if summary.baseline_system:
        lines += [
            f"**Baseline:** `{summary.baseline_system}`  ",
            f"**Candidate:** `{summary.candidate_system}`  ",
        ]
    else:
        lines.append(f"**Systems:** {', '.join(f'`{s}`' for s in summary.systems)}  ")

    lines += ["", "---", "", "## Interpretation", "", f"> {summary.interpretation}", ""]

    if summary.key_metrics:
        lines += ["## Key Metrics", ""]

        if summary.mode == "comparison":
            baseline = summary.baseline_system or ""
            candidate = summary.candidate_system or ""
            lines += [
                f"| Metric | Group | {baseline} | {candidate} | Δ | Improvement | Verdict |",
                "|---|---|---:|---:|---:|---:|:---:|",
            ]
            for e in summary.key_metrics:
                b_val = e.values.get(baseline, float("nan"))
                c_val = e.values.get(candidate, float("nan"))
                delta_str = f"{e.delta:+.4f}" if e.delta is not None else "—"
                pct_str = (
                    f"{e.improvement_percent:+.1f}%"
                    if e.improvement_percent is not None
                    else "—"
                )
                verdict_icon = _verdict_icon(e.verdict)
                lines.append(
                    f"| `{e.metric}` | {e.group} | {b_val:.4f} | {c_val:.4f} "
                    f"| {delta_str} | {pct_str} | {verdict_icon} |"
                )

        elif summary.mode == "sanity":
            lines += [
                "| Pattern | Prefix overlap | Cache reuse potential |",
                "|---|---:|:---:|",
            ]
            for e in summary.key_metrics:
                pattern = list(e.values.keys())[0] if e.values else "?"
                overlap = list(e.values.values())[0] if e.values else 0.0
                lines.append(
                    f"| `{pattern}` | {overlap:.1%} | **{e.verdict}** |"
                )

        else:  # measurement
            sys_names = summary.systems
            header = "| Metric | Group | " + " | ".join(sys_names) + " |"
            separator = "|---|---|" + "---:|" * len(sys_names)
            lines += [header, separator]
            for e in summary.key_metrics:
                vals = " | ".join(
                    f"{e.values.get(s, float('nan')):.4f}" for s in sys_names
                )
                lines.append(f"| `{e.metric}` | {e.group} | {vals} |")

        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _verdict(delta: float | None, direction: str) -> str:
    if delta is None or abs(delta) < _EPSILON:
        return "unchanged"
    if direction == "higher_is_better":
        return "improved" if delta > 0 else "regressed"
    if direction == "lower_is_better":
        return "improved" if delta < 0 else "regressed"
    return "unchanged"


def _verdict_icon(verdict: str) -> str:
    return {"improved": "✓", "regressed": "✗", "unchanged": "—", "n/a": "n/a"}.get(
        verdict, verdict
    )


def _best_system_for(entry: MetricEntry) -> str | None:
    if not entry.values:
        return None
    if entry.direction == "higher_is_better":
        return max(entry.values, key=entry.values.__getitem__)
    if entry.direction == "lower_is_better":
        return min(entry.values, key=entry.values.__getitem__)
    return next(iter(entry.values))
