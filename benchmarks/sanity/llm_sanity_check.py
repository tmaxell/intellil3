"""Optional LLM sanity-check for workload patterns.

Verifies that the three canonical workload patterns (shared_prefix, no_reuse,
workflow_like) exhibit the structural properties expected for KV-cache analysis.

Supports two modes:
  dry_run  — analyses prompt structure only; no LLM is called.
  ollama   — sends prompts to a local Ollama instance and measures latency.

Usage (CLI):
  python -m benchmarks.sanity.llm_sanity_check --mode dry_run
  python -m benchmarks.sanity.llm_sanity_check --mode ollama --model llama3.2:1b
"""
from __future__ import annotations

import json
import pathlib
import textwrap
import time
from dataclasses import dataclass, field
from typing import Literal

from benchmarks.sanity.prompt_patterns import (
    no_reuse_prompts,
    shared_prefix_prompts,
    workflow_like_prompts,
)

Pattern = Literal["shared_prefix", "no_reuse", "workflow_like"]
Mode = Literal["dry_run", "ollama"]

_PATTERN_GENERATORS = {
    "shared_prefix": shared_prefix_prompts,
    "no_reuse": no_reuse_prompts,
    "workflow_like": workflow_like_prompts,
}

_DRY_RUN_RESPONSE = "[DRY_RUN — no LLM call made]"
_ALL_PATTERNS: list[Pattern] = ["shared_prefix", "no_reuse", "workflow_like"]


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------


@dataclass
class SanityCheckConfig:
    mode: Mode = "dry_run"
    patterns: list[Pattern] = field(default_factory=lambda: list(_ALL_PATTERNS))
    num_prompts: int = 5
    seed: int = 42
    ollama_model: str = "llama3.2:1b"
    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout_s: float = 30.0
    output_dir: str | pathlib.Path = "benchmarks/results/sanity"


@dataclass
class PatternResult:
    pattern: Pattern
    mode: Mode
    prompts: list[str]
    responses: list[str]
    latencies_ms: list[float]
    prefix_overlap_ratio: float
    median_prompt_chars: float
    cache_reuse_potential: Literal["high", "medium", "low"]
    notes: list[str]
    error: str | None = None

    def as_dict(self) -> dict:
        return {
            "pattern": self.pattern,
            "mode": self.mode,
            "num_prompts": len(self.prompts),
            "prefix_overlap_ratio": round(self.prefix_overlap_ratio, 4),
            "median_prompt_chars": self.median_prompt_chars,
            "cache_reuse_potential": self.cache_reuse_potential,
            "latencies_ms": self.latencies_ms,
            "notes": self.notes,
            "error": self.error,
        }


@dataclass
class SanityCheckReport:
    mode: Mode
    patterns: list[PatternResult]
    timestamp: str

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "timestamp": self.timestamp,
            "patterns": [p.as_dict() for p in self.patterns],
        }


# ---------------------------------------------------------------------------
# Main checker
# ---------------------------------------------------------------------------


class LLMSanityChecker:
    def __init__(self, config: SanityCheckConfig | None = None) -> None:
        self.config = config or SanityCheckConfig()

    def run(self) -> SanityCheckReport:
        results = [self._run_pattern(p) for p in self.config.patterns]
        report = SanityCheckReport(
            mode=self.config.mode,
            patterns=results,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        self._save(report)
        return report

    # ------------------------------------------------------------------

    def _run_pattern(self, pattern: Pattern) -> PatternResult:
        generator = _PATTERN_GENERATORS[pattern]
        prompts = generator(self.config.num_prompts, self.config.seed)

        overlap = _prefix_overlap_ratio(prompts)
        median_chars = float(sorted(len(p) for p in prompts)[len(prompts) // 2])

        if self.config.mode == "dry_run":
            responses = [_DRY_RUN_RESPONSE] * len(prompts)
            latencies: list[float] = []
            error = None
        else:
            responses, latencies, error = self._call_ollama_batch(prompts)

        potential = _cache_reuse_potential(overlap)
        notes = _build_notes(pattern, overlap, median_chars, self.config.mode, error)

        return PatternResult(
            pattern=pattern,
            mode=self.config.mode,
            prompts=prompts,
            responses=responses,
            latencies_ms=latencies,
            prefix_overlap_ratio=overlap,
            median_prompt_chars=median_chars,
            cache_reuse_potential=potential,
            notes=notes,
            error=error,
        )

    def _call_ollama_batch(
        self, prompts: list[str]
    ) -> tuple[list[str], list[float], str | None]:
        from benchmarks.sanity.ollama_client import call_ollama, check_ollama_available

        if not check_ollama_available(self.config.ollama_base_url):
            err = f"Ollama not reachable at {self.config.ollama_base_url}"
            return [err] * len(prompts), [], err

        responses: list[str] = []
        latencies: list[float] = []
        error: str | None = None
        for prompt in prompts:
            resp = call_ollama(
                prompt,
                model=self.config.ollama_model,
                base_url=self.config.ollama_base_url,
                timeout_s=self.config.ollama_timeout_s,
            )
            responses.append(resp.text if resp.ok else f"ERROR: {resp.error}")
            latencies.append(resp.latency_ms)
            if not resp.ok and error is None:
                error = resp.error
        return responses, latencies, error

    def _save(self, report: SanityCheckReport) -> None:
        out = pathlib.Path(self.config.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        ts = report.timestamp.replace(":", "-")
        (out / f"sanity_{report.mode}_{ts}.json").write_text(
            json.dumps(report.as_dict(), indent=2), encoding="utf-8"
        )
        (out / f"sanity_{report.mode}_{ts}.md").write_text(
            _render_markdown(report), encoding="utf-8"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _longest_common_prefix(strings: list[str]) -> str:
    if not strings:
        return ""
    prefix = strings[0]
    for s in strings[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix


def _prefix_overlap_ratio(prompts: list[str]) -> float:
    """Fraction of the average prompt length covered by the longest common prefix."""
    if not prompts:
        return 0.0
    if len(prompts) == 1:
        return 1.0
    lcp = _longest_common_prefix(prompts)
    avg_len = sum(len(p) for p in prompts) / len(prompts)
    if avg_len == 0:
        return 0.0
    return len(lcp) / avg_len


def _cache_reuse_potential(
    overlap: float,
) -> Literal["high", "medium", "low"]:
    if overlap >= 0.5:
        return "high"
    if overlap >= 0.15:
        return "medium"
    return "low"


def _build_notes(
    pattern: Pattern,
    overlap: float,
    median_chars: float,
    mode: Mode,
    error: str | None,
) -> list[str]:
    notes: list[str] = []
    if pattern == "shared_prefix":
        notes.append(
            f"Shared-prefix prompts have {overlap:.0%} prefix overlap — "
            "ideal for KV-cache reuse across parallel requests."
        )
    elif pattern == "no_reuse":
        notes.append(
            f"Independent prompts have {overlap:.0%} prefix overlap — "
            "no KV-cache benefit expected; measures cold-start latency."
        )
    else:
        notes.append(
            f"Workflow prompts accumulate context step-by-step ({overlap:.0%} overlap) — "
            "benefits from prefix caching that grows over the session."
        )
    notes.append(f"Median prompt length: {median_chars:.0f} chars.")
    if mode == "dry_run":
        notes.append("Mode: dry_run — no LLM was called; latencies not measured.")
    if error:
        notes.append(f"Error encountered: {error}")
    return notes


def _render_markdown(report: SanityCheckReport) -> str:
    lines: list[str] = [
        f"# LLM Sanity Check — `{report.mode}`",
        "",
        f"Generated: {report.timestamp}",
        "",
        "## Summary",
        "",
        "| Pattern | Cache reuse potential | Prefix overlap | Median prompt (chars) |",
        "|---|---|---|---|",
    ]
    for pr in report.patterns:
        lines.append(
            f"| `{pr.pattern}` | **{pr.cache_reuse_potential}** "
            f"| {pr.prefix_overlap_ratio:.1%} | {pr.median_prompt_chars:.0f} |"
        )
    lines.append("")

    for pr in report.patterns:
        lines += [
            f"## Pattern: `{pr.pattern}`",
            "",
            f"| Property | Value |",
            f"|---|---|",
            f"| Cache reuse potential | **{pr.cache_reuse_potential}** |",
            f"| Prefix overlap ratio | {pr.prefix_overlap_ratio:.2%} |",
            f"| Median prompt length (chars) | {pr.median_prompt_chars:.0f} |",
            f"| Prompts generated | {len(pr.prompts)} |",
        ]
        if pr.latencies_ms:
            sorted_lat = sorted(pr.latencies_ms)
            p50 = sorted_lat[len(sorted_lat) // 2]
            p95 = sorted_lat[min(int(len(sorted_lat) * 0.95), len(sorted_lat) - 1)]
            lines += [
                f"| Latency P50 (ms) | {p50:.1f} |",
                f"| Latency P95 (ms) | {p95:.1f} |",
            ]
        lines.append("")
        for note in pr.notes:
            lines.append(f"- {note}")
        lines.append("")
        lines.append("### Prompt preview (first prompt, truncated)")
        lines.append("")
        preview = textwrap.shorten(pr.prompts[0], width=400, placeholder=" [...]")
        lines.append(f"```\n{preview}\n```")
        lines.append("")
        if pr.error:
            lines.append(f"> **Warning:** {pr.error}")
            lines.append("")

    return "\n".join(lines)

