from __future__ import annotations

import json
import pathlib

import pytest

from benchmarks.sanity.llm_sanity_check import (
    LLMSanityChecker,
    PatternResult,
    SanityCheckConfig,
    SanityCheckReport,
    _cache_reuse_potential,
    _longest_common_prefix,
    _prefix_overlap_ratio,
)
from benchmarks.sanity.prompt_patterns import (
    no_reuse_prompts,
    shared_prefix_prompts,
    workflow_like_prompts,
)


# ---------------------------------------------------------------------------
# prompt_patterns unit tests
# ---------------------------------------------------------------------------


def test_shared_prefix_prompts_count() -> None:
    prompts = shared_prefix_prompts(7)
    assert len(prompts) == 7


def test_no_reuse_prompts_count() -> None:
    prompts = no_reuse_prompts(6)
    assert len(prompts) == 6


def test_workflow_like_prompts_count() -> None:
    prompts = workflow_like_prompts(8)
    assert len(prompts) == 8


def test_shared_prefix_prompts_are_not_identical() -> None:
    prompts = shared_prefix_prompts(5)
    assert len(set(prompts)) > 1, "All shared-prefix prompts are identical"


def test_no_reuse_prompts_are_all_different() -> None:
    prompts = no_reuse_prompts(8)
    assert len(set(prompts)) == 8


def test_workflow_prompts_accumulate_context() -> None:
    prompts = workflow_like_prompts(5)
    for i in range(1, len(prompts)):
        assert len(prompts[i]) > len(prompts[i - 1]), (
            f"Step {i + 1} should be longer than step {i}"
        )


def test_prompts_are_reproducible_with_same_seed() -> None:
    a = shared_prefix_prompts(5, seed=7)
    b = shared_prefix_prompts(5, seed=7)
    assert a == b


def test_prompts_differ_with_different_seeds() -> None:
    a = shared_prefix_prompts(5, seed=1)
    b = shared_prefix_prompts(5, seed=99)
    assert a != b


def test_no_reuse_prompts_reproducible() -> None:
    a = no_reuse_prompts(5, seed=3)
    b = no_reuse_prompts(5, seed=3)
    assert a == b


# ---------------------------------------------------------------------------
# Prefix-overlap helpers
# ---------------------------------------------------------------------------


def test_longest_common_prefix_identical() -> None:
    assert _longest_common_prefix(["abcdef", "abcdef"]) == "abcdef"


def test_longest_common_prefix_partial() -> None:
    assert _longest_common_prefix(["abcXYZ", "abcDEF"]) == "abc"


def test_longest_common_prefix_no_overlap() -> None:
    assert _longest_common_prefix(["foo", "bar"]) == ""


def test_longest_common_prefix_single() -> None:
    assert _longest_common_prefix(["hello"]) == "hello"


def test_longest_common_prefix_empty_list() -> None:
    assert _longest_common_prefix([]) == ""


def test_prefix_overlap_ratio_identical() -> None:
    ratio = _prefix_overlap_ratio(["abc", "abc", "abc"])
    assert ratio == pytest.approx(1.0)


def test_prefix_overlap_ratio_no_overlap() -> None:
    ratio = _prefix_overlap_ratio(["foo", "bar", "baz"])
    assert ratio == pytest.approx(0.0)


def test_prefix_overlap_ratio_partial() -> None:
    # "XXXX" is 4 chars, avg = (6 + 6) / 2 = 6, ratio = 4/6 ≈ 0.667
    ratio = _prefix_overlap_ratio(["XXXXAB", "XXXXCD"])
    assert 0.5 < ratio < 1.0


def test_prefix_overlap_ratio_single_prompt() -> None:
    assert _prefix_overlap_ratio(["anything"]) == pytest.approx(1.0)


def test_prefix_overlap_ratio_empty_list() -> None:
    assert _prefix_overlap_ratio([]) == pytest.approx(0.0)


def test_cache_reuse_potential_high() -> None:
    assert _cache_reuse_potential(0.8) == "high"


def test_cache_reuse_potential_medium() -> None:
    assert _cache_reuse_potential(0.3) == "medium"


def test_cache_reuse_potential_low() -> None:
    assert _cache_reuse_potential(0.05) == "low"


def test_cache_reuse_potential_boundary_high() -> None:
    assert _cache_reuse_potential(0.5) == "high"


def test_cache_reuse_potential_boundary_medium() -> None:
    assert _cache_reuse_potential(0.15) == "medium"


# ---------------------------------------------------------------------------
# LLMSanityChecker — dry_run mode
# ---------------------------------------------------------------------------


def test_dry_run_produces_results_for_all_patterns(tmp_path) -> None:
    config = SanityCheckConfig(
        mode="dry_run",
        output_dir=tmp_path / "sanity",
        num_prompts=4,
    )
    report = LLMSanityChecker(config).run()
    assert len(report.patterns) == 3
    patterns_seen = {pr.pattern for pr in report.patterns}
    assert patterns_seen == {"shared_prefix", "no_reuse", "workflow_like"}


def test_dry_run_returns_sanity_check_report_type(tmp_path) -> None:
    config = SanityCheckConfig(mode="dry_run", output_dir=tmp_path, num_prompts=3)
    report = LLMSanityChecker(config).run()
    assert isinstance(report, SanityCheckReport)


def test_dry_run_has_no_latencies(tmp_path) -> None:
    config = SanityCheckConfig(mode="dry_run", output_dir=tmp_path, num_prompts=3)
    report = LLMSanityChecker(config).run()
    for pr in report.patterns:
        assert pr.latencies_ms == [], f"{pr.pattern} should have no latencies in dry_run"


def test_dry_run_responses_are_placeholders(tmp_path) -> None:
    config = SanityCheckConfig(mode="dry_run", output_dir=tmp_path, num_prompts=3)
    report = LLMSanityChecker(config).run()
    for pr in report.patterns:
        assert all("DRY_RUN" in r for r in pr.responses)


def test_dry_run_has_no_error(tmp_path) -> None:
    config = SanityCheckConfig(mode="dry_run", output_dir=tmp_path, num_prompts=3)
    report = LLMSanityChecker(config).run()
    for pr in report.patterns:
        assert pr.error is None


def test_dry_run_num_prompts_respected(tmp_path) -> None:
    config = SanityCheckConfig(mode="dry_run", output_dir=tmp_path, num_prompts=7)
    report = LLMSanityChecker(config).run()
    for pr in report.patterns:
        assert len(pr.prompts) == 7


# ---------------------------------------------------------------------------
# Cache-reuse potential ordering
# ---------------------------------------------------------------------------


def test_shared_prefix_has_high_cache_reuse_potential(tmp_path) -> None:
    config = SanityCheckConfig(mode="dry_run", output_dir=tmp_path, num_prompts=5)
    report = LLMSanityChecker(config).run()
    sp = next(pr for pr in report.patterns if pr.pattern == "shared_prefix")
    assert sp.cache_reuse_potential == "high"


def test_no_reuse_has_low_cache_reuse_potential(tmp_path) -> None:
    config = SanityCheckConfig(mode="dry_run", output_dir=tmp_path, num_prompts=5)
    report = LLMSanityChecker(config).run()
    nr = next(pr for pr in report.patterns if pr.pattern == "no_reuse")
    assert nr.cache_reuse_potential == "low"


def test_shared_prefix_has_higher_overlap_than_no_reuse(tmp_path) -> None:
    config = SanityCheckConfig(mode="dry_run", output_dir=tmp_path, num_prompts=6)
    report = LLMSanityChecker(config).run()
    sp = next(pr for pr in report.patterns if pr.pattern == "shared_prefix")
    nr = next(pr for pr in report.patterns if pr.pattern == "no_reuse")
    assert sp.prefix_overlap_ratio > nr.prefix_overlap_ratio


def test_workflow_overlap_is_nonzero(tmp_path) -> None:
    config = SanityCheckConfig(mode="dry_run", output_dir=tmp_path, num_prompts=5)
    report = LLMSanityChecker(config).run()
    wl = next(pr for pr in report.patterns if pr.pattern == "workflow_like")
    assert wl.prefix_overlap_ratio > 0.0


# ---------------------------------------------------------------------------
# Artifact files
# ---------------------------------------------------------------------------


def test_artifacts_are_written_to_output_dir(tmp_path) -> None:
    out = tmp_path / "sanity_out"
    config = SanityCheckConfig(mode="dry_run", output_dir=out, num_prompts=3)
    LLMSanityChecker(config).run()
    json_files = list(out.glob("sanity_dry_run_*.json"))
    md_files = list(out.glob("sanity_dry_run_*.md"))
    assert len(json_files) == 1, f"Expected 1 JSON file, found {json_files}"
    assert len(md_files) == 1, f"Expected 1 Markdown file, found {md_files}"


def test_output_dir_created_if_missing(tmp_path) -> None:
    out = tmp_path / "deep" / "nested" / "sanity"
    assert not out.exists()
    config = SanityCheckConfig(mode="dry_run", output_dir=out, num_prompts=3)
    LLMSanityChecker(config).run()
    assert out.exists()


def test_report_json_has_expected_structure(tmp_path) -> None:
    out = tmp_path / "sanity"
    config = SanityCheckConfig(mode="dry_run", output_dir=out, num_prompts=3)
    LLMSanityChecker(config).run()
    json_file = next(out.glob("sanity_dry_run_*.json"))
    data = json.loads(json_file.read_text(encoding="utf-8"))
    assert data["mode"] == "dry_run"
    assert "timestamp" in data
    assert "patterns" in data
    assert len(data["patterns"]) == 3
    for entry in data["patterns"]:
        assert "pattern" in entry
        assert "prefix_overlap_ratio" in entry
        assert "cache_reuse_potential" in entry
        assert "num_prompts" in entry


def test_markdown_report_mentions_all_patterns(tmp_path) -> None:
    out = tmp_path / "sanity"
    config = SanityCheckConfig(mode="dry_run", output_dir=out, num_prompts=3)
    LLMSanityChecker(config).run()
    md_file = next(out.glob("sanity_dry_run_*.md"))
    text = md_file.read_text(encoding="utf-8")
    for pattern in ("shared_prefix", "no_reuse", "workflow_like"):
        assert pattern in text, f"{pattern} missing from markdown report"


def test_markdown_report_has_summary_table(tmp_path) -> None:
    out = tmp_path / "sanity"
    config = SanityCheckConfig(mode="dry_run", output_dir=out, num_prompts=3)
    LLMSanityChecker(config).run()
    md_file = next(out.glob("sanity_dry_run_*.md"))
    text = md_file.read_text(encoding="utf-8")
    assert "## Summary" in text
    assert "Cache reuse potential" in text


# ---------------------------------------------------------------------------
# Single-pattern run
# ---------------------------------------------------------------------------


def test_single_pattern_run(tmp_path) -> None:
    config = SanityCheckConfig(
        mode="dry_run",
        patterns=["shared_prefix"],
        output_dir=tmp_path,
        num_prompts=3,
    )
    report = LLMSanityChecker(config).run()
    assert len(report.patterns) == 1
    assert report.patterns[0].pattern == "shared_prefix"


# ---------------------------------------------------------------------------
# Ollama mode with unavailable server
# ---------------------------------------------------------------------------


def test_ollama_mode_gracefully_handles_unavailable_server(tmp_path) -> None:
    config = SanityCheckConfig(
        mode="ollama",
        patterns=["shared_prefix"],
        ollama_base_url="http://127.0.0.1:19999",  # guaranteed-unreachable port
        output_dir=tmp_path,
        num_prompts=3,
        ollama_timeout_s=2.0,
    )
    report = LLMSanityChecker(config).run()
    pr = report.patterns[0]
    assert pr.error is not None, "Expected an error when Ollama is unavailable"
    assert pr.latencies_ms == []
    assert all("Ollama not reachable" in r or "ERROR" in r for r in pr.responses)


def test_ollama_unavailable_still_writes_artifacts(tmp_path) -> None:
    config = SanityCheckConfig(
        mode="ollama",
        patterns=["no_reuse"],
        ollama_base_url="http://127.0.0.1:19999",
        output_dir=tmp_path / "ollama_out",
        num_prompts=2,
        ollama_timeout_s=2.0,
    )
    LLMSanityChecker(config).run()
    json_files = list((tmp_path / "ollama_out").glob("sanity_ollama_*.json"))
    assert len(json_files) == 1


# ---------------------------------------------------------------------------
# as_dict contract
# ---------------------------------------------------------------------------


def test_pattern_result_as_dict_has_required_keys(tmp_path) -> None:
    config = SanityCheckConfig(mode="dry_run", output_dir=tmp_path, num_prompts=3)
    report = LLMSanityChecker(config).run()
    required = {
        "pattern", "mode", "num_prompts", "prefix_overlap_ratio",
        "median_prompt_chars", "cache_reuse_potential", "latencies_ms",
        "notes", "error",
    }
    for pr in report.patterns:
        d = pr.as_dict()
        assert required <= set(d.keys()), f"Missing keys in {pr.pattern}: {required - set(d.keys())}"


def test_report_as_dict_roundtrip(tmp_path) -> None:
    config = SanityCheckConfig(mode="dry_run", output_dir=tmp_path, num_prompts=3)
    report = LLMSanityChecker(config).run()
    d = report.as_dict()
    assert d["mode"] == "dry_run"
    assert len(d["patterns"]) == 3


# ---------------------------------------------------------------------------
# Isolation from main benchmark pipeline
# ---------------------------------------------------------------------------


def test_import_sanity_does_not_affect_benchmark_runner() -> None:
    from benchmarks.runners.benchmark_runner import BenchmarkRunner  # noqa: F401
    from benchmarks.sanity import LLMSanityChecker as _  # noqa: F401


def test_sanity_check_has_separate_results_dir(tmp_path) -> None:
    sanity_out = tmp_path / "sanity"
    benchmark_out = tmp_path / "benchmark"
    config = SanityCheckConfig(mode="dry_run", output_dir=sanity_out, num_prompts=2)
    LLMSanityChecker(config).run()
    assert sanity_out.exists()
    assert not benchmark_out.exists()
