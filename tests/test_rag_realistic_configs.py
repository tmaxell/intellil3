from __future__ import annotations

import glob
import json
from pathlib import Path

import pytest
import yaml

from benchmarks.measurement.run_all_comparisons import run_all_comparisons
from benchmarks.measurement.run_comparison import run_comparison
from benchmarks.measurement.run_measurement import MeasurementSuite


_CONFIGS_DIR = Path("benchmarks/configs")
_MAIN_CONFIG = _CONFIGS_DIR / "exp10_rag_realistic.yaml"
_LIGHTWEIGHT_CONFIG = _CONFIGS_DIR / "lightweight" / "exp10_rag_realistic_lightweight.yaml"

_RAG_METRIC_KEYS = {
    "evidence_recall",
    "evidence_precision",
    "answer_correctness",
    "groundedness",
    "abstention_correctness",
    "rag_task_success",
}


# ---------------------------------------------------------------------------
# F1: main config exists and is valid YAML
# ---------------------------------------------------------------------------

def test_exp10_main_config_exists() -> None:
    assert _MAIN_CONFIG.exists(), f"Missing config: {_MAIN_CONFIG}"


def test_exp10_main_config_is_valid_yaml() -> None:
    config = yaml.safe_load(_MAIN_CONFIG.read_text(encoding="utf-8"))
    assert config["experiment"]["name"] == "Realistic RAG"
    assert config["workload"]["type"] == "rag_realistic"
    assert len(config["systems"]) >= 2


def test_exp10_main_config_workload_parameters() -> None:
    config = yaml.safe_load(_MAIN_CONFIG.read_text(encoding="utf-8"))
    wl = config["workload"]
    assert wl["num_queries"] >= 10
    assert wl["retrieval_top_k"] >= 3
    assert 0.0 <= wl["noise_level"] <= 1.0
    assert wl["version_preference"] in {"latest", "oldest", "any"}


def test_exp10_main_config_declares_rag_metrics() -> None:
    config = yaml.safe_load(_MAIN_CONFIG.read_text(encoding="utf-8"))
    declared = set(config.get("metrics", []))
    assert _RAG_METRIC_KEYS <= declared, f"Missing RAG metrics: {_RAG_METRIC_KEYS - declared}"


def test_exp10_main_config_runs_end_to_end(tmp_path) -> None:
    # Override num_queries to keep the test fast.
    config = yaml.safe_load(_MAIN_CONFIG.read_text(encoding="utf-8"))
    config["workload"]["num_queries"] = 6
    fast_config = tmp_path / "exp10_fast.yaml"
    fast_config.write_text(yaml.dump(config), encoding="utf-8")

    comparison = run_comparison(
        config_path=fast_config,
        output_dir=tmp_path / "out",
        repetitions=1,
    )
    metric_names = {m.metric for m in comparison.metrics}
    for key in _RAG_METRIC_KEYS:
        assert key in metric_names, f"{key} missing from comparison metrics"


# ---------------------------------------------------------------------------
# F2: lightweight config exists and is valid YAML
# ---------------------------------------------------------------------------

def test_exp10_lightweight_config_exists() -> None:
    assert _LIGHTWEIGHT_CONFIG.exists(), f"Missing config: {_LIGHTWEIGHT_CONFIG}"


def test_exp10_lightweight_config_is_valid_yaml() -> None:
    config = yaml.safe_load(_LIGHTWEIGHT_CONFIG.read_text(encoding="utf-8"))
    assert "Lightweight" in config["experiment"]["name"]
    assert config["workload"]["type"] == "rag_realistic"


def test_exp10_lightweight_config_is_smaller_than_main() -> None:
    main = yaml.safe_load(_MAIN_CONFIG.read_text(encoding="utf-8"))
    lw = yaml.safe_load(_LIGHTWEIGHT_CONFIG.read_text(encoding="utf-8"))
    assert lw["workload"]["num_queries"] <= main["workload"]["num_queries"]
    assert lw["workload"]["retrieval_top_k"] <= main["workload"]["retrieval_top_k"]


def test_exp10_lightweight_config_declares_rag_metrics() -> None:
    config = yaml.safe_load(_LIGHTWEIGHT_CONFIG.read_text(encoding="utf-8"))
    declared = set(config.get("metrics", []))
    assert _RAG_METRIC_KEYS <= declared


def test_exp10_lightweight_config_runs_end_to_end(tmp_path) -> None:
    config = yaml.safe_load(_LIGHTWEIGHT_CONFIG.read_text(encoding="utf-8"))
    config["workload"]["num_queries"] = 4
    fast_config = tmp_path / "exp10_lw_fast.yaml"
    fast_config.write_text(yaml.dump(config), encoding="utf-8")

    comparison = run_comparison(
        config_path=fast_config,
        output_dir=tmp_path / "out",
        repetitions=1,
    )
    assert comparison.experiment_name != ""
    metric_names = {m.metric for m in comparison.metrics}
    assert "evidence_recall" in metric_names


# ---------------------------------------------------------------------------
# F3: glob pattern coverage
# ---------------------------------------------------------------------------

def test_exp10_matched_by_all_runs_glob() -> None:
    matched = glob.glob("benchmarks/configs/exp*.yaml")
    assert str(_MAIN_CONFIG) in matched, (
        f"exp10 not matched by all-runs glob. Matched: {matched}"
    )


def test_exp10_lightweight_matched_by_lightweight_glob() -> None:
    matched = glob.glob("benchmarks/configs/lightweight/*.yaml")
    assert str(_LIGHTWEIGHT_CONFIG) in matched


def test_exp10_not_matched_by_storage_only_glob() -> None:
    # Storage-only runs only exp01-exp04; exp10 must NOT be included.
    matched = glob.glob("benchmarks/configs/exp0[1-4]_*.yaml")
    assert str(_MAIN_CONFIG) not in matched


def test_all_exp_configs_have_matching_lightweight(tmp_path) -> None:
    # real_storage variants are infra-heavy and intentionally have no lightweight counterpart.
    all_configs = [
        p for p in sorted(glob.glob("benchmarks/configs/exp[0-9]*.yaml"))
        if "real_storage" not in p
    ]
    lw_configs = {
        p.replace("_lightweight", "").replace("/lightweight/", "/")
        for p in glob.glob("benchmarks/configs/lightweight/exp*_lightweight.yaml")
    }
    missing = [c for c in all_configs if c not in lw_configs]
    assert not missing, f"Configs without a lightweight counterpart: {missing}"


# ---------------------------------------------------------------------------
# F3: run_all_comparisons picks up exp10 from the all-runs glob
# ---------------------------------------------------------------------------

def test_run_all_comparisons_includes_exp10(tmp_path) -> None:
    # Build a minimal fast config that matches the exp*.yaml glob pattern.
    config = yaml.safe_load(_MAIN_CONFIG.read_text(encoding="utf-8"))
    config["workload"]["num_queries"] = 4
    fast_config = tmp_path / "exp10_rag_realistic.yaml"
    fast_config.write_text(yaml.dump(config), encoding="utf-8")

    comparisons = run_all_comparisons(
        config_glob=str(tmp_path / "exp10_*.yaml"),
        output_dir=tmp_path / "all_out",
        repetitions=1,
    )
    assert len(comparisons) == 1
    assert comparisons[0].experiment_name == "Realistic RAG"

    index_md = (tmp_path / "all_out" / "comparison_index.md").read_text(encoding="utf-8")
    assert "evidence_recall" in index_md or "Realistic RAG" in index_md


def test_run_all_comparisons_lightweight_glob(tmp_path) -> None:
    config = yaml.safe_load(_LIGHTWEIGHT_CONFIG.read_text(encoding="utf-8"))
    config["workload"]["num_queries"] = 4
    fast_config = tmp_path / "exp10_rag_realistic_lightweight.yaml"
    fast_config.write_text(yaml.dump(config), encoding="utf-8")

    comparisons = run_all_comparisons(
        config_glob=str(tmp_path / "*.yaml"),
        output_dir=tmp_path / "lw_out",
        repetitions=1,
    )
    assert len(comparisons) == 1
    metric_names = {m.metric for c in comparisons for m in c.metrics}
    assert "evidence_recall" in metric_names


# ---------------------------------------------------------------------------
# F3: artifacts produced for exp10 contain RAG metrics in comparison_index
# ---------------------------------------------------------------------------

def test_comparison_index_includes_rag_key_metrics(tmp_path) -> None:
    from benchmarks.measurement.metric_catalog import KEY_METRICS, RAG_KEY_METRICS
    for key in RAG_KEY_METRICS:
        assert key in KEY_METRICS, f"{key} missing from KEY_METRICS"

    config = yaml.safe_load(_MAIN_CONFIG.read_text(encoding="utf-8"))
    config["workload"]["num_queries"] = 4
    fast_config = tmp_path / "exp10_rag_realistic.yaml"
    fast_config.write_text(yaml.dump(config), encoding="utf-8")

    run_all_comparisons(
        config_glob=str(tmp_path / "exp10_*.yaml"),
        output_dir=tmp_path / "index_out",
        repetitions=1,
    )
    index_csv = (tmp_path / "index_out" / "comparison_index.csv").read_text(encoding="utf-8")
    found = {line.split(",")[1] for line in index_csv.splitlines()[1:] if line}
    assert any(k in found for k in _RAG_METRIC_KEYS), (
        f"No RAG metrics in comparison_index.csv. Found: {found}"
    )
