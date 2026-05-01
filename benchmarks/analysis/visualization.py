from __future__ import annotations

from pathlib import Path

import numpy as np

from benchmarks.metrics import Metrics


def plot_latency_cdf(
    baseline: Metrics,
    candidate: Metrics,
    output_path: str | Path,
) -> Path:
    plt = _import_pyplot()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_cdf(ax, baseline.latencies_ms, "Baseline")
    _plot_cdf(ax, candidate.latencies_ms, "Candidate")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("CDF")
    ax.set_title("Latency Distribution")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_cache_hit_rate(results: dict[str, Metrics], output_path: str | Path) -> Path:
    plt = _import_pyplot()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    names = list(results)
    hit_rates = [results[name].cache_hit_rate for name in names]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(names, hit_rates)
    ax.set_ylabel("Cache hit rate")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Cache Hit Rate by System")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_cdf(ax, values: list[float], label: str) -> None:
    if not values:
        ax.plot([], [], label=label)
        return
    sorted_values = sorted(values)
    ax.plot(sorted_values, np.linspace(0, 1, len(sorted_values)), label=label)


def _import_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for benchmark visualizations"
        ) from exc
    return plt
