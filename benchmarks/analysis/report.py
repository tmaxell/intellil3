from __future__ import annotations

import argparse
import json
from pathlib import Path


def render_markdown(results: list[dict]) -> str:
    lines = ["# Benchmark Report", ""]
    for result in results:
        metrics = result["metrics"]
        lines.extend(
            [
                f"## {result['system_name']}",
                "",
                f"- Experiment: {result['experiment_name']}",
                f"- Workload: {result['workload_type']}",
                f"- Latency P95: {metrics['latency_p95_ms']:.3f} ms",
                f"- Cache hit rate: {metrics['cache_hit_rate']:.3f}",
                f"- Throughput: {metrics['throughput_req_s']:.3f} req/s",
                "",
            ]
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    Path(args.output).write_text(render_markdown(results), encoding="utf-8")


if __name__ == "__main__":
    main()
