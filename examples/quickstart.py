#!/usr/bin/env python3
"""agentbench quickstart: run the basic suite against a local Ollama model.

Prereq: `ollama serve` running and a model pulled (e.g. `ollama pull qwen2.5:7b`).

Usage:
    python examples/quickstart.py [endpoint] [model]
"""
from __future__ import annotations

import sys

from agentbench import Benchmark, LLMClient, to_markdown

DEFAULT_ENDPOINT = "http://localhost:11434/v1"
DEFAULT_MODEL = "qwen2.5:7b"


def main() -> int:
    endpoint = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ENDPOINT
    model = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL

    bench = Benchmark.load_path("benchmarks/basic")
    print(f"Running {len(bench.tasks)} tasks against {model} @ {endpoint}")

    client = LLMClient(endpoint=endpoint, model=model, api_key="ollama", timeout=300, retries=1)
    result = bench.run(client, concurrency=2)

    print()
    for t in result.tasks:
        mark = "PASS" if t.passed else ("ERROR" if t.error else "FAIL")
        print(f"  [{mark}] {t.task_name}: {t.score:.2f} ({t.latency_s:.1f}s)")
    s = result.summary()
    print(f"\nmean={s['mean_score']:.2f}  pass_rate={s['pass_rate']:.0%}  p95={result.p95_latency:.1f}s")

    report = to_markdown(result)
    with open("quickstart_report.md", "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    print("wrote quickstart_report.md")
    return 0 if s["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
