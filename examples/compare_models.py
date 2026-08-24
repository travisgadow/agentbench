#!/usr/bin/env python3
"""Compare two models (or two configs of one model) on the research suite.

Example:
    python examples/compare_models.py \\
        --endpoint http://localhost:11434/v1 \\
        --model-a qwen2.5:14b --temperature-a 0.3 \\
        --model-b qwen2.5:14b --temperature-b 0.7

Writes a side-by-side markdown report (default: comparison.md).
"""
from __future__ import annotations

import argparse
import json

from agentbench import Benchmark, LLMClient, compare


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", default="http://localhost:11434/v1")
    ap.add_argument("--suite", default="benchmarks/research")
    ap.add_argument("--model-a", required=True)
    ap.add_argument("--model-b", required=True)
    ap.add_argument("--temperature-a", type=float, default=None)
    ap.add_argument("--temperature-b", type=float, default=None)
    ap.add_argument("--api-key", default="ollama")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--output", default="comparison.md")
    args = ap.parse_args()

    bench = Benchmark.load_path(args.suite)

    def run(model: str, temperature: float | None) -> "Benchmark":
        client = LLMClient(
            endpoint=args.endpoint,
            model=model,
            api_key=args.api_key,
            timeout=args.timeout,
            temperature=temperature,
        )
        return bench.run(client, concurrency=args.concurrency)

    print(f"Running A: {args.model_a} (temp={args.temperature_a})")
    result_a = run(args.model_a, args.temperature_a)
    print(f"Running B: {args.model_b} (temp={args.temperature_b})")
    result_b = run(args.model_b, args.temperature_b)

    # keep raw JSON around for `agentbench compare` later
    for name, result in (("a", result_a), ("b", result_b)):
        path = f"results_{name}.json"
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(result.to_dict(), indent=2) + "\n")

    report = compare(result_a, result_b)
    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    print(f"wrote {args.output} (+ results_a.json, results_b.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
