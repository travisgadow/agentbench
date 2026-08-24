#!/usr/bin/env python3
"""CI gate: run agentbench against an agentflow pipeline and fail on regression.

This is the "build (agentflow) -> verify (agentbench) -> ship" pattern.
It imports an agentflow pipeline module, wraps its `run()` in a client that
speaks agentbench's interface, runs the research suite, and exits non-zero
if the pass rate or mean score drops below the thresholds.

Usage:
    AGENTBENCH_GATE_MODULE=agentflow_gate_pipeline \
    AGENTBENCH_GATE_RUN=min_score \\\
        python examples/agentflow_gate.py --min-score 0.7 --min-pass-rate 0.8

The target module must expose:
    def run(task: str) -> str   # accepts a user prompt, returns the final output
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys

from agentbench import Benchmark, to_markdown


class AgentflowClient:
    """Adapter: expose an agentflow `run(task) -> str` as an agentbench client."""

    endpoint = "agentflow://pipeline"
    model = "agentflow"

    def __init__(self, run_fn) -> None:
        self._run = run_fn
        self.usage = {}

    def chat(self, messages, **overrides):
        import time

        prompt = "".join(m.get("content", "") for m in messages if m.get("role") == "user")
        started = time.perf_counter()
        try:
            content = self._run(prompt)
        except Exception as exc:  # noqa: BLE001
            from agentbench.runner import EndpointError

            raise EndpointError(f"agentflow pipeline failed: {exc}") from exc
        return _Resp(content, time.perf_counter() - started)


class _Resp:
    def __init__(self, content: str, latency: float) -> None:
        self.content = content
        self.latency_s = latency
        self.usage = {}
        self.model = "agentflow"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="benchmarks/research")
    ap.add_argument("--min-score", type=float, default=0.7)
    ap.add_argument("--min-pass-rate", type=float, default=0.8)
    ap.add_argument("--output", default="gate_report.md")
    args = ap.parse_args()

    module_name = os.environ.get("AGENTBENCH_GATE_MODULE")
    if not module_name:
        print("error: set AGENTBENCH_GATE_MODULE to an importable module exposing run(task) -> str", file=sys.stderr)
        return 2
    module = importlib.import_module(module_name)
    if not hasattr(module, "run"):
        print(f"error: module {module_name} has no run() function", file=sys.stderr)
        return 2

    client = AgentflowClient(module.run)
    bench = Benchmark.load_path(args.suite)
    result = bench.run(client)
    s = result.summary()

    print(f"tasks={s['task_count']}  mean_score={s['mean_score']:.3f}  pass_rate={s['pass_rate']:.0%}  errors={s['errors']}")
    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write(to_markdown(result) + "\n")
    with open("gate_result.json", "w", encoding="utf-8") as fh:
        fh.write(json.dumps(result.to_dict(), indent=2) + "\n")

    ok = s["errors"] == 0 and s["mean_score"] >= args.min_score and s["pass_rate"] >= args.min_pass_rate
    verdict = "PASS" if ok else "FAIL"
    print(f"gate: {verdict} (thresholds: min_score={args.min_score}, min_pass_rate={args.min_pass_rate})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
