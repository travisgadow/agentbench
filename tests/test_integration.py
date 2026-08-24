"""Integration: validate the shipped benchmark suites + full offline run."""
from __future__ import annotations

from pathlib import Path

from agentbench.benchmark import Benchmark
from agentbench.runner import MockClient
from agentbench.report import to_json, to_markdown

ROOT = Path(__file__).resolve().parents[1]
SUITE_DIRS = [ROOT / "benchmarks" / s for s in ("basic", "research", "tooluse")]


def test_shipped_suites_are_valid():
    total = 0
    for d in SUITE_DIRS:
        assert d.is_dir(), f"missing suite dir: {d}"
        bench = Benchmark.load_path(d)
        assert len(bench.tasks) == 5, f"{d.name} should have 5 tasks, has {len(bench.tasks)}"
        for t in bench.tasks:
            assert t.rubric.rules, f"{t.id} has no rubric rules"
        total += len(bench.tasks)
    assert total == 15


def test_full_offline_run_produces_reports():
    bench = Benchmark.load_path(ROOT / "benchmarks")
    assert len(bench.tasks) == 15
    client = MockClient(response='{"tool": "mock", "note": "offline"}')
    result = bench.run(client, model="mock-1")
    assert len(result.tasks) == 15
    # every task ran without transport errors
    assert result.summary()["errors"] == 0
    js = to_json(result)
    md = to_markdown(result)
    assert '"task_count": 15' in js
    assert "## Tasks" in md
