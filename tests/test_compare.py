"""Compare tests: side-by-side tables, deltas, missing tasks."""
from __future__ import annotations

import json

from agentbench.benchmark import Benchmark, BenchmarkResult, TaskResult
from agentbench.compare import compare
from agentbench.runner import MockClient
from agentbench.task import Rule, Rubric, TaskDef


def _task(task_id: str) -> TaskDef:
    return TaskDef(id=task_id, name=task_id, user_prompt=f"{task_id}?", rubric=Rubric(rules=[Rule(name="min_length", params={"min_length": 1})]))


def _result(model: str, scores: dict, extra: str | None = None) -> BenchmarkResult:
    tasks = []
    for tid, score in scores.items():
        name = tid
        if extra == "b" and tid == "only_b":
            name = "only_b"
        tasks.append(
            TaskResult(
                task_id=tid,
                task_name=name,
                output="x" * 10,
                score=score,
                passed=score >= 0.7,
                latency_s=0.1,
            )
        )
    return BenchmarkResult(endpoint="mock://x", model=model, started_at="2026-01-01T00:00:00Z", tasks=tasks)


def test_compare_basic_table():
    a = _result("modelA", {"t1": 1.0, "t2": 0.5})
    b = _result("modelB", {"t1": 0.8, "t2": 0.8})
    report = compare(a, b)
    assert "modelA" in report and "modelB" in report
    assert "+0.20" in report  # t1 delta A-B
    assert "-0.30" in report  # t2 delta A-B


def test_compare_missing_task_in_one_result():
    a = _result("modelA", {"t1": 1.0})
    b = _result("modelB", {"t1": 0.5, "t2": 0.9})
    report = compare(a, b)
    assert "—" in report  # em-dash placeholders for the missing task


def test_compare_accepts_json_strings_and_dicts():
    a = _result("modelA", {"t1": 1.0})
    b = _result("modelB", {"t1": 0.5})
    report_from_json = compare(json.dumps(a.to_dict()), json.dumps(b.to_dict()))
    report_from_dict = compare(a.to_dict(), b.to_dict())
    assert "modelA" in report_from_json
    assert "modelA" in report_from_dict


def test_compare_accepts_file_paths(tmp_path):
    a = _result("modelA", {"t1": 1.0})
    b = _result("modelB", {"t1": 0.5})
    pa = tmp_path / "a.json"
    pb = tmp_path / "b.json"
    pa.write_text(json.dumps(a.to_dict()), encoding="utf-8")
    pb.write_text(json.dumps(b.to_dict()), encoding="utf-8")
    report = compare(str(pa), str(pb))
    assert "Comparison" in report


def test_compare_real_run(mock_client):
    """Two full runs against the mock endpoint compare cleanly."""
    bench = Benchmark([_task("t1"), _task("t2")])
    a = bench.run(MockClient(response="alpha", usage={"total_tokens": 5}), model="mock-a")
    b = bench.run(MockClient(response="beta", usage={"total_tokens": 9}), model="mock-b")
    report = compare(a, b)
    assert "mock-a" in report and "mock-b" in report
