"""Benchmark loading, filtering, and aggregation tests."""
from __future__ import annotations

import pytest
import yaml

from agentbench.benchmark import Benchmark
from agentbench.task import Rule, Rubric, TaskDef, TaskError


def _task(task_id: str, tag: str) -> TaskDef:
    return TaskDef(
        id=task_id,
        name=task_id,
        user_prompt=f"task {task_id}",
        rubric=Rubric(rules=[Rule(name="contains", params={"contains": "x"})]),
        tags=[tag],
    )


def test_benchmark_requires_tasks():
    with pytest.raises(TaskError, match="at least one task"):
        Benchmark([])


def test_benchmark_rejects_duplicate_ids():
    with pytest.raises(TaskError, match="duplicate"):
        Benchmark([_task("a", "t"), _task("a", "t")])


def test_load_from_directory(tmp_path):
    for i, tag in enumerate(["one", "two", "three"]):
        (tmp_path / f"task{i}.yaml").write_text(yaml.safe_dump(_task(f"t{i}", tag).to_dict()), encoding="utf-8")
    bench = Benchmark.load_path(tmp_path)
    assert len(bench.tasks) == 3


def test_load_from_single_file(tmp_path):
    path = tmp_path / "solo.yaml"
    path.write_text(yaml.safe_dump(_task("solo", "t").to_dict()), encoding="utf-8")
    bench = Benchmark.load_path(path)
    assert len(bench.tasks) == 1 and bench.tasks[0].id == "solo"


def test_load_suite_with_tasks_list(tmp_path):
    path = tmp_path / "suite.yaml"
    doc = {
        "suite": "mysuite",
        "tasks": [_task("a", "x").to_dict(), _task("b", "y").to_dict()],
    }
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    bench = Benchmark.load_path(path)
    assert bench.name == "mysuite" and len(bench.tasks) == 2


def test_load_empty_dir_rejected(tmp_path):
    with pytest.raises(TaskError, match="no YAML"):
        Benchmark.load_path(tmp_path)


def test_load_missing_path_rejected(tmp_path):
    with pytest.raises(TaskError, match="not found"):
        Benchmark.load_path(tmp_path / "nope.yaml")


def test_filter_by_tag():
    bench = Benchmark([_task("a", "x"), _task("b", "y"), _task("c", "x")])
    filtered = bench.filter(tags=["x"])
    assert {t.id for t in filtered.tasks} == {"a", "c"}


def test_filter_by_task_ids():
    bench = Benchmark([_task("a", "x"), _task("b", "y")])
    filtered = bench.filter(task_ids=["b"])
    assert [t.id for t in filtered.tasks] == ["b"]


def test_filter_no_match_rejected():
    bench = Benchmark([_task("a", "x")])
    with pytest.raises(TaskError, match="no tasks"):
        bench.filter(tags=["nope"])


def test_aggregates(mock_client, sample_task):
    second = TaskDef(
        id="second",
        name="Second",
        user_prompt="task two",
        rubric=Rubric(rules=[Rule(name="contains", params={"contains": "x"})]),
    )
    bench = Benchmark([sample_task, second])
    result = bench.run(mock_client)
    assert len(result.tasks) == 2
    assert result.pass_rate == 1.0
    assert result.p95_latency >= 0.0
    assert result.summary()["task_count"] == 2


def test_run_records_errors(mock_client):
    # MockClient raises on prompts containing FAIL_500
    task = TaskDef(id="boom", name="Boom", user_prompt="please FAIL_500 now", rubric=Rubric(rules=[]))
    result = Benchmark([task]).run(mock_client)
    assert result.tasks[0].error is not None
    assert result.tasks[0].passed is False
    assert result.summary()["errors"] == 1
