"""Side-by-side comparison of two benchmark results."""
from __future__ import annotations

from typing import Any, Dict, Union

from .benchmark import BenchmarkResult
from .report import to_markdown


def _load_result(data: Union[BenchmarkResult, dict, str]) -> BenchmarkResult:
    """Accept a BenchmarkResult, its dict form, a JSON string, or a file path."""
    if isinstance(data, BenchmarkResult):
        return data
    if isinstance(data, dict):
        return _result_from_dict(data)
    if isinstance(data, str):
        import json
        text = data
        if not text.lstrip().startswith("{"):
            with open(data, "r", encoding="utf-8") as fh:
                text = fh.read()
        return _result_from_dict(json.loads(text))
    raise TypeError(f"cannot load BenchmarkResult from {type(data).__name__}")


def _result_from_dict(data: dict) -> BenchmarkResult:
    from .scorer import RuleResult
    from .benchmark import TaskResult

    tasks = [
        TaskResult(
            task_id=t["task_id"],
            task_name=t.get("task_name", t["task_id"]),
            output=t.get("output"),
            score=float(t.get("score", 0.0)),
            passed=bool(t.get("passed", False)),
            rules=[
                RuleResult(
                    name=r["name"],
                    passed=bool(r["passed"]),
                    detail=r.get("detail"),
                    weight=float(r.get("weight", 1.0)),
                )
                for r in t.get("rules", [])
            ],
            latency_s=float(t.get("latency_s", 0.0)),
            usage=dict(t.get("usage", {})),
            error=t.get("error"),
            model=t.get("model"),
        )
        for t in data.get("tasks", [])
    ]
    return BenchmarkResult(
        endpoint=data.get("endpoint", "unknown"),
        model=data.get("model", "unknown"),
        started_at=data.get("started_at", ""),
        tasks=tasks,
        metadata=dict(data.get("metadata", {})),
    )


def compare(a: Union[BenchmarkResult, dict, str], b: Union[BenchmarkResult, dict, str]) -> str:
    """Compare two results and return a markdown report.

    Inputs may be ``BenchmarkResult`` objects, their dict/JSON form, or file
    paths to JSON files.
    """
    ra, rb = _load_result(a), _load_result(b)
    return to_markdown(ra, other=rb, title=f"agentbench compare — {ra.model} vs {rb.model}")
