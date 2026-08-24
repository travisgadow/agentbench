"""Benchmark suite loading, execution, and result aggregation."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Union

from .runner import EndpointError, LLMClient, LLMResponse, MockClient
from .scorer import RuleResult, score_rubric
from .task import TaskDef, TaskError


@dataclass
class TaskResult:
    """Outcome of one task: score, pass/fail, rule results, timing, usage."""

    task_id: str
    task_name: str
    output: Optional[str]
    score: float
    passed: bool
    rules: List[RuleResult] = field(default_factory=list)
    latency_s: float = 0.0
    usage: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    model: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "output": self.output,
            "score": round(self.score, 4),
            "passed": self.passed,
            "rules": [
                {"name": r.name, "passed": r.passed, "detail": r.detail, "weight": r.weight}
                for r in self.rules
            ],
            "latency_s": round(self.latency_s, 4),
            "usage": dict(self.usage),
            "error": self.error,
            "model": self.model,
        }


@dataclass
class BenchmarkResult:
    """Aggregate result for a benchmark run."""

    endpoint: str
    model: str
    started_at: str
    tasks: List[TaskResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def mean_score(self) -> float:
        if not self.tasks:
            return 0.0
        return sum(t.score for t in self.tasks) / len(self.tasks)

    @property
    def pass_rate(self) -> float:
        if not self.tasks:
            return 0.0
        return sum(1 for t in self.tasks if t.passed) / len(self.tasks)

    @property
    def p95_latency(self) -> float:
        latencies = sorted(t.latency_s for t in self.tasks)
        if not latencies:
            return 0.0
        idx = int(round(0.95 * (len(latencies) - 1)))
        return latencies[idx]

    def summary(self) -> Dict[str, Any]:
        return {
            "endpoint": self.endpoint,
            "model": self.model,
            "started_at": self.started_at,
            "task_count": len(self.tasks),
            "passed": sum(1 for t in self.tasks if t.passed),
            "failed": sum(1 for t in self.tasks if not t.passed),
            "errors": sum(1 for t in self.tasks if t.error),
            "mean_score": round(self.mean_score, 4),
            "pass_rate": round(self.pass_rate, 4),
            "p95_latency_s": round(self.p95_latency, 4),
        }

    def to_dict(self) -> Dict[str, Any]:
        d = self.summary()
        d["tasks"] = [t.to_dict() for t in self.tasks]
        d["metadata"] = dict(self.metadata)
        return d


def _utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Benchmark:
    """A suite of tasks, runnable against any client with a ``chat`` method."""

    def __init__(self, tasks: List[TaskDef], name: str = "benchmark") -> None:
        if not tasks:
            raise TaskError("benchmark must contain at least one task")
        seen = set()
        for t in tasks:
            if t.id in seen:
                raise TaskError(f"duplicate task id: {t.id}")
            seen.add(t.id)
        self.tasks = tasks
        self.name = name

    # -- loading ------------------------------------------------------------

    @classmethod
    def load_path(cls, path: Union[str, os.PathLike]) -> "Benchmark":
        """Load from a YAML file (single task or suite) or a directory of YAML files (recursive)."""
        p = os.fspath(path)
        if os.path.isdir(p):
            files = sorted(
                os.path.join(dp, f)
                for dp, _, fns in os.walk(p)
                for f in fns
                if f.lower().endswith((".yaml", ".yml"))
            )
            if not files:
                raise TaskError(f"no YAML task files found in {p}")
            import yaml as _yaml

            tasks: List[TaskDef] = []
            suite_name = None
            for f in files:
                with open(f, "r", encoding="utf-8") as fh:
                    data = _yaml.safe_load(fh)
                if isinstance(data, dict) and isinstance(data.get("tasks"), list) and "id" not in data:
                    if suite_name is None:
                        suite_name = data.get("suite") or data.get("name")
                    for i, td in enumerate(data["tasks"]):
                        tasks.append(TaskDef.from_dict(td, context=f"{os.path.basename(f)}.tasks[{i}]"))
                else:
                    tasks.append(TaskDef.from_dict(data, context=os.path.basename(f)))
            return cls(tasks, name=str(suite_name or os.path.basename(os.path.abspath(p))))
        if not os.path.isfile(p):
            raise TaskError(f"path not found: {p}")
        import yaml as _yaml
        with open(p, "r", encoding="utf-8") as fh:
            data = _yaml.safe_load(fh)
        if isinstance(data, dict) and isinstance(data.get("tasks"), list):
            suite_name = data.get("suite") or data.get("name")
            tasks = [
                TaskDef.from_dict(td, context=f"{os.path.basename(p)}.tasks[{i}]")
                for i, td in enumerate(data["tasks"])
            ]
            return cls(tasks, name=str(suite_name or os.path.basename(p)))
        return cls([TaskDef.from_dict(data, context=os.path.basename(p))], name=os.path.basename(p))

    # -- filtering ----------------------------------------------------------

    def filter(self, tags: Optional[List[str]] = None, task_ids: Optional[List[str]] = None) -> "Benchmark":
        """Return a new Benchmark with only matching tasks (AND across filters)."""
        tasks = self.tasks
        if tags:
            wanted = set(tags)
            tasks = [t for t in tasks if wanted & set(t.tags)]
        if task_ids:
            wanted = set(task_ids)
            tasks = [t for t in tasks if t.id in wanted]
        if not tasks:
            raise TaskError("filter matched no tasks")
        return Benchmark(tasks, name=f"{self.name} (filtered)")

    # -- execution ----------------------------------------------------------

    def run(
        self,
        client: Union[LLMClient, MockClient, Any],
        model: Optional[str] = None,
        endpoint: Optional[str] = None,
        concurrency: int = 1,
    ) -> BenchmarkResult:
        """Run every task against ``client`` and aggregate results.

        ``concurrency`` > 1 runs tasks in a thread pool (order preserved in output).
        """
        endpoint = endpoint or getattr(client, "endpoint", "unknown")
        model = model or getattr(client, "model", "unknown")
        result = BenchmarkResult(
            endpoint=str(endpoint),
            model=str(model),
            started_at=_utc_now_iso(),
        )

        def _run_one(task: TaskDef) -> TaskResult:
            messages: List[Dict[str, str]] = []
            if task.system_prompt:
                messages.append({"role": "system", "content": task.system_prompt})
            messages.append({"role": "user", "content": task.user_prompt})
            overrides = {
                k: task.metadata[k]
                for k in ("temperature", "max_tokens")
                if k in task.metadata and task.metadata[k] is not None
            }
            try:
                resp: LLMResponse = client.chat(messages, **overrides)
            except (EndpointError, Exception) as exc:  # noqa: BLE001 - record, don't crash the suite
                return TaskResult(
                    task_id=task.id,
                    task_name=task.name,
                    output=None,
                    score=0.0,
                    passed=False,
                    error=str(exc),
                    model=model,
                )
            scoring = score_rubric(task.rubric, resp.content)
            return TaskResult(
                task_id=task.id,
                task_name=task.name,
                output=resp.content,
                score=scoring["score"],
                passed=scoring["passed"],
                rules=scoring["rules"],
                latency_s=resp.latency_s,
                usage=dict(resp.usage),
                model=resp.model or model,
            )

        if concurrency and concurrency > 1 and len(self.tasks) > 1:
            by_id: Dict[str, TaskResult] = {}
            with ThreadPoolExecutor(max_workers=min(concurrency, len(self.tasks))) as pool:
                futures = {pool.submit(_run_one, t): t for t in self.tasks}
                for fut in as_completed(futures):
                    tr = fut.result()
                    by_id[tr.task_id] = tr
            result.tasks = [by_id[t.id] for t in self.tasks]
        else:
            result.tasks = [_run_one(t) for t in self.tasks]
        return result
