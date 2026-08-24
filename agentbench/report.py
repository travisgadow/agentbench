"""Report export: JSON and markdown for benchmark results."""
from __future__ import annotations

import json
from typing import Any, Dict

from .benchmark import BenchmarkResult


def to_json(result: BenchmarkResult, pretty: bool = True) -> str:
    """Serialize a BenchmarkResult to a JSON string."""
    data = result.to_dict()
    if pretty:
        return json.dumps(data, indent=2, ensure_ascii=False)
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def _fmt_score(x: float) -> str:
    return f"{x:.2f}"


def _fmt_delta(x: float) -> str:
    sign = "+" if x > 0 else ""
    return f"{sign}{x:.2f}"


def to_markdown(
    result: BenchmarkResult,
    other: BenchmarkResult | None = None,
    title: str | None = None,
) -> str:
    """Render a BenchmarkResult (optionally compared to ``other``) as markdown.

    When ``other`` is given, a side-by-side comparison is emitted (see also
    :func:`agentbench.compare.compare`).
    """
    lines: list[str] = []
    lines.append(f"# {title or f'agentbench — {result.model}'}")
    lines.append("")
    lines.append(f"- **Endpoint:** `{result.endpoint}`")
    lines.append(f"- **Model:** `{result.model}`")
    lines.append(f"- **Started:** {result.started_at}")
    s = result.summary()
    lines.append(
        f"- **Tasks:** {s['task_count']} (passed {s['passed']}, failed {s['failed']}, errors {s['errors']})"
    )
    lines.append(f"- **Mean score:** {_fmt_score(result.mean_score)}")
    lines.append(f"- **Pass rate:** {_fmt_score(result.pass_rate)}")
    lines.append(f"- **p95 latency:** {result.p95_latency:.3f}s")
    lines.append("")

    if other is not None:
        lines.extend(_comparison_table(result, other))
    else:
        lines.append("## Tasks")
        lines.append("")
        lines.append("| Task | Score | Passed | Latency (s) |")
        lines.append("|---|---:|:---:|---:|")
        for t in result.tasks:
            mark = "\u2705" if t.passed else ("\u274c" if not t.error else "\u26a0\ufe0f")
            lines.append(f"| {t.task_name} | {_fmt_score(t.score)} | {mark} | {t.latency_s:.3f} |")
        lines.append("")

    # per-task rule detail
    lines.append("## Rule detail")
    lines.append("")
    for t in result.tasks:
        lines.append(f"### {t.task_name} — score {_fmt_score(t.score)}")
        lines.append("")
        if t.error:
            lines.append(f"> \u26a0\ufe0f error: {t.error}")
            lines.append("")
            continue
        for r in t.rules:
            mark = "\u2713" if r.passed else "\u2717"
            weight = f" (w={r.weight:g})" if r.weight != 1.0 else ""
            lines.append(f"- {mark} `{r.name}`{weight}: {r.detail or ''}")
        lines.append("")
    return "\n".join(lines)


def _comparison_table(a: BenchmarkResult, b: BenchmarkResult) -> list[str]:
    """Side-by-side task table with per-task deltas (a - b)."""
    b_by_id = {t.task_id: t for t in b.tasks}
    lines: list[str] = []
    lines.append("## Comparison")
    lines.append("")
    lines.append(f"Comparing **{a.model}** (A) vs **{b.model}** (B)")
    lines.append("")
    lines.append(
        f"Mean: A {_fmt_score(a.mean_score)} vs B {_fmt_score(b.mean_score)} "
        f"(delta {_fmt_delta(a.mean_score - b.mean_score)})"
    )
    lines.append(
        f"Pass rate: A {_fmt_score(a.pass_rate)} vs B {_fmt_score(b.pass_rate)}"
    )
    lines.append("")
    lines.append("| Task | A score | B score | Delta | A pass | B pass |")
    lines.append("|---|---:|---:|---:|:---:|:---:|")
    for ta in a.tasks:
        tb = b_by_id.get(ta.task_id)
        if tb is None:
            lines.append(f"| {ta.task_name} | {_fmt_score(ta.score)} | — | — | {'\u2705' if ta.passed else '\u274c'} | — |")
            continue
        delta = ta.score - tb.score
        lines.append(
            f"| {ta.task_name} | {_fmt_score(ta.score)} | {_fmt_score(tb.score)} | "
            f"{_fmt_delta(delta)} | {'\u2705' if ta.passed else '\u274c'} | {'\u2705' if tb.passed else '\u274c'} |"
        )
    # tasks only present in B
    a_ids = {t.task_id for t in a.tasks}
    for tb in b.tasks:
        if tb.task_id not in a_ids:
            lines.append(f"| {tb.task_name} | — | {_fmt_score(tb.score)} | — | — | {'\u2705' if tb.passed else '\u274c'} |")
    lines.append("")
    return lines
