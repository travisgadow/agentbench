# CI gate demo: a tiny agentflow-style pipeline used as an agentbench target.
#
# Exposes: run(task: str) -> str
#
# For the real thing, import your actual agentflow Pipeline and delegate to
# it inside `run()` below. This stub lets the gate example run standalone.
from __future__ import annotations


def run(task: str) -> str:
    """Stub pipeline: pretend to research and return a structured report."""
    return (
        "## Summary\n"
        f"Report for: {task[:60]}\n\n"
        "## Evidence\n"
        "- local inference keeps data on-device\n"
        "- OpenAI-compatible endpoints make local servers swappable\n\n"
        "## Risks\n"
        "- rubric-based scoring misses nuance\n\n"
        "## Recommendation\n"
        "- run the research suite weekly in CI\n"
    )
