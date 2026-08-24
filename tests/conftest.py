"""Shared fixtures for the agentbench test suite."""
from __future__ import annotations

import pytest

from agentbench.runner import MockClient
from agentbench.task import Rule, Rubric, TaskDef


@pytest.fixture
def mock_client() -> MockClient:
    """Offline mock endpoint returning a fixed JSON-ish response."""
    return MockClient(
        response='{"tool": "demo_x", "note": "offline mock output"}',
        usage={"prompt_tokens": 5, "completion_tokens": 15, "total_tokens": 20},
    )


@pytest.fixture
def sample_task() -> TaskDef:
    rubric = Rubric(
        rules=[
            Rule(name="json_valid", params={"strict": True}),
            Rule(name="contains", params={"contains": "tool"}),
            Rule(name="not_contains", params={"not_contains": "```"}, weight=0.5),
        ],
        pass_threshold=0.7,
    )
    return TaskDef(
        id="sample",
        name="Sample task",
        user_prompt="Do the thing.",
        rubric=rubric,
        tags=["sample"],
        metadata={"temperature": 0.2},
    )
