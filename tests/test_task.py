"""TaskDef / Rubric parsing and validation tests."""
from __future__ import annotations

import pytest
import yaml

from agentbench.task import Rule, Rubric, TaskDef, TaskError


def _valid_doc() -> dict:
    return {
        "id": "t1",
        "name": "T1",
        "user_prompt": "hello",
        "rubric": {"rules": [{"contains": "world"}], "pass_threshold": 0.5},
        "tags": ["a"],
    }


def test_task_from_dict_valid():
    task = TaskDef.from_dict(_valid_doc())
    assert task.id == "t1"
    assert task.rubric.rules[0].name == "contains"
    assert task.rubric.pass_threshold == 0.5


def test_task_missing_required_fields():
    for missing in ("id", "user_prompt", "rubric"):
        doc = _valid_doc()
        doc.pop(missing)
        with pytest.raises(TaskError, match=missing):
            TaskDef.from_dict(doc)


def test_task_bad_id_slug():
    doc = _valid_doc()
    doc["id"] = "has space"
    with pytest.raises(TaskError, match="id"):
        TaskDef.from_dict(doc)


def test_rubric_unknown_rule_rejected():
    with pytest.raises(TaskError, match="unknown rule"):
        Rubric.from_dict({"rules": [{"frobnicate": True}]})


def test_rubric_bad_threshold_rejected():
    with pytest.raises(TaskError, match="pass_threshold"):
        Rubric.from_dict({"rules": [{"contains": "x"}], "pass_threshold": 1.5})


def test_rubric_bad_weight_rejected():
    with pytest.raises(TaskError, match="weight"):
        Rule.from_dict({"contains": "x", "weight": -1})


def test_rule_regex_invalid_rejected():
    with pytest.raises(TaskError, match="regex"):
        Rule.from_dict({"regex": "([unclosed"})


def test_rule_python_non_callable_rejected():
    with pytest.raises(TaskError, match="python"):
        Rule.from_dict({"python": "not-a-function"})


def test_task_yaml_roundtrip(tmp_path):
    task = TaskDef.from_dict(_valid_doc())
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(task.to_dict()), encoding="utf-8")
    loaded = TaskDef.load(path)
    assert loaded.id == task.id
    assert [r.name for r in loaded.rubric.rules] == [r.name for r in task.rubric.rules]
