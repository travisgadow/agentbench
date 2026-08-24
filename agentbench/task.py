"""Task and rubric definitions.

A benchmark task is a single (prompt -> expected quality) pair, defined in
YAML and loaded into a ``TaskDef``. Quality is expressed with a composable,
weighted ``Rubric`` made of simple ``Rule`` checks.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import yaml


class TaskError(ValueError):
    """Raised when a task or rubric definition is invalid."""


RULE_KEYS = {
    "contains",
    "not_contains",
    "json_valid",
    "json_fields",
    "min_length",
    "max_length",
    "regex",
    "python",
}


def _check_type(value: Any, types: tuple, context: str) -> None:
    if not isinstance(value, types):
        raise TaskError(f"{context}: expected {types}, got {type(value).__name__}")


def _check_weight(weight: Any, context: str) -> float:
    _check_type(weight, (int, float), f"{context}.weight")
    if not isinstance(weight, bool) and weight < 0:
        raise TaskError(f"{context}.weight must be >= 0")
    return float(weight)


def _check_threshold(threshold: Any, context: str) -> float:
    _check_type(threshold, (int, float), f"{context}.pass_threshold")
    if not isinstance(threshold, bool) and not (0.0 <= threshold <= 1.0):
        raise TaskError(f"{context}.pass_threshold must be between 0 and 1")
    return float(threshold)


def _check_contains(rule: Dict[str, Any], name: str, context: str) -> None:
    _check_type(rule.get(name), (str, list), f"{context}.{name}")
    if isinstance(rule[name], list):
        for i, item in enumerate(rule[name]):
            if not isinstance(item, str):
                raise TaskError(f"{context}.{name}[{i}] must be a string")


@dataclass
class Rule:
    """One weighted scoring rule."""

    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0

    @classmethod
    def from_dict(cls, data: Any, context: str = "rule") -> "Rule":
        if not isinstance(data, dict) or not data:
            raise TaskError(f"{context}: rule must be a single-key mapping, e.g. {{contains: 'text'}}")
        body = {k: v for k, v in data.items() if k != "weight"}
        if len(body) != 1:
            raise TaskError(f"{context}: rule must be a single-key mapping, e.g. {{contains: 'text'}}")
        (name, value), = body.items()
        if name not in RULE_KEYS:
            raise TaskError(f"{context}: unknown rule '{name}' (allowed: {', '.join(sorted(RULE_KEYS))})")
        rule = cls(name=name)
        if name in ("contains", "not_contains"):
            _check_contains(data, name, context)
            rule.params[name] = value
        elif name == "json_valid":
            if isinstance(value, dict):
                rule.params["strict"] = bool(value.get("strict", True))
            elif isinstance(value, bool):
                rule.params["strict"] = value
            else:
                raise TaskError(f"{context}.json_valid must be a mapping or boolean")
        elif name == "json_fields":
            _check_type(value, (str, list), f"{context}.json_fields")
            rule.params["json_fields"] = value
        elif name in ("min_length", "max_length"):
            _check_type(value, (int, float), f"{context}.{name}")
            if not isinstance(value, bool) and value < 0:
                raise TaskError(f"{context}.{name} must be >= 0")
            rule.params[name] = int(value)
        elif name == "regex":
            _check_type(value, (str,), f"{context}.regex")
            try:
                re.compile(value)
            except re.error as exc:
                raise TaskError(f"{context}.regex: invalid pattern: {exc}") from None
            rule.params["regex"] = value
        elif name == "python":
            if not callable(value):
                raise TaskError(
                    f"{context}.python must be a callable (programmatic task construction only)"
                )
            rule.params["python"] = value
        if "weight" in data:
            rule.weight = _check_weight(data["weight"], context)
        return rule

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {self.name: self.params.get(self.name, True)}
        if self.name == "python":
            out["python"] = "<callable>"
        if self.weight != 1.0:
            out["weight"] = self.weight
        return out


@dataclass
class Rubric:
    """A weighted set of rules with an overall pass threshold."""

    rules: List[Rule]
    pass_threshold: float = 0.7

    @classmethod
    def from_dict(cls, data: Any, context: str = "rubric") -> "Rubric":
        if not isinstance(data, dict):
            raise TaskError(f"{context}: must be a mapping")
        threshold = _check_threshold(data.get("pass_threshold", 0.7), context)
        rules_raw = data.get("rules", [])
        if not isinstance(rules_raw, list):
            raise TaskError(f"{context}.rules must be a list")
        rules = [Rule.from_dict(r, context=f"{context}.rules[{i}]") for i, r in enumerate(rules_raw)]
        return cls(rules=rules, pass_threshold=threshold)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rules": [r.to_dict() for r in self.rules],
            "pass_threshold": self.pass_threshold,
        }


@dataclass
class TaskDef:
    """One benchmark task: prompt(s) + rubric + metadata."""

    id: str
    name: str
    description: str = ""
    system_prompt: Optional[str] = None
    user_prompt: str = ""
    rubric: Rubric = field(default_factory=lambda: Rubric(rules=[]))
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Any, context: str = "task") -> "TaskDef":
        if not isinstance(data, dict):
            raise TaskError(f"{context}: must be a mapping")
        for required in ("id", "user_prompt", "rubric"):
            if required not in data:
                raise TaskError(f"{context}: missing required field '{required}'")

        def _str(key: str, default: str = "") -> str:
            value = data.get(key, default)
            if not isinstance(value, str):
                raise TaskError(f"{context}.{key} must be a string")
            return value

        task_id = _str("id")
        if not task_id or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", task_id):
            raise TaskError(
                f"{context}.id must be a stable slug (letters, digits, '.', '_', '-'), got {task_id!r}"
            )
        system_prompt = data.get("system_prompt")
        if system_prompt is not None and not isinstance(system_prompt, str):
            raise TaskError(f"{context}.system_prompt must be a string")
        rubric = Rubric.from_dict(data["rubric"], context=f"{context}.rubric")
        tags = data.get("tags", [])
        if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
            raise TaskError(f"{context}.tags must be a list of strings")
        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            raise TaskError(f"{context}.metadata must be a mapping")
        return cls(
            id=task_id,
            name=_str("name", task_id),
            description=_str("description", ""),
            system_prompt=system_prompt,
            user_prompt=_str("user_prompt"),
            rubric=rubric,
            tags=list(tags),
            metadata=dict(metadata),
        )

    @classmethod
    def load(cls, path: Union[str, os.PathLike]) -> "TaskDef":
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        doc = cls._extract_task(data, context=f"file:{os.path.basename(str(path))}")
        return doc

    @classmethod
    def _extract_task(cls, data: Any, context: str) -> "TaskDef":
        """Accept either a single task document or ``{suite: ..., tasks: [...]}``."""
        if isinstance(data, dict) and isinstance(data.get("tasks"), list) and "id" not in data:
            return cls._suite_first_task(data, context)
        return cls.from_dict(data, context=context)

    @classmethod
    def _suite_first_task(cls, data: Dict[str, Any], context: str) -> "TaskDef":
        tasks = data["tasks"]
        if not tasks:
            raise TaskError(f"{context}: suite has no tasks")
        return cls.from_dict(tasks[0], context=f"{context}.tasks[0]")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "rubric": self.rubric.to_dict(),
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }
