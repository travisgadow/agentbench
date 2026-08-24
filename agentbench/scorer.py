"""Rubric evaluation: turn a model output into a weighted 0.0-1.0 score.

Each rule is evaluated independently and yields a pass/fail plus optional
detail. The overall score is the weighted fraction of passing rules, and the
task passes when the score meets the rubric's ``pass_threshold``.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, List, Optional

from .task import Rule, Rubric


@dataclass
class RuleResult:
    """Outcome of one rule against one output."""

    name: str
    passed: bool
    detail: Optional[str] = None
    weight: float = 1.0


def _extract_json(text: str) -> Optional[Any]:
    """Parse JSON from a model output, tolerating markdown fences and noise."""
    if text is None:
        return None
    text = text.strip()
    candidates: List[str] = [text]
    if text.startswith("```"):
        body = text.strip("`").strip()
        # drop a leading language tag line like ```json
        lines = body.splitlines()
        if lines and lines[0].strip().lower() in {"json", "jsonc", "js", "javascript"}:
            body = "\n".join(lines[1:]).strip()
        candidates.append(body)
    # first object/array span
    for opener, closer in (("{", "}"), ("[", "]")):
        i, j = text.find(opener), text.rfind(closer)
        if i != -1 and j != -1 and j > i:
            candidates.append(text[i:j + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return None


def _rule_detail(rule: Rule) -> str:
    """Short human-readable description of a rule for reports."""
    p = rule.params
    if rule.name == "contains":
        vals = p["contains"]
        return f"contains {vals!r}" if not isinstance(vals, list) else f"contains any of {vals!r}"
    if rule.name == "not_contains":
        vals = p["not_contains"]
        return f"excludes {vals!r}" if not isinstance(vals, list) else f"excludes any of {vals!r}"
    if rule.name == "json_valid":
        return "valid JSON"
    if rule.name == "json_fields":
        return f"JSON fields {p['json_fields']!r}"
    if rule.name in ("min_length", "max_length"):
        return f"{rule.name} >= {p[rule.name]}" if rule.name == "min_length" else f"{rule.name} <= {p[rule.name]}"
    if rule.name == "regex":
        return f"matches /{p['regex']}/"
    if rule.name == "python":
        return "custom python check"
    return rule.name


def evaluate_rule(rule: Rule, output: Any) -> RuleResult:
    """Evaluate one rule against a raw model output (any type)."""
    text = "" if output is None else str(output)

    def fail(detail: str = "") -> RuleResult:
        return RuleResult(name=rule.name, passed=False, detail=detail or _rule_detail(rule), weight=rule.weight)

    def ok(detail: str = "") -> RuleResult:
        return RuleResult(name=rule.name, passed=True, detail=detail or _rule_detail(rule), weight=rule.weight)

    name = rule.name
    p = rule.params

    if name == "contains":
        wanted = p["contains"]
        if isinstance(wanted, list):
            for w in wanted:
                if w in text:
                    return ok(f"contains {w!r}")
            return fail(f"none of {wanted!r} present")
        return ok() if wanted in text else fail(f"missing {wanted!r}")

    if name == "not_contains":
        forbidden = p["not_contains"]
        if isinstance(forbidden, list):
            for f in forbidden:
                if f in text:
                    return fail(f"found {f!r}")
            return ok(f"none of {forbidden!r} present")
        return ok() if forbidden not in text else fail(f"found {forbidden!r}")

    if name == "json_valid":
        value = _extract_json(text)
        strict = p.get("strict", True)
        if value is None:
            return fail("output is not valid JSON")
        if strict and not isinstance(value, (dict, list)):
            return fail("valid JSON but not an object/array")
        return ok()

    if name == "json_fields":
        value = _extract_json(text)
        if not isinstance(value, dict):
            return fail("output is not a JSON object")
        wanted = p["json_fields"]
        if isinstance(wanted, str):
            wanted = [wanted]
        missing = [k for k in wanted if k not in value]
        if missing:
            return fail(f"missing keys: {missing!r}")
        return ok(f"has keys {wanted!r}")

    if name == "min_length":
        bound = p["min_length"]
        if len(text) >= bound:
            return ok(f"len {len(text)} >= {bound}")
        return fail(f"len {len(text)} < {bound}")

    if name == "max_length":
        bound = p["max_length"]
        if len(text) <= bound:
            return ok(f"len {len(text)} <= {bound}")
        return fail(f"len {len(text)} > {bound}")

    if name == "regex":
        if re.search(p["regex"], text):
            return ok(f"matches /{p['regex']}/")
        return fail(f"no match for /{p['regex']}/")

    if name == "python":
        try:
            result = p["python"](text)
        except Exception as exc:  # a broken checker should not kill the run
            return fail(f"checker raised {type(exc).__name__}: {exc}")
        if result is True:
            return ok()
        return fail("checker returned False")

    return fail(f"unknown rule '{name}'")


def score_rubric(rubric: Rubric, output: Any) -> dict:
    """Score an output against a rubric.

    Returns a dict: ``{score, passed, rules: [RuleResult, ...]}``.
    The score is the weighted fraction of passing rules (0.0 with no rules).
    """
    if not rubric.rules:
        return {"score": 0.0, "passed": False, "rules": []}
    results = [evaluate_rule(rule, output) for rule in rubric.rules]
    total_weight = sum(rule.weight for rule in rubric.rules)
    earned = sum(r.weight for r in results if r.passed)
    score = earned / total_weight if total_weight > 0 else 0.0
    passed = score >= rubric.pass_threshold
    return {"score": score, "passed": passed, "rules": results}
