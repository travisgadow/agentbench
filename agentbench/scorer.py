"""Rubric evaluation: turn a model output into a weighted 0.0-1.0 score.

Each rule is evaluated independently and yields a pass/fail plus optional
detail. The overall score is the weighted fraction of passing rules, and the
task passes when the score meets the rubric's ``pass_threshold``.

``json_valid`` semantics:
    * ``strict: true`` (default) — the output must contain a JSON *object* or
      *array*. Bare scalars (``42``, ``"str"``, ``true``) are rejected.
    * ``strict: false`` — *any* valid JSON value is accepted, including scalars.
    Extraction is robust to markdown fences and to prose around a JSON span.

``python`` rules accept either an in-memory callable or a ``module.path:func``
string reference (resolvable from YAML); the checker receives the raw output
string and a *truthy* return value counts as a pass.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .task import Rule, Rubric


@dataclass
class RuleResult:
    """Outcome of one rule against one output."""

    name: str
    passed: bool
    detail: Optional[str] = None
    weight: float = 1.0


def _balanced_span(text: str, start: int) -> Optional[str]:
    """Return the balanced ``{...}``/``[...]`` span starting at ``start``.

    Tracks nesting while respecting JSON string literals and escapes, so
    braces/brackets inside strings do not break the match.
    """
    opener = text[start]
    closer = "}" if opener == "{" else ("]" if opener == "[" else None)
    if closer is None:
        return None
    depth = 0
    in_str = False
    escaped = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start:idx + 1]
    return None


def _strip_fence(text: str) -> Optional[str]:
    """Return the body of a ```json``` fence, or None if not fenced."""
    if not text.lstrip().startswith("```"):
        return None
    body = text.strip("`").strip()
    lines = body.splitlines()
    if lines and lines[0].strip().lower() in {"json", "jsonc", "js", "javascript"}:
        body = "\n".join(lines[1:]).strip()
    return body


def _extract_json(text: str) -> Optional[Any]:
    """Parse JSON from a model output, tolerating markdown fences and prose.

    Unlike a naive first-``{``-to-last-``}`` slice, this scans for the first
    *balanced, valid* JSON object or array, so prose with several JSON spans
    (or a single object surrounded by text) is handled correctly.
    """
    if text is None:
        return None
    text = text.strip()

    def _try(candidate: Optional[str]) -> Optional[Any]:
        if not candidate:
            return None
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError, ValueError):
            return None

    for candidate in (text, _strip_fence(text)):
        value = _try(candidate)
        if value is not None:
            return value

    # scan every balanced object/array span, in order of appearance
    for opener in ("{", "["):
        pos = text.find(opener)
        while pos != -1:
            span = _balanced_span(text, pos)
            value = _try(span)
            if value is not None:
                return value
            pos = text.find(opener, pos + 1)
    return None


def _resolve_python(value: Any) -> Callable[[str], Any]:
    """Resolve a ``python`` rule value to a callable.

    Accepts an in-memory callable or a ``module.path:func`` string reference
    (the latter importable from YAML task files).
    """
    if callable(value):
        return value
    if isinstance(value, str):
        module_path, sep, attr_path = value.rpartition(":")
        if not sep or not module_path or not attr_path:
            raise ValueError(f"invalid python rule reference {value!r} (expected 'module.path:func')")
        import importlib

        module = importlib.import_module(module_path)
        obj: Any = module
        for part in attr_path.split("."):
            obj = getattr(obj, part)
        if not callable(obj):
            raise ValueError(f"python rule reference {value!r} did not resolve to a callable")
        return obj
    raise ValueError(f"python rule must be a callable or 'module.path:func' reference, got {type(value).__name__}")


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
            checker = _resolve_python(p.get("python"))
            result = checker(text)
        except Exception as exc:  # a broken checker should not kill the run
            return fail(f"checker {type(exc).__name__}: {exc}")
        if result:
            return ok(f"checker returned {result!r}")
        return fail(f"checker returned {result!r}")

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
