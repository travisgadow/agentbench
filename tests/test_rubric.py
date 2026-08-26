"""Rubric scoring: each rule type, weighting, thresholds, edge cases."""
from __future__ import annotations

from agentbench.scorer import evaluate_rule, score_rubric
from agentbench.task import Rule, Rubric


def _rule(name: str, **params) -> Rule:
    return Rule(name=name, params=params)


def test_contains_single_and_list():
    assert evaluate_rule(_rule("contains", contains="apple"), "I have an apple pie").passed
    assert not evaluate_rule(_rule("contains", contains="banana"), "I have an apple pie").passed
    assert evaluate_rule(_rule("contains", contains=["banana", "pie"]), "apple pie").passed
    assert not evaluate_rule(_rule("contains", contains=["banana", "kiwi"]), "apple pie").passed


def test_not_contains():
    assert evaluate_rule(_rule("not_contains", not_contains="banana"), "apple pie").passed
    assert not evaluate_rule(_rule("not_contains", not_contains="pie"), "apple pie").passed
    assert evaluate_rule(_rule("not_contains", not_contains=["x", "y"]), "apple pie").passed


def test_json_valid_variants():
    assert evaluate_rule(_rule("json_valid", strict=True), '{"a": 1}').passed
    assert evaluate_rule(_rule("json_valid", strict=True), "```json\n{\"a\": 1}\n```").passed
    assert evaluate_rule(_rule("json_valid", strict=True), 'pre {"a": 1} post').passed
    assert not evaluate_rule(_rule("json_valid", strict=True), "not json at all").passed
    assert evaluate_rule(_rule("json_valid", strict=False), "123").passed
    assert not evaluate_rule(_rule("json_valid", strict=True), '"just a string"').passed


def test_json_fields():
    r = _rule("json_fields", json_fields=["a", "b"])
    assert evaluate_rule(r, '{"a": 1, "b": 2}').passed
    res = evaluate_rule(r, '{"a": 1}')
    assert not res.passed and "b" in (res.detail or "")
    assert not evaluate_rule(r, "[1, 2]").passed
    r2 = _rule("json_fields", json_fields="a")
    assert evaluate_rule(r2, '{"a": 1}').passed


def test_min_max_length():
    assert evaluate_rule(_rule("min_length", min_length=3), "abcd").passed
    assert not evaluate_rule(_rule("min_length", min_length=5), "abcd").passed
    assert evaluate_rule(_rule("max_length", max_length=5), "abcd").passed
    assert not evaluate_rule(_rule("max_length", max_length=3), "abcd").passed


def test_regex_rule():
    assert evaluate_rule(_rule("regex", regex=r"Answer:\s*\d+"), "Answer: 42").passed
    assert not evaluate_rule(_rule("regex", regex=r"Answer:\s*\d+"), "answer 42").passed


def test_python_rule():
    ok = evaluate_rule(_rule("python", python=lambda t: "yes" in t), "yes indeed")
    assert ok.passed
    bad = evaluate_rule(_rule("python", python=lambda t: ("nope" in t, False)[0]), "hello")
    assert not bad.passed
    broken = evaluate_rule(_rule("python", python=lambda t: 1 / 0), "x")
    assert not broken.passed and "ZeroDivisionError" in (broken.detail or "")


def test_weighted_scoring_math():
    rubric = Rubric(
        rules=[
            Rule(name="contains", params={"contains": "apple"}, weight=2.0),
            Rule(name="contains", params={"contains": "banana"}, weight=1.0),
        ],
        pass_threshold=0.5,
    )
    result = score_rubric(rubric, "apple apple")
    # 2.0 of 3.0 weight earned
    assert abs(result["score"] - (2.0 / 3.0)) < 1e-9
    assert result["passed"] is True


def test_pass_threshold_boundary():
    rubric = Rubric(
        rules=[
            Rule(name="contains", params={"contains": "apple"}, weight=1.0),
            Rule(name="contains", params={"contains": "banana"}, weight=1.0),
        ],
        pass_threshold=1.0,
    )
    assert score_rubric(rubric, "apple")["passed"] is False
    assert score_rubric(rubric, "apple banana")["passed"] is True


def test_empty_rubric_scores_zero():
    result = score_rubric(Rubric(rules=[]), "anything")
    assert result["score"] == 0.0
    assert result["passed"] is False
    assert result["rules"] == []


def test_empty_output_does_not_crash():
    rubric = Rubric(rules=[Rule(name="json_valid", params={"strict": True}), Rule(name="min_length", params={"min_length": 10})])
    result = score_rubric(rubric, "")
    assert result["passed"] is False
    assert all(not r.passed for r in result["rules"])


def test_unicode_output():
    rubric = Rubric(rules=[Rule(name="contains", params={"contains": "übung"})])
    assert score_rubric(rubric, "Guten Tag, übung macht den Meister")["passed"] is True


def test_json_valid_multiple_objects_in_prose():
    # prose with several JSON spans must still extract valid JSON (P0 #3)
    rule = _rule("json_valid", strict=True)
    assert evaluate_rule(rule, 'Here is {"a": 1} and then {"b": 2} after').passed
    assert evaluate_rule(rule, 'Result: {"ok": true}').passed
    assert not evaluate_rule(rule, "no json here at all").passed


def test_json_valid_array_is_strict_ok():
    assert evaluate_rule(_rule("json_valid", strict=True), "[1, 2, 3]").passed


def test_python_rule_truthy_non_bool_passes():
    # a truthy non-bool return value counts as a pass (P0 #2)
    ok = evaluate_rule(_rule("python", python=lambda t: [1, 2]), "anything")
    assert ok.passed
    bad = evaluate_rule(_rule("python", python=lambda t: []), "anything")
    assert not bad.passed


def test_python_rule_from_module_ref():
    # a 'module.path:func' reference resolves at eval time (P2 #7)
    rule = _rule("python", python="agentbench.task:_valid_python_ref")
    assert evaluate_rule(rule, "a:b").passed
    assert not evaluate_rule(rule, "nope").passed
