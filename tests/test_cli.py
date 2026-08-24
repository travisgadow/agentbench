"""CLI tests: validate, run (mock), compare."""
from __future__ import annotations

import json

import pytest
import yaml
from typer.testing import CliRunner

from agentbench import cli
from agentbench.compare import _load_result

runner = CliRunner()


def _write_task(path, task_id: str = "cli-task", prompt: str = "do it") -> None:
    doc = {
        "id": task_id,
        "name": task_id,
        "user_prompt": prompt,
        "rubric": {"rules": [{"contains": "mock"}], "pass_threshold": 0.5},
    }
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")


def test_cli_version():
    result = runner.invoke(cli.app, ["--version"])
    assert result.exit_code == 0
    assert "agentbench" in result.output


def test_validate_good(tmp_path):
    _write_task(tmp_path / "good.yaml")
    result = runner.invoke(cli.app, ["validate", str(tmp_path)])
    assert result.exit_code == 0
    assert "valid" in result.output


def test_validate_bad_yaml(tmp_path):
    (tmp_path / "bad.yaml").write_text("id: [unclosed", encoding="utf-8")
    result = runner.invoke(cli.app, ["validate", str(tmp_path)])
    assert result.exit_code == 1


def test_validate_missing_required_field(tmp_path):
    (tmp_path / "incomplete.yaml").write_text(yaml.safe_dump({"id": "x"}), encoding="utf-8")
    result = runner.invoke(cli.app, ["validate", str(tmp_path)])
    assert result.exit_code == 1


def test_run_mock_with_output_files(tmp_path):
    _write_task(tmp_path / "t.yaml")
    out_json = tmp_path / "out.json"
    out_md = tmp_path / "out.md"
    result = runner.invoke(
        cli.app,
        [
            "run", str(tmp_path),
            "--mock",
            "--output", str(out_json),
            "--markdown", str(out_md),
        ],
    )
    assert result.exit_code == 0, (result.output, getattr(result, "stderr", ""))
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["task_count"] == 1
    assert data["pass_rate"] >= 0.0
    assert "Comparison" not in out_md.read_text(encoding="utf-8")
    assert "## Tasks" in out_md.read_text(encoding="utf-8")


def test_run_mock_json_stdout(tmp_path):
    _write_task(tmp_path / "t.yaml")
    result = runner.invoke(cli.app, ["run", str(tmp_path), "--mock", "--json"])
    assert result.exit_code == 0, (result.output, getattr(result, "stderr", ""))
    assert '"task_count": 1' in result.output
    # extract the pretty-printed JSON block starting at the first '{' line
    lines = result.output.splitlines()
    idx = next(i for i, line in enumerate(lines) if line.lstrip().startswith("{"))
    payload = json.loads("\n".join(lines[idx:]))
    assert payload["task_count"] == 1


def test_run_requires_endpoint_or_mock(tmp_path):
    _write_task(tmp_path / "t.yaml")
    result = runner.invoke(cli.app, ["run", str(tmp_path), "--model", "x"])
    assert result.exit_code == 1
    combined = result.output + getattr(result, "stderr", "")
    assert "endpoint" in combined.lower()


def test_compare_cli_with_fixtures(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    base = {
        "endpoint": "e", "model": "A", "started_at": "t",
        "tasks": [
            {"task_id": "t1", "task_name": "t1", "output": "x", "score": 1.0, "passed": True,
             "rules": [], "latency_s": 0.1, "usage": {}, "error": None, "model": "A"}
        ],
        "metadata": {},
    }
    base_b = dict(base)
    base_b["model"] = "B"
    base_b["tasks"] = [dict(base["tasks"][0], model="B", score=0.5, passed=False)]
    a.write_text(json.dumps(base), encoding="utf-8")
    b.write_text(json.dumps(base_b), encoding="utf-8")

    result = runner.invoke(cli.app, ["compare", str(a), str(b), "--output", str(tmp_path / "report.md")])
    assert result.exit_code == 0
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "modelA" not in report
    assert "A" in report and "B" in report


def test_compare_cli_bad_input(tmp_path):
    (tmp_path / "bad.json").write_text("not json", encoding="utf-8")
    result = runner.invoke(cli.app, ["compare", str(tmp_path / "bad.json"), str(tmp_path / "bad.json")])
    assert result.exit_code == 1
