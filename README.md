# agentbench

**A local-first benchmarking & eval harness for agentic AI pipelines.**

Most agentic-AI toolkits focus on *orchestration* — how to wire agents together.
This project focuses on *verification* — how to know they're actually working.

agentbench lets you:

- define benchmark tasks in **YAML**,
- run them against **any OpenAI-compatible endpoint** (Ollama, vLLM, cloud),
- score outputs with **composable weighted rubrics**,
- and **compare runs side-by-side** in markdown/JSON.

No cloud required. No heavy framework. Just Python.

Built as the QA counterpart to [agentflow](https://github.com/travisgadow/agentflow) —
**agentflow is the engine, agentbench is the dyno.**

## Why

- Verify that model param changes (temperature, context, top_p) actually change agent behavior.
- Regression-test agentflow releases before shipping.
- Compare models on your hardware without cloud APIs.
- Give teams a CI gate: *build → test → ship*.

The local-LLM ecosystem lacks a lightweight eval tool that runs anywhere Python does.
This is it: `requests` + `PyYAML` + stdlib-level ergonomics.

## Install

```bash
pip install -e .            # or: pip install agentbench
```

## Quick start

```bash
# 1. dry-run: check your task YAML is well-formed
agentbench validate benchmarks/

# 2. run a suite against Ollama
agentbench run benchmarks/basic \
    --endpoint http://localhost:11434/v1 \
    --model qwen2.5:14b \
    --concurrency 2 \
    --output results/basic.json \
    --markdown reports/basic.md

# 3. run the same suite offline (mock endpoint) — handy for demos & tests
agentbench run benchmarks/basic --mock --json

# 4. compare two runs side-by-side
agentbench compare results/basic_a.json results/basic_b.json --output reports/comparison.md
```

## Writing tasks

A task is one YAML file: prompt(s) + rubric + metadata.

```yaml
id: extract-facts            # stable slug, unique within a suite
name: Fact extraction to JSON
tags: [basic, json]
system_prompt: You are a precise data-extraction agent. JSON only.
user_prompt: >
  Extract the discrete facts from the text below as a JSON array of strings.
  Text: "The Eiffel Tower is in Paris and was completed in 1889."
rubric:
  pass_threshold: 0.7
  rules:
    - json_valid: true
    - contains: ["Eiffel", "Paris"]
    - not_contains: ["Here are"]
    - min_length: 40
    - regex: "\\d{4}"
    # a custom checker — in code, or a `module.path:func` reference usable in YAML:
    # - python: "my_pkg.checks:is_valid"   # resolved lazily at eval time
metadata:
  temperature: 0.2
```

A **suite** is either a directory of task YAML files, or one file with a
`tasks:` list:

```yaml
suite: my-suite
tasks:
  - id: a
    user_prompt: "..."
    rubric: {rules: [{contains: "ok"}]}
  - id: b
    user_prompt: "..."
    rubric: {rules: [{regex: "Answer:\\s*\\d+"}]}
```

### Rubric rules

| Rule | Meaning |
|---|---|
| `contains: "x"` / `contains: [a, b]` | substring present (any-of for lists) |
| `not_contains: "x"` | substring absent (any-of for lists) |
| `json_valid: true` | output parses as JSON (tolerates markdown fences / surrounding prose). `strict: true` (default) requires an object/array and rejects scalars; `strict: false` accepts any valid JSON value |
| `json_fields: [a, b]` | JSON object contains keys `a`, `b` |
| `min_length: N` / `max_length: N` | character bounds |
| `regex: "..."` | pattern must match somewhere in the output |
| `python: fn` / `python: "pkg.mod:fn"` | custom checker `fn(output: str)` — a *truthy* return passes. Accepts an in-memory callable or a `module.path:func` reference (usable in YAML) |

Each rule gets a `weight` (default `1.0`). Score = weighted fraction of passing
rules, in `[0, 1]`. The task **passes** when score ≥ `pass_threshold` (default `0.7`).

## Python API

```python
from agentbench import Benchmark, LLMClient, MockClient, to_json, to_markdown, compare

bench = Benchmark.load_path("benchmarks/basic")

# against a real endpoint
client = LLMClient(
    endpoint="http://localhost:11434/v1",
    model="qwen2.5:14b",
    api_key="ollama",
    temperature=0.3,
    max_tokens=512,
    retries=2,
)
result = bench.run(client, concurrency=2)

# or offline
result = bench.run(MockClient(response='{"ok": true}'))

print(result.summary())          # mean score, pass rate, p95 latency, token totals
with open("out.json", "w") as fh: fh.write(to_json(result))
with open("out.md",   "w") as fh: fh.write(to_markdown(result))
```

Filter subsets before running:

```python
bench.filter(tags=["json"]).run(client)
bench.filter(task_ids=["extract-facts"]).run(client)
```

## Reports

`to_markdown()` and the CLI render a **scorecard** (score bars, pass/fail
badges, and summed token totals), a per-task table, and a rule-by-rule
breakdown with ✓/✗ marks. `compare()` adds a side-by-side table with ▲/▼
per-task deltas against a second run.

## Examples

| Script | What it does |
|---|---|
| `examples/quickstart.py` | Run the basic suite against Ollama, print a summary, write a markdown report |
| `examples/compare_models.py` | Run the research suite against two models/configs and produce a side-by-side report |
| `examples/agentflow_gate.py` | CI gate: run a suite against an agentflow pipeline `run(task) -> str` and exit non-zero on regression |
| `examples/agentflow_gate_pipeline.py` | Standalone stub pipeline so the gate example runs without a real agentflow install |

## agentflow integration (CI gate)

```bash
# your pipeline module must expose: def run(task: str) -> str
AGENTBENCH_GATE_MODULE=agentflow_gate_pipeline \
python examples/agentflow_gate.py --min-score 0.7 --min-pass-rate 0.8
```

This is the "build (agentflow) → verify (agentbench) → ship" pattern.

## Non-goals (v1)

- No distributed execution, no vector/semantic scoring, no web dashboard.
- No LLM-as-judge (rubric rules only — LLM-judge is a v2 candidate).
- No MLflow/W&B — local JSON + markdown is the boundary.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

The full test suite (including the shipped 15-task suite validation and a full
offline run) is deterministic and runs with no network access.

## Relationship to agentflow

| | agentflow | agentbench |
|---|---|---|
| Role | Runtime orchestration | Quality assurance / benchmarking |
| Question | "How do I run agents?" | "How well do they perform?" |
| Output | Results + governance + audit trail | Scores + comparison reports |

## License

MIT — see [LICENSE](LICENSE).
