# Changelog

All notable changes to agentbench are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/), and the
project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026-08-24

### Added
- Core primitives: `TaskDef`, `Rubric`, `Rule` with YAML loading and validation.
- Rubric rules: `contains`, `not_contains`, `json_valid`, `json_fields`,
  `min_length`, `max_length`, `regex`, `python` (programmatic), with per-rule
  weights and a `pass_threshold`.
- Weighted scoring in `[0, 1]` with pass/fail determination.
- `LLMClient`: OpenAI-compatible endpoint client with retry/backoff on 5xx and
  transient errors, fail-fast on 4xx, timeouts, per-task `temperature`/`max_tokens`
  overrides.
- `MockClient` for fully offline runs and deterministic tests.
- `Benchmark`: load suites from a directory, a single YAML file, or a `tasks:`
  list; filter by tag or task id; run with thread-pool concurrency.
- `BenchmarkResult`: per-task scores, rule results, latency, token usage,
  errors; aggregates (mean score, pass rate, p95 latency); `to_dict()`.
- Reports: `to_json()` and `to_markdown()` (with optional side-by-side comparison).
- `compare()`: side-by-side markdown report from two results (objects, dicts,
  JSON strings, or file paths).
- CLI: `agentbench validate`, `agentbench run` (`--mock`, `--concurrency`,
  `--tags`, `--tasks`, `--output`, `--markdown`, `--json`, `--temperature`,
  `--max-tokens`, `--retries`, `--timeout`), and `agentbench compare --output`.
- Shipped starter benchmark suites: `basic/` (5), `research/` (5), `tooluse/` (5).
- Examples: `quickstart.py`, `compare_models.py`, `agentflow_gate.py`
  (+ standalone `agentflow_gate_pipeline.py` stub).
- GitHub Actions CI (Python 3.10–3.13) running the offline test suite and
  `agentbench validate benchmarks/`.
- Test suite: task/rubric parsing & validation, every rubric rule, weighted
  scoring & thresholds, benchmark loading/filtering/aggregation, endpoint
  success/4xx/5xx-retry/timeout paths, concurrency & order preservation,
  compare (deltas, missing tasks, JSON/dict/file inputs), CLI (validate good/bad,
  run mock, compare fixtures), and an integration test that validates and runs
  the full shipped 15-task suite offline.

### Fixed
- Markdown report rendering now works on Python 3.10/3.11 (no backslashes
  inside f-string expressions).
