# Changelog

All notable changes to agentbench are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/), and the
project adheres to [Semantic Versioning](https://semver.org/).

## [0.2.0] — 2026-08-26

### Added
- **Token-usage aggregation:** `BenchmarkResult.summary()` now includes a
  `tokens` block (`prompt` / `completion` / `total`) summed across all tasks,
  and it surfaces in the JSON, markdown, and CLI scorecard.
- **`python` rules from YAML:** a `python` rule now accepts a `module.path:func`
  string reference (e.g. `python: "my_pkg.checks:is_valid"`), resolved lazily at
  eval time. In-memory callables still work for programmatic construction. This
  makes the "custom checker" rule usable in task files, not just in code.
- **Prettier reports:** the markdown report is now a full scorecard — summary
  table with score bars, a per-task table with score bars + pass badges, a
  rule-by-rule breakdown with ✓/✗ marks, and a side-by-side comparison with
  ▲/▼ deltas. The CLI `run` table shows color-graded scores, score bars, and a
  bordered Scorecard panel.

### Fixed
- `contains` / `not_contains` with an empty string (or empty list, e.g.
  `contains: ["", "x"]`) no longer passes vacuously; empty/blank needles and
  empty lists are rejected at parse time with a clear `TaskError`.
- `json_fields: []` is now rejected at parse time instead of passing on every
  output.
- `python` rule checkers: a **truthy** return value (non-bool such as a dict or
  list) now counts as a pass; previously only the literal `True` passed.
- `json_valid` no longer reports a **false negative** when the output contains
  more than one JSON span or JSON surrounded by prose; it now scans for the
  first balanced, valid JSON object/array (respecting string literals/escapes).
- `Benchmark.run()` no longer swallows *all* client exceptions: transport-level
  errors (`EndpointError`, `requests.RequestException`) are still recorded per
  task, but genuine programmer errors (e.g. `TypeError`) propagate instead of
  silently zero-scoring the whole suite.
- Removed a dead `import requests as _rq` from `MockClient.chat`.
- Hardened `_endpoint_url`: canonical `.../v1/chat/completions` building without
  the incorrect vLLM `/api/chat` special-case.
- Documented `json_valid` `strict` semantics: `strict: true` (default) requires a
  JSON object/array and rejects scalars; `strict: false` accepts any valid JSON
  value. Added tests.

### Changed
- `Rule.to_dict()` now round-trips `python` rule references (string refs are
  preserved instead of being collapsed to `<callable>`).
- Test suite expanded: 57 → 69 offline tests covering all of the above.

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
