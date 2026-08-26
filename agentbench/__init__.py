"""agentbench — a local-first benchmarking & eval harness for agentic AI pipelines.

Project #2 in the AI agentic workflows portfolio. agentflow answers
"how do I run agents?"; agentbench answers "how well do they perform?".

Define benchmark tasks in YAML, run them against any OpenAI-compatible
endpoint (Ollama, vLLM, cloud), score outputs with composable weighted
rubrics, and produce side-by-side comparison reports. No cloud required.
"""
__version__ = "0.2.0"

from .task import Rule, Rubric, TaskDef, TaskError  # noqa: F401
from .scorer import RuleResult, score_rubric  # noqa: F401
from .runner import EndpointError, LLMClient, MockClient  # noqa: F401
from .benchmark import Benchmark, BenchmarkResult, TaskResult  # noqa: F401
from .report import to_json, to_markdown  # noqa: F401
from .compare import compare  # noqa: F401

__all__ = [
    "__version__",
    "Rule", "Rubric", "TaskDef", "TaskError",
    "RuleResult", "score_rubric",
    "EndpointError", "LLMClient", "MockClient",
    "Benchmark", "BenchmarkResult", "TaskResult",
    "to_json", "to_markdown",
    "compare",
]
