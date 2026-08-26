"""agentbench CLI.

Commands:
    validate  check YAML tasks + rubric syntax (dry-run)
    run       run a suite against an endpoint (or --mock for offline runs)
    compare   compare two result files side-by-side
"""
from __future__ import annotations

import os
import sys
from typing import List, Optional

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .benchmark import Benchmark, BenchmarkResult
from .compare import _load_result
from .report import to_json, to_markdown
from .runner import LLMClient, MockClient
from .task import TaskDef, TaskError

app = typer.Typer(
    name="agentbench",
    help="Local-first benchmarking & eval harness for agentic AI pipelines.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
err_console = Console(stderr=True)


def _text_bar(score: float, width: int = 10) -> str:
    """A block score bar for terminal output, e.g. ``███░░``."""
    filled = int(round(max(0.0, min(1.0, float(score))) * width))
    return "█" * filled + "░" * (width - filled)


def _score_style(score: float, errored: bool = False) -> str:
    if errored:
        return "dim yellow"
    if score >= 0.8:
        return "bold green"
    if score >= 0.5:
        return "yellow"
    return "red"


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"agentbench {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", "-V", callback=_version_callback, help="Print version and exit."
    ),
) -> None:
    """agentbench — benchmark and eval harness for agentic AI pipelines."""


@app.command()
def validate(
    path: str = typer.Argument(..., help="YAML task file or directory of task files."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="List each rule in each task."),
) -> None:
    """Dry-run: check that task YAML and rubrics parse and are well-formed."""
    try:
        bench = Benchmark.load_path(path)
    except TaskError as exc:
        err_console.print(f"[red]\u274c invalid:[/red] {exc}")
        raise typer.Exit(1)
    if verbose:
        for task in bench.tasks:
            rules = ", ".join(r.name for r in task.rubric.rules) or "\u2014"
            console.print(f"  \u2022 [bold]{task.id}[/bold] \u2014 {rules}")
    console.print(
        f"[green]\u2705 valid:[/green] [bold]{len(bench.tasks)}[/bold] task(s) in [italic]{bench.name}[/italic] "
        f"\u00b7 {sum(len(t.rubric.rules) for t in bench.tasks)} rubric rule(s)"
    )


def _client_from_args(
    endpoint: Optional[str],
    model: Optional[str],
    api_key: Optional[str],
    timeout: float,
    retries: int,
    temperature: Optional[float],
    max_tokens: Optional[int],
    mock: bool,
) -> LLMClient | MockClient:
    if mock:
        return MockClient()
    if not endpoint:
        err_console.print("[red]missing --endpoint (or use --mock for offline runs)[/red]")
        raise typer.Exit(1)
    if not model:
        err_console.print("[red]missing --model (or use --mock for offline runs)[/red]")
        raise typer.Exit(1)
    api_key = api_key or os.environ.get("AGENTBENCH_API_KEY") or os.environ.get("AGENTFLOW_API_KEY")
    return LLMClient(
        endpoint=endpoint,
        model=model,
        api_key=api_key,
        timeout=timeout,
        retries=retries,
        temperature=temperature,
        max_tokens=max_tokens,
    )


@app.command()
def run(
    suite: str = typer.Argument(..., help="Suite path: directory of YAML tasks or a single YAML file."),
    endpoint: Optional[str] = typer.Option(None, "--endpoint", "-e", help="OpenAI-compatible base URL."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model name to evaluate."),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="API key (defaults to $AGENTBENCH_API_KEY / $AGENTFLOW_API_KEY)."),
    concurrency: int = typer.Option(1, "--concurrency", "-c", min=1, help="Parallel tasks."),
    tags: Optional[str] = typer.Option(None, "--tags", help="Comma-separated tag filter (e.g. 'research,hard')."),
    task_ids: Optional[str] = typer.Option(None, "--tasks", help="Comma-separated task id filter."),
    temperature: Optional[float] = typer.Option(None, "--temperature", help="Sampling temperature override."),
    max_tokens: Optional[int] = typer.Option(None, "--max-tokens", help="Max completion tokens override."),
    timeout: float = typer.Option(120.0, "--timeout", help="Per-request timeout (seconds)."),
    retries: int = typer.Option(2, "--retries", help="Retries on transient (5xx) failures."),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Write JSON result to this file."),
    markdown: Optional[str] = typer.Option(None, "--markdown", help="Write markdown report to this file."),
    json_out: bool = typer.Option(False, "--json", help="Print the full JSON result to stdout."),
    mock: bool = typer.Option(False, "--mock", help="Run offline against the built-in mock endpoint."),
) -> None:
    """Run a suite of benchmark tasks and print a score summary."""
    try:
        bench = Benchmark.load_path(suite)
    except TaskError as exc:
        err_console.print(f"[red]\u274c cannot load suite:[/red] {exc}")
        raise typer.Exit(1)

    tag_list = [t.strip() for t in tags.split(",")] if tags else None
    id_list = [t.strip() for t in task_ids.split(",")] if task_ids else None
    if tag_list or id_list:
        try:
            bench = bench.filter(tags=tag_list, task_ids=id_list)
        except TaskError as exc:
            err_console.print(f"[red]\u274c filter:[/red] {exc}")
            raise typer.Exit(1)

    client = _client_from_args(endpoint, model, api_key, timeout, retries, temperature, max_tokens, mock)
    result = bench.run(client, concurrency=concurrency)

    table = Table(
        title=f"\U0001f3c1 agentbench \u2014 {result.model} @ {result.endpoint}",
        box=box.ROUNDED,
        title_justify="center",
        show_lines=True,
    )
    table.add_column("Task", overflow="fold", min_width=18)
    table.add_column("Score", justify="right")
    table.add_column("Bar")
    table.add_column("Pass", justify="center")
    table.add_column("Latency", justify="right")
    table.add_column("Error", style="dim", overflow="fold", max_width=42)
    for t in result.tasks:
        errored = t.error is not None
        style = _score_style(t.score, errored=errored)
        mark = "[yellow]\u26a0[/yellow]" if errored else ("[green]\u2705[/green]" if t.passed else "[red]\u274c[/red]")
        table.add_row(
            t.task_name,
            f"[{style}]{t.score:.2f}[/]",
            f"[{style}]{_text_bar(t.score)}[/]",
            mark,
            f"{t.latency_s:.2f}s",
            t.error or "",
        )
    console.print(table)

    s = result.summary()
    tok = s.get("tokens", {})
    verdict = "bold green" if s["pass_rate"] >= 0.8 else ("yellow" if s["pass_rate"] >= 0.5 else "red")
    panel_body = (
        f"Mean score  [{verdict}]{s['mean_score']:.2f}[/]    "
        f"Pass rate  [{verdict}]{s['pass_rate']:.0%}[/]    "
        f"p95 latency  {result.p95_latency:.2f}s"
    )
    panel_meta = (
        f"Tasks {s['task_count']}  (\u2705 {s['passed']} \u00b7 \u274c {s['failed']} \u00b7 \u26a0 {s['errors']})   "
        f"tokens {tok.get('total', 0):,}   ({result.model} @ {result.endpoint})"
    )
    console.print(Panel(f"{panel_body}\n[dim]{panel_meta}[/]", title="\U0001f4ca Scorecard", box=box.ROUNDED, border_style="cyan"))

    if json_out:
        print(to_json(result))
    if output:
        with open(output, "w", encoding="utf-8") as fh:
            fh.write(to_json(result) + "\n")
        console.print(f"wrote [cyan]{output}[/cyan]")
    if markdown:
        with open(markdown, "w", encoding="utf-8") as fh:
            fh.write(to_markdown(result) + "\n")
        console.print(f"wrote [cyan]{markdown}[/cyan]")


@app.command()
def compare(
    a: str = typer.Argument(..., help="First result (JSON file path or JSON string)."),
    b: str = typer.Argument(..., help="Second result (JSON file path or JSON string)."),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Write markdown report to this file."),
) -> None:
    """Compare two benchmark result files side-by-side (markdown table)."""
    try:
        ra, rb = _load_result(a), _load_result(b)
    except Exception as exc:  # noqa: BLE001
        err_console.print(f"[red]\u274c cannot load result:[/red] {exc}")
        raise typer.Exit(1)
    report = to_markdown(ra, other=rb, title=f"agentbench compare — {ra.model} vs {rb.model}")
    if output:
        with open(output, "w", encoding="utf-8") as fh:
            fh.write(report + "\n")
        console.print(f"wrote [cyan]{output}[/cyan]")
    else:
        print(report)


if __name__ == "__main__":
    app()
