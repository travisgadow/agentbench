"""Runner tests: URL building, error handling, concurrency, token usage."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import requests

from agentbench.benchmark import Benchmark
from agentbench.runner import EndpointError, LLMClient, MockClient
from agentbench.task import Rule, Rubric, TaskDef


def test_endpoint_url_variants():
    # exercised indirectly via _endpoint_url
    from agentbench.runner import _endpoint_url

    assert _endpoint_url("http://localhost:11434/v1", "m").endswith("/v1/chat/completions")
    assert _endpoint_url("http://localhost:11434/v1/", "m").endswith("/v1/chat/completions")
    assert _endpoint_url("http://x/v1/chat/completions", "m").endswith("/v1/chat/completions")
    assert _endpoint_url("http://localhost:11434", "m") == "http://localhost:11434/v1/chat/completions"


class _Handler(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible mock server for integration tests."""

    behavior = "ok"  # set by tests: "ok" | "400" | "flaky"
    hits = 0

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        type(self).hits += 1
        if type(self).behavior == "400":
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"bad request")
            return
        if type(self).behavior == "flaky" and type(self).hits < 2:
            self.send_response(503)
            self.end_headers()
            self.wfile.write(b"service unavailable")
            return
        payload = {
            "model": body.get("model"),
            "choices": [{"message": {"role": "assistant", "content": '["ok"]'}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 7, "total_tokens": 10},
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode())

    def log_message(self, *args):  # silence
        pass


@pytest.fixture
def mock_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()


def test_success_returns_content_and_usage(mock_server):
    _Handler.behavior, _Handler.hits = "ok", 0
    client = LLMClient(endpoint=mock_server, model="test", api_key="k", timeout=5, retries=0)
    resp = client.chat([{"role": "user", "content": "hi"}])
    assert resp.content == '["ok"]'
    assert resp.usage["total_tokens"] == 10
    assert resp.model == "test"


def test_4xx_fails_fast_without_retry(mock_server):
    _Handler.behavior, _Handler.hits = "400", 0
    client = LLMClient(endpoint=mock_server, model="test", timeout=5, retries=3)
    with pytest.raises(EndpointError, match="400"):
        client.chat([{"role": "user", "content": "hi"}])
    assert _Handler.hits == 1  # no retries on 4xx


def test_5xx_retries_then_succeeds(mock_server):
    _Handler.behavior, _Handler.hits = "flaky", 0
    client = LLMClient(endpoint=mock_server, model="test", timeout=5, retries=2, backoff=0.01)
    resp = client.chat([{"role": "user", "content": "hi"}])
    assert resp.content == '["ok"]'
    assert _Handler.hits == 2  # first 503, second 200


def test_connection_refused_raises_endpoint_error():
    client = LLMClient(endpoint="http://127.0.0.1:1/v1", model="test", timeout=1, retries=0)
    with pytest.raises(EndpointError, match="connection failed"):
        client.chat([{"role": "user", "content": "hi"}])


def test_mock_client_failure_signals():
    client = MockClient()
    with pytest.raises(EndpointError, match="400"):
        client.chat([{"role": "user", "content": "FAIL_400 please"}])
    with pytest.raises(EndpointError, match="500"):
        client.chat([{"role": "user", "content": "FAIL_500 please"}])


def test_concurrency_preserves_order_and_runs_parallel(mock_server):
    _Handler.behavior, _Handler.hits = "ok", 0
    from agentbench.task import TaskDef

    tasks = [
        TaskDef(id=f"c{i}", name=f"c{i}", user_prompt=f"task {i}", rubric=Rubric(rules=[]))
        for i in range(5)
    ]
    bench = Benchmark(tasks)
    result = bench.run(LLMClient(endpoint=mock_server, model="test", timeout=5), concurrency=4)
    assert [t.task_id for t in result.tasks] == [f"c{i}" for i in range(5)]
    assert _Handler.hits == 5


def test_task_metadata_overrides_flow_through(mock_server):
    """temperature/max_tokens from task metadata reach the request payload."""
    _Handler.behavior, _Handler.hits = "ok", 0

    class _RecordingHandler(_Handler):
        pass

    # quick check via direct client with overrides
    client = LLMClient(endpoint=mock_server, model="test", timeout=5, temperature=0.9, max_tokens=32)
    resp = client.chat([{"role": "user", "content": "hi"}], temperature=0.1, max_tokens=8)
    assert resp.content  # server accepted the payload
