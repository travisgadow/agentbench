"""Endpoint client: talk to any OpenAI-compatible chat API, with retry + timeout.

``LLMClient`` uses ``requests``. ``MockClient`` is an offline stand-in that
lets tests (and offline demos) run the full pipeline deterministically.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import requests


class EndpointError(RuntimeError):
    """Raised when the endpoint call ultimately fails (after retries)."""


@dataclass
class LLMResponse:
    """One completed chat call."""

    content: str
    latency_s: float
    usage: Dict[str, Any] = field(default_factory=dict)
    model: Optional[str] = None


def _endpoint_url(endpoint: str, model: str) -> str:
    """Build the OpenAI-compatible ``/chat/completions`` URL from a base endpoint.

    Accepts the full path, a ``.../v1`` base, or a bare host, and returns the
    canonical ``.../v1/chat/completions`` form exposed by Ollama, vLLM and most
    OpenAI-compatible servers. (``model`` is kept for a stable signature.)
    """
    base = endpoint.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


class LLMClient:
    """Minimal OpenAI-compatible chat client (no SDK, no streaming)."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        retries: int = 2,
        backoff: float = 0.5,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.retries = max(0, retries)
        self.backoff = backoff
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(self, messages: List[Dict[str, str]], **overrides: Any) -> LLMResponse:
        """Run one chat completion with retry/backoff on transient errors.

        ``overrides`` may include per-task ``temperature`` / ``max_tokens``
        from the task's metadata.
        """
        payload: Dict[str, Any] = {"model": self.model, "messages": messages}
        temperature = overrides.get("temperature", self.temperature)
        max_tokens = overrides.get("max_tokens", self.max_tokens)
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = _endpoint_url(self.endpoint, self.model)
        attempts = self.retries + 1
        last_error: Optional[BaseException] = None

        for attempt in range(attempts):
            started = time.perf_counter()
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            except requests.RequestException as exc:
                last_error = exc
                if attempt < attempts - 1:
                    time.sleep(self.backoff * (2 ** attempt))
                    continue
                raise EndpointError(f"connection failed after {attempts} attempts: {exc}") from exc

            latency = time.perf_counter() - started

            if resp.status_code == 200:
                try:
                    data = resp.json()
                except ValueError as exc:
                    raise EndpointError("endpoint returned non-JSON body") from exc
                content = ""
                choices = data.get("choices") or []
                if choices and isinstance(choices[0], dict):
                    message = choices[0].get("message") or {}
                    content = str(message.get("content") or "")
                return LLMResponse(
                    content=content,
                    latency_s=latency,
                    usage=dict(data.get("usage") or {}),
                    model=data.get("model", self.model),
                )

            if 400 <= resp.status_code < 500:
                # client errors won't fix themselves; fail fast
                raise EndpointError(f"HTTP {resp.status_code}: {resp.text[:300]}")

            last_error = EndpointError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            if attempt < attempts - 1:
                time.sleep(self.backoff * (2 ** attempt))
                continue

        assert last_error is not None
        raise last_error


class MockClient:
    """Offline stand-in for ``LLMClient`` with the same ``chat`` interface.

    Behaviours available per message:
    * default: returns a fixed canned response,
    * if the user prompt contains "FAIL_400" -> raises HTTP 400,
    * if the user prompt contains "FAIL_500" -> raises HTTP 500,
    * if the user prompt contains "FAIL_TIMEOUT" -> raises a connection timeout.
    """

    def __init__(
        self,
        endpoint: str = "mock://offline",
        model: str = "mock-1",
        response: str = "This is a mock response.",
        latency_s: float = 0.0,
        usage: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.endpoint = endpoint
        self.model = model
        self.response = response
        self.latency_s = latency_s
        self.usage = usage or {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}

    def chat(self, messages: List[Dict[str, str]], **overrides: Any) -> LLMResponse:
        user = ""
        for m in messages:
            if m.get("role") == "user":
                user = str(m.get("content", ""))
        if "FAIL_TIMEOUT" in user:
            raise EndpointError("connection failed after 1 attempts: simulated timeout")
        if "FAIL_400" in user:
            raise EndpointError("HTTP 400: simulated client error")
        if "FAIL_500" in user:
            raise EndpointError("HTTP 500: simulated server error")
        return LLMResponse(
            content=self.response,
            latency_s=self.latency_s,
            usage=dict(self.usage),
            model=self.model,
        )
