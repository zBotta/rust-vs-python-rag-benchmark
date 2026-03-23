"""Tests for the llm_client module — Task 8.

Sub-task 8.1 — Property 6: Prompt template correctness
Sub-task 8.2 — Property 7: Model name sourced from environment variable
Sub-task 8.3 — Property 8: Total tokens = prompt_tokens + completion_tokens
Sub-task 8.4 — Unit tests for LLM client

Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from python_pipeline.llm_client import LLMResponse, _parse_token_counts, build_prompt, generate

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

text_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Zs"), whitelist_characters=".,!?"),
    min_size=0,
    max_size=100,
)

query_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Zs"), whitelist_characters=".,!?"),
    min_size=1,
    max_size=200,
)

# ---------------------------------------------------------------------------
# Sub-task 8.1 — Property 6: Prompt template correctness
# Feature: rust-vs-python-rag-benchmark, Property 6: Constructed prompt exactly matches "Context:\n{chunks}\n\nQuestion: {query}\nAnswer:"
# ---------------------------------------------------------------------------

# Feature: rust-vs-python-rag-benchmark, Property 6: Constructed prompt exactly matches "Context:\n{chunks}\n\nQuestion: {query}\nAnswer:"


@given(
    chunks=st.lists(text_strategy, min_size=0, max_size=20),
    query=query_strategy,
)
@settings(max_examples=20)
def test_prompt_template_correctness(chunks: list[str], query: str) -> None:
    """Property 6: Constructed prompt exactly matches the template.

    # Feature: rust-vs-python-rag-benchmark, Property 6: Constructed prompt exactly matches "Context:\n{chunks}\n\nQuestion: {query}\nAnswer:"
    Validates: Requirements 5.3
    """
    prompt = build_prompt(chunks, query)
    expected = f"Context:\n{chr(10).join(chunks)}\n\nQuestion: {query}\nAnswer:"
    assert prompt == expected, f"Prompt mismatch:\n  got:      {prompt!r}\n  expected: {expected!r}"


# ---------------------------------------------------------------------------
# Sub-task 8.2 — Property 7: Model name sourced from environment variable
# Feature: rust-vs-python-rag-benchmark, Property 7: model field in every Ollama request body equals BENCHMARK_MODEL env var value
# ---------------------------------------------------------------------------

# Feature: rust-vs-python-rag-benchmark, Property 7: model field in every Ollama request body equals BENCHMARK_MODEL env var value


@given(
    model_name=st.text(
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="._:-"),
        min_size=1,
        max_size=50,
    ),
    chunks=st.lists(text_strategy, min_size=0, max_size=5),
    query=query_strategy,
)
@settings(max_examples=20)
def test_model_name_from_env(model_name: str, chunks: list[str], query: str) -> None:
    """Property 7: model field in every Ollama request body equals BENCHMARK_MODEL env var value.

    # Feature: rust-vs-python-rag-benchmark, Property 7: model field in every Ollama request body equals BENCHMARK_MODEL env var value
    Validates: Requirements 5.2
    """
    # Test the pure construction: the payload built inside generate() must use the env var value.
    # We verify this by capturing the json kwarg passed to client.stream().
    captured_bodies: list[dict] = []

    import httpx

    def fake_stream(method, url, json=None, **kwargs):
        captured_bodies.append(json)
        # Raise an httpx error so generate() treats it as a retryable failure
        raise httpx.RequestError("mock connection error", request=MagicMock())

    old_val = os.environ.get("BENCHMARK_MODEL")
    try:
        os.environ["BENCHMARK_MODEL"] = model_name

        with patch("python_pipeline.llm_client.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value.__enter__ = lambda s: mock_client
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.stream.side_effect = fake_stream

            with patch("python_pipeline.llm_client.time.sleep"):
                generate(query, chunks, llm_host="http://localhost:11434", max_retries=1)

        assert len(captured_bodies) >= 1, "Expected at least one request attempt"
        for body in captured_bodies:
            assert body["model"] == model_name, (
                f"model field {body['model']!r} != env var {model_name!r}"
            )
    finally:
        if old_val is None:
            os.environ.pop("BENCHMARK_MODEL", None)
        else:
            os.environ["BENCHMARK_MODEL"] = old_val


# ---------------------------------------------------------------------------
# Sub-task 8.3 — Property 8: Total tokens = prompt_tokens + completion_tokens
# Feature: rust-vs-python-rag-benchmark, Property 8: total_tokens = prompt_tokens + completion_tokens
# ---------------------------------------------------------------------------

# Feature: rust-vs-python-rag-benchmark, Property 8: total_tokens = prompt_tokens + completion_tokens


@given(
    prompt_tokens=st.integers(min_value=0, max_value=10_000),
    completion_tokens=st.integers(min_value=0, max_value=10_000),
)
@settings(max_examples=20)
def test_total_tokens_is_sum(prompt_tokens: int, completion_tokens: int) -> None:
    """Property 8: total_tokens = prompt_tokens + completion_tokens.

    # Feature: rust-vs-python-rag-benchmark, Property 8: total_tokens = prompt_tokens + completion_tokens
    Validates: Requirements 5.6
    """
    final_obj = {
        "model": "llama3.2:3b",
        "response": "",
        "done": True,
        "prompt_eval_count": prompt_tokens,
        "eval_count": completion_tokens,
    }
    total = _parse_token_counts(final_obj)
    assert total == prompt_tokens + completion_tokens, (
        f"total_tokens={total} != {prompt_tokens} + {completion_tokens}"
    )


# ---------------------------------------------------------------------------
# Sub-task 8.4 — Unit tests for LLM client
# Requirements: 5.1, 5.4, 5.5
# ---------------------------------------------------------------------------


def _make_streaming_lines(tokens: list[str], prompt_eval_count: int = 50, eval_count: int = 100) -> list[str]:
    """Build a list of NDJSON lines simulating Ollama streaming response."""
    lines = []
    for i, token in enumerate(tokens):
        is_last = i == len(tokens) - 1
        obj: dict = {"model": "llama3.2:3b", "response": token, "done": is_last}
        if is_last:
            obj["prompt_eval_count"] = prompt_eval_count
            obj["eval_count"] = eval_count
        lines.append(json.dumps(obj))
    return lines


class _MockStreamResponse:
    """Minimal mock for httpx streaming response context manager."""

    def __init__(self, lines: list[str], status_code: int = 200):
        self._lines = lines
        self.status_code = status_code

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=MagicMock(),
                response=MagicMock(status_code=self.status_code),
            )

    def iter_lines(self):
        return iter(self._lines)


def test_request_body_construction():
    """Test correct HTTP request body construction (mocked HTTP server).

    Requirements: 5.1
    """
    captured: list[dict] = []
    tokens = ["Hello", " world", ""]

    def fake_stream(method, url, json=None, **kwargs):
        captured.append(json)
        return _MockStreamResponse(_make_streaming_lines(tokens))

    old_val = os.environ.get("BENCHMARK_MODEL")
    try:
        os.environ["BENCHMARK_MODEL"] = "test-model"

        with patch("python_pipeline.llm_client.httpx.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value.__enter__ = lambda s: mock_client
            mock_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.stream.side_effect = fake_stream

            result = generate("What is AI?", ["chunk1", "chunk2"], llm_host="http://localhost:11434")

        assert len(captured) == 1
        body = captured[0]
        assert body["model"] == "test-model"
        assert body["stream"] is True
        assert body["options"]["num_predict"] == 256
        expected_prompt = build_prompt(["chunk1", "chunk2"], "What is AI?")
        assert body["prompt"] == expected_prompt
        assert not result.failed
    finally:
        if old_val is None:
            os.environ.pop("BENCHMARK_MODEL", None)
        else:
            os.environ["BENCHMARK_MODEL"] = old_val


def test_exactly_3_retries_on_persistent_error():
    """Test exactly 3 retries on persistent HTTP error (mocked server always errors).

    Requirements: 5.5
    """
    call_count = 0

    def always_error(method, url, json=None, **kwargs):
        nonlocal call_count
        call_count += 1
        import httpx
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: mock_resp
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500 Internal Server Error",
            request=MagicMock(),
            response=MagicMock(status_code=500),
        )
        return mock_resp

    with patch("python_pipeline.llm_client.httpx.Client") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value.__enter__ = lambda s: mock_client
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.stream.side_effect = always_error

        with patch("python_pipeline.llm_client.time.sleep") as mock_sleep:
            result = generate("query", [], llm_host="http://localhost:11434", max_retries=3)

    assert call_count == 3, f"Expected 3 attempts, got {call_count}"
    assert mock_sleep.call_count == 2, f"Expected 2 sleeps (between retries), got {mock_sleep.call_count}"
    assert result.failed
    assert result.failure_reason is not None
    assert "3 retries" in result.failure_reason


def test_ttft_and_generation_time_recorded():
    """Test TTFT and generation time recorded from mocked streaming response.

    Requirements: 5.4
    """
    tokens = ["First", " token", " here", ""]

    def fake_stream(method, url, json=None, **kwargs):
        return _MockStreamResponse(_make_streaming_lines(tokens, prompt_eval_count=30, eval_count=70))

    with patch("python_pipeline.llm_client.httpx.Client") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value.__enter__ = lambda s: mock_client
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.stream.side_effect = fake_stream

        result = generate("test", ["ctx"], llm_host="http://localhost:11434")

    assert not result.failed
    assert result.ttft_ms >= 0.0, "TTFT must be non-negative"
    assert result.generation_ms >= 0.0, "generation_ms must be non-negative"
    assert result.generation_ms >= result.ttft_ms, "generation_ms must be >= ttft_ms"
    assert result.total_tokens == 100, f"Expected 100 tokens (30+70), got {result.total_tokens}"
    assert result.text == "First token here"


def test_build_prompt_basic():
    """Test build_prompt with multiple chunks."""
    chunks = ["chunk one", "chunk two"]
    query = "What is this?"
    prompt = build_prompt(chunks, query)
    assert prompt == "Context:\nchunk one\nchunk two\n\nQuestion: What is this?\nAnswer:"


def test_build_prompt_empty_chunks():
    """Test build_prompt with empty chunks list."""
    prompt = build_prompt([], "Any question?")
    assert prompt == "Context:\n\n\nQuestion: Any question?\nAnswer:"


def test_parse_token_counts_missing_fields():
    """Test _parse_token_counts handles missing fields gracefully."""
    assert _parse_token_counts({}) == 0
    assert _parse_token_counts({"prompt_eval_count": 10}) == 10
    assert _parse_token_counts({"eval_count": 20}) == 20
    assert _parse_token_counts({"prompt_eval_count": 10, "eval_count": 20}) == 30

