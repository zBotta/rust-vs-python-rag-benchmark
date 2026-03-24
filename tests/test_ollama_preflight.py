"""Bug condition exploration tests for the Ollama preflight check.

Task 1 — Property 1: Bug Condition — Silent Failure When Ollama Unreachable

CRITICAL: This test MUST FAIL on unfixed code — failure confirms the bug exists.
The test encodes the EXPECTED behavior (sys.exit called before any query is dispatched).
It will PASS once the preflight fix is implemented.

Validates: Requirements 1.1, 1.2
"""
from __future__ import annotations

import json
import sys
import tempfile
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

# ---------------------------------------------------------------------------
# Pre-stub heavy optional dependencies so pipeline.py can be imported without
# torch / transformers / hnswlib being installed in the test environment.
# ---------------------------------------------------------------------------
import types

# Stub torch before any pipeline import
if "torch" not in sys.modules:
    sys.modules["torch"] = types.ModuleType("torch")

# Stub transformers
if "transformers" not in sys.modules:
    _transformers = types.ModuleType("transformers")
    _transformers.AutoTokenizer = MagicMock()
    _transformers.AutoModel = MagicMock()
    sys.modules["transformers"] = _transformers

# Stub llama_cpp
if "llama_cpp" not in sys.modules:
    sys.modules["llama_cpp"] = types.ModuleType("llama_cpp")

# Now import the pipeline (stubs are in place)
from python_pipeline.pipeline import run_pipeline  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers — build a minimal config TOML and query set on disk
# ---------------------------------------------------------------------------

def _write_minimal_config(tmp_dir: Path, llm_backend: str = "ollama_http") -> Path:
    """Write a minimal benchmark_config.toml to tmp_dir and return its path."""
    query_set_path = tmp_dir / "queries.json"
    query_set_path.write_text(
        json.dumps([{"id": 1, "question": "What is RAG?"}]),
        encoding="utf-8",
    )

    # Use forward slashes in TOML strings to avoid backslash escape issues on Windows
    query_set_posix = query_set_path.as_posix()
    output_posix = (tmp_dir / "output").as_posix()

    config_text = textwrap.dedent(f"""\
        dataset_name = "wikimedia/wikipedia"
        dataset_subset = "20231101.simple"
        num_documents = 1
        chunk_size = 512
        chunk_overlap = 64
        embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        top_k = 3
        llm_model = "llama3.2:1b"
        llm_host = "http://127.0.0.1:19999"
        query_set_path = "{query_set_posix}"
        output_dir = "{output_posix}"
        llm_backend = "{llm_backend}"
    """)

    config_path = tmp_dir / "benchmark_config.toml"
    config_path.write_text(config_text, encoding="utf-8")
    return config_path


# ---------------------------------------------------------------------------
# Task 1 — Bug Condition Exploration Test
# Property 1: sys.exit IS called before any query reaches llm_client.generate
#
# **Validates: Requirements 1.1, 1.2**
#
# EXPECTED OUTCOME on UNFIXED code: FAIL
#   run_pipeline completes without calling sys.exit even when Ollama is
#   unreachable — the missing preflight gate is confirmed.
# ---------------------------------------------------------------------------

def test_bug_condition_preflight_exits_before_queries_when_ollama_unreachable():
    """Property 1: Bug Condition — Silent Failure When Ollama Unreachable.

    When llm_backend="ollama_http" and the Ollama endpoint is unreachable,
    run_pipeline SHALL call sys.exit(1) before dispatching any query to the LLM.

    On UNFIXED code this test FAILS because:
      - run_pipeline has no preflight check
      - it proceeds to dispatch queries, each of which fails silently
      - sys.exit is never called
      - llm_client.generate IS called (up to max_retries per query)

    **Validates: Requirements 1.1, 1.2**
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        config_path = _write_minimal_config(tmp_dir, llm_backend="ollama_http")

        # Mock dataset_loader so we don't need real data on disk
        fake_doc = "Wikipedia article about RAG."
        mock_load = MagicMock(return_value=[fake_doc])

        # Mock chunker so we don't need real chunking
        mock_chunk = MagicMock(return_value=["chunk about RAG"])

        # Mock embedder so we don't need torch/transformers
        mock_embed = MagicMock(return_value=[[0.1] * 384])

        # Mock VectorStore so we don't need hnswlib
        mock_vs_instance = MagicMock()
        mock_vs_instance.build_index = MagicMock()
        mock_vs_cls = MagicMock(return_value=mock_vs_instance)

        # Mock Retriever
        mock_retriever_instance = MagicMock()
        mock_retriever_instance.retrieve = MagicMock(return_value=["chunk about RAG"])
        mock_retriever_cls = MagicMock(return_value=mock_retriever_instance)

        # Mock httpx.get (used by the preflight check) to raise ConnectError
        def mock_httpx_get(*args, **kwargs):
            raise httpx.ConnectError("Connection refused")

        # Track calls to llm_client.generate
        generate_call_count = {"n": 0}

        def counting_generate(*args, **kwargs):
            generate_call_count["n"] += 1
            # Return a failed response so the pipeline records a failure
            from python_pipeline.llm_client import LLMResponse
            return LLMResponse(
                text="",
                total_tokens=0,
                ttft_ms=0.0,
                generation_ms=0.0,
                failed=True,
                failure_reason="Connection refused",
            )

        with (
            patch("python_pipeline.pipeline.dataset_loader.load_documents", mock_load),
            patch("python_pipeline.pipeline.chunker.chunk_documents", mock_chunk),
            patch("python_pipeline.pipeline.embedder.embed_chunks", mock_embed),
            patch("python_pipeline.pipeline.VectorStore", mock_vs_cls),
            patch("python_pipeline.pipeline.Retriever", mock_retriever_cls),
            patch("python_pipeline.pipeline.llm_client.generate", counting_generate),
            patch("httpx.get", side_effect=mock_httpx_get),
            pytest.raises(SystemExit) as exc_info,
        ):
            run_pipeline(str(config_path))

        # --- Assertions (encode EXPECTED behavior) ---

        # 1. sys.exit MUST have been called (preflight gate)
        assert exc_info.value.code == 1, (
            f"Expected sys.exit(1) but got exit code: {exc_info.value.code!r}. "
            "BUG CONFIRMED: run_pipeline did not call sys.exit when Ollama was unreachable."
        )

        # 2. No queries must have reached llm_client.generate
        assert generate_call_count["n"] == 0, (
            f"Expected 0 calls to llm_client.generate but got {generate_call_count['n']}. "
            "BUG CONFIRMED: queries were dispatched even though Ollama was unreachable."
        )


# ---------------------------------------------------------------------------
# Task 2 — Preservation Property Tests (BEFORE implementing fix)
# Property 2: Preservation — Happy-Path Metrics and llama_cpp Path Unchanged
#
# These tests MUST PASS on UNFIXED code — they confirm baseline behavior to preserve.
#
# **Validates: Requirements 3.1, 3.2, 3.3, 3.4**
# ---------------------------------------------------------------------------

from hypothesis import given, settings, HealthCheck
import hypothesis.strategies as st


# ---------------------------------------------------------------------------
# Strategy: generate a list of successful LLMResponse objects
# ---------------------------------------------------------------------------

@st.composite
def successful_llm_responses(draw):
    """Generate a list of 1–5 successful LLMResponse objects with realistic metric values."""
    from python_pipeline.llm_client import LLMResponse

    n = draw(st.integers(min_value=1, max_value=5))
    responses = []
    for _ in range(n):
        responses.append(
            LLMResponse(
                text=draw(st.text(min_size=1, max_size=50)),
                total_tokens=draw(st.integers(min_value=1, max_value=512)),
                ttft_ms=draw(st.floats(min_value=0.1, max_value=5000.0, allow_nan=False, allow_infinity=False)),
                generation_ms=draw(st.floats(min_value=0.1, max_value=10000.0, allow_nan=False, allow_infinity=False)),
                failed=False,
                failure_reason=None,
            )
        )
    return responses


@given(mock_responses=successful_llm_responses())
@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
def test_preservation_metrics_match_mock_responses(mock_responses):
    """Property 2: Preservation — QueryMetrics fields match mock LLMResponse values.

    For all successful mock LLMResponse objects, the QueryMetrics recorded by the
    pipeline must have ttft_ms, generation_ms, and total_tokens equal to the values
    in the mock responses.

    This test MUST PASS on UNFIXED code — it confirms the baseline behavior to preserve.

    **Validates: Requirements 3.1, 3.3**
    """
    import json
    import tempfile
    from pathlib import Path
    from unittest.mock import MagicMock, patch

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        # Build a query set with one entry per mock response
        queries = [{"id": i + 1, "question": f"Question {i + 1}"} for i in range(len(mock_responses))]
        query_set_path = tmp_dir / "queries.json"
        query_set_path.write_text(json.dumps(queries), encoding="utf-8")

        import textwrap
        query_set_posix = query_set_path.as_posix()
        output_posix = (tmp_dir / "output").as_posix()

        config_text = textwrap.dedent(f"""\
            dataset_name = "wikimedia/wikipedia"
            dataset_subset = "20231101.simple"
            num_documents = 1
            chunk_size = 512
            chunk_overlap = 64
            embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
            top_k = 3
            llm_model = "llama3.2:1b"
            llm_host = "http://127.0.0.1:19999"
            query_set_path = "{query_set_posix}"
            output_dir = "{output_posix}"
            llm_backend = "ollama_http"
        """)
        config_path = tmp_dir / "benchmark_config.toml"
        config_path.write_text(config_text, encoding="utf-8")

        # Iterator that yields mock responses in order
        response_iter = iter(mock_responses)

        def mock_generate(*args, **kwargs):
            return next(response_iter)

        mock_load = MagicMock(return_value=["Wikipedia article about RAG."])
        mock_chunk = MagicMock(return_value=["chunk about RAG"])
        mock_embed = MagicMock(return_value=[[0.1] * 384])

        mock_vs_instance = MagicMock()
        mock_vs_instance.build_index = MagicMock()
        mock_vs_cls = MagicMock(return_value=mock_vs_instance)

        mock_retriever_instance = MagicMock()
        mock_retriever_instance.retrieve = MagicMock(return_value=["chunk about RAG"])
        mock_retriever_cls = MagicMock(return_value=mock_retriever_instance)

        # Mock httpx.get to simulate a reachable Ollama with the expected model
        mock_preflight_resp = MagicMock()
        mock_preflight_resp.raise_for_status = MagicMock()
        mock_preflight_resp.json = MagicMock(return_value={"models": [{"name": "llama3.2:1b"}]})

        with (
            patch("python_pipeline.pipeline.dataset_loader.load_documents", mock_load),
            patch("python_pipeline.pipeline.chunker.chunk_documents", mock_chunk),
            patch("python_pipeline.pipeline.embedder.embed_chunks", mock_embed),
            patch("python_pipeline.pipeline.VectorStore", mock_vs_cls),
            patch("python_pipeline.pipeline.Retriever", mock_retriever_cls),
            patch("python_pipeline.pipeline.llm_client.generate", mock_generate),
            patch("httpx.get", return_value=mock_preflight_resp),
        ):
            run_pipeline(str(config_path))

        # Read back the written JSONL and verify metrics match mock responses
        from python_pipeline.metrics_collector import deserialize_from_jsonl
        output_file = tmp_dir / "output" / "metrics_python_ollama_http.jsonl"
        assert output_file.exists(), "Expected JSONL output file to be written"

        pipeline_metrics = deserialize_from_jsonl(str(output_file))
        recorded = pipeline_metrics.queries

        assert len(recorded) == len(mock_responses), (
            f"Expected {len(mock_responses)} query records, got {len(recorded)}"
        )

        for i, (qm, resp) in enumerate(zip(recorded, mock_responses)):
            assert not qm.failed, f"Query {i} unexpectedly marked as failed: {qm.failure_reason}"
            assert qm.ttft_ms == resp.ttft_ms, (
                f"Query {i}: ttft_ms mismatch: recorded={qm.ttft_ms}, expected={resp.ttft_ms}"
            )
            assert qm.generation_ms == resp.generation_ms, (
                f"Query {i}: generation_ms mismatch: recorded={qm.generation_ms}, expected={resp.generation_ms}"
            )
            assert qm.total_tokens == resp.total_tokens, (
                f"Query {i}: total_tokens mismatch: recorded={qm.total_tokens}, expected={resp.total_tokens}"
            )


def test_preservation_llama_cpp_no_httpx_get_to_api_tags():
    """Property 2: Preservation — llama_cpp backend never calls httpx.get /api/tags.

    When llm_backend="llama_cpp", the pipeline must NOT make any HTTP call to
    /api/tags (the Ollama preflight endpoint). The llama_cpp path must remain
    completely unchanged.

    This test MUST PASS on UNFIXED code — it confirms the baseline behavior to preserve.

    **Validates: Requirements 3.2**
    """
    import json
    import tempfile
    import textwrap
    from pathlib import Path
    from unittest.mock import MagicMock, patch, call

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        queries = [{"id": 1, "question": "What is RAG?"}]
        query_set_path = tmp_dir / "queries.json"
        query_set_path.write_text(json.dumps(queries), encoding="utf-8")

        query_set_posix = query_set_path.as_posix()
        output_posix = (tmp_dir / "output").as_posix()

        config_text = textwrap.dedent(f"""\
            dataset_name = "wikimedia/wikipedia"
            dataset_subset = "20231101.simple"
            num_documents = 1
            chunk_size = 512
            chunk_overlap = 64
            embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
            top_k = 3
            llm_model = "llama3.2:1b"
            llm_host = "http://127.0.0.1:19999"
            query_set_path = "{query_set_posix}"
            output_dir = "{output_posix}"
            llm_backend = "llama_cpp"
            gguf_model_path = "/fake/model.gguf"
        """)
        config_path = tmp_dir / "benchmark_config.toml"
        config_path.write_text(config_text, encoding="utf-8")

        from python_pipeline.llm_client import LLMResponse
        mock_llama_response = LLMResponse(
            text="Answer",
            total_tokens=10,
            ttft_ms=50.0,
            generation_ms=200.0,
            failed=False,
            failure_reason=None,
        )

        mock_load = MagicMock(return_value=["Wikipedia article about RAG."])
        mock_chunk = MagicMock(return_value=["chunk about RAG"])
        mock_embed = MagicMock(return_value=[[0.1] * 384])

        mock_vs_instance = MagicMock()
        mock_vs_instance.build_index = MagicMock()
        mock_vs_cls = MagicMock(return_value=mock_vs_instance)

        mock_retriever_instance = MagicMock()
        mock_retriever_instance.retrieve = MagicMock(return_value=["chunk about RAG"])
        mock_retriever_cls = MagicMock(return_value=mock_retriever_instance)

        httpx_get_calls = []

        def tracking_httpx_get(url, *args, **kwargs):
            httpx_get_calls.append(url)
            raise AssertionError(f"httpx.get should not be called, but was called with: {url}")

        with (
            patch("python_pipeline.pipeline.dataset_loader.load_documents", mock_load),
            patch("python_pipeline.pipeline.chunker.chunk_documents", mock_chunk),
            patch("python_pipeline.pipeline.embedder.embed_chunks", mock_embed),
            patch("python_pipeline.pipeline.VectorStore", mock_vs_cls),
            patch("python_pipeline.pipeline.Retriever", mock_retriever_cls),
            patch("python_pipeline.pipeline.llm_client_llama_cpp.generate", return_value=mock_llama_response),
            patch("python_pipeline.pipeline._preflight_gguf"),
            patch("httpx.get", side_effect=tracking_httpx_get),
        ):
            run_pipeline(str(config_path))

        # Assert no httpx.get calls were made targeting /api/tags
        api_tags_calls = [url for url in httpx_get_calls if "/api/tags" in url]
        assert api_tags_calls == [], (
            f"Expected no httpx.get calls to /api/tags for llama_cpp backend, "
            f"but got: {api_tags_calls}"
        )


# ---------------------------------------------------------------------------
# Task 4 — Unit Tests for New Code
# Requirements: 2.1, 2.2, 2.3, 2.4
# ---------------------------------------------------------------------------

from python_pipeline.pipeline import _preflight_ollama


# --- Unit tests for _preflight_ollama ---

def test_preflight_reachable_correct_model_no_exit(capsys):
    """Req 2.1/2.2: Reachable endpoint with correct model → no sys.exit, prints OK.

    Validates: Requirements 2.1, 2.2
    """
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = MagicMock(return_value={"models": [{"name": "llama3.2:1b"}, {"name": "other:latest"}]})

    with patch("httpx.get", return_value=mock_resp):
        # Should not raise SystemExit
        _preflight_ollama("http://127.0.0.1:11434", "llama3.2:1b")

    captured = capsys.readouterr()
    assert "Preflight OK" in captured.out
    assert "llama3.2:1b" in captured.out


def test_preflight_unreachable_endpoint_exits(capsys):
    """Req 2.1: Unreachable endpoint → sys.exit(1) called, error message printed.

    Validates: Requirements 2.1
    """
    with patch("httpx.get", side_effect=httpx.ConnectError("Connection refused")):
        with pytest.raises(SystemExit) as exc_info:
            _preflight_ollama("http://127.0.0.1:19999", "llama3.2:1b")

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "ERROR" in captured.out
    assert "127.0.0.1:19999" in captured.out


def test_preflight_model_not_in_list_exits(capsys):
    """Req 2.2: Reachable endpoint but model not in list → sys.exit(1), lists available models.

    Validates: Requirements 2.2
    """
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = MagicMock(return_value={"models": [{"name": "mistral:latest"}, {"name": "phi3:mini"}]})

    with patch("httpx.get", return_value=mock_resp):
        with pytest.raises(SystemExit) as exc_info:
            _preflight_ollama("http://127.0.0.1:11434", "llama3.2:1b")

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "ERROR" in captured.out
    assert "llama3.2:1b" in captured.out
    # Available models should be listed
    assert "mistral:latest" in captured.out or "phi3:mini" in captured.out


# --- Unit tests for per-query status prints ---

def test_sequential_loop_prints_ok_and_fail_status(capsys):
    """Req 2.3: Sequential loop prints OK and FAIL status lines with correct query index.

    Validates: Requirements 2.3
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        # Two queries: first succeeds, second fails
        queries = [
            {"id": 1, "question": "What is RAG?"},
            {"id": 2, "question": "What is a vector store?"},
        ]
        query_set_path = tmp_dir / "queries.json"
        query_set_path.write_text(json.dumps(queries), encoding="utf-8")

        import textwrap
        config_text = textwrap.dedent(f"""\
            dataset_name = "wikimedia/wikipedia"
            dataset_subset = "20231101.simple"
            num_documents = 1
            chunk_size = 512
            chunk_overlap = 64
            embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
            top_k = 3
            llm_model = "llama3.2:1b"
            llm_host = "http://127.0.0.1:19999"
            query_set_path = "{query_set_path.as_posix()}"
            output_dir = "{(tmp_dir / 'output').as_posix()}"
            llm_backend = "ollama_http"
        """)
        config_path = tmp_dir / "benchmark_config.toml"
        config_path.write_text(config_text, encoding="utf-8")

        from python_pipeline.llm_client import LLMResponse

        responses = [
            LLMResponse(text="answer", total_tokens=10, ttft_ms=50.0, generation_ms=200.0, failed=False, failure_reason=None),
            LLMResponse(text="", total_tokens=0, ttft_ms=0.0, generation_ms=0.0, failed=True, failure_reason="timeout"),
        ]
        response_iter = iter(responses)

        mock_preflight_resp = MagicMock()
        mock_preflight_resp.raise_for_status = MagicMock()
        mock_preflight_resp.json = MagicMock(return_value={"models": [{"name": "llama3.2:1b"}]})

        with (
            patch("python_pipeline.pipeline.dataset_loader.load_documents", MagicMock(return_value=["doc"])),
            patch("python_pipeline.pipeline.chunker.chunk_documents", MagicMock(return_value=["chunk"])),
            patch("python_pipeline.pipeline.embedder.embed_chunks", MagicMock(return_value=[[0.1] * 384])),
            patch("python_pipeline.pipeline.VectorStore", MagicMock(return_value=MagicMock(build_index=MagicMock()))),
            patch("python_pipeline.pipeline.Retriever", MagicMock(return_value=MagicMock(retrieve=MagicMock(return_value=["chunk"])))),
            patch("python_pipeline.pipeline.llm_client.generate", side_effect=lambda **kw: next(response_iter)),
            patch("httpx.get", return_value=mock_preflight_resp),
        ):
            run_pipeline(str(config_path))

    captured = capsys.readouterr()
    # Both status lines must appear
    assert "1/2: OK" in captured.out
    assert "2/2: FAIL" in captured.out
    assert "timeout" in captured.out


def test_stress_runner_prints_ok_and_fail_status(capsys):
    """Req 2.4: Stress runner prints OK and FAIL status lines with correct query id and reason.

    Validates: Requirements 2.4
    """
    from python_pipeline.stress_runner import StressRunner
    from python_pipeline.llm_client import LLMResponse
    from python_pipeline.vector_store import VectorStore

    # Build a minimal in-memory vector store with one chunk
    chunks = ["chunk about RAG"]

    # Mock VectorStore and embedder so we don't need real data
    mock_vs = MagicMock()
    mock_vs.search = MagicMock(return_value=[0])

    call_count = {"n": 0}

    def alternating_generate(query: str, chunks: list) -> LLMResponse:
        n = call_count["n"]
        call_count["n"] += 1
        if n % 2 == 0:
            return LLMResponse(text="ok", total_tokens=5, ttft_ms=10.0, generation_ms=50.0, failed=False, failure_reason=None)
        else:
            return LLMResponse(text="", total_tokens=0, ttft_ms=0.0, generation_ms=0.0, failed=True, failure_reason="connection refused")

    mock_embedder_fn = MagicMock(return_value=[0.1] * 384)

    with patch("python_pipeline.stress_runner.embedder_module.embed_chunks", return_value=[[0.1] * 384]):
        with patch("python_pipeline.stress_runner.Retriever") as mock_retriever_cls:
            mock_retriever_instance = MagicMock()
            mock_retriever_instance.retrieve = MagicMock(return_value=["chunk about RAG"])
            mock_retriever_cls.return_value = mock_retriever_instance

            runner = StressRunner(
                chunks=chunks,
                vector_store=mock_vs,
                llm_generate_fn=alternating_generate,
                query_set=["What is RAG?", "What is a vector store?"],
                concurrency=2,
                query_repetitions=1,
            )
            results = runner.run()

    captured = capsys.readouterr()
    # Both OK and FAIL lines must appear
    assert "OK" in captured.out
    assert "FAIL" in captured.out
    assert "connection refused" in captured.out
    # Stress query lines must include query id
    assert "Stress query" in captured.out
