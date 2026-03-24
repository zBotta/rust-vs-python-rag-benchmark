"""Property-based tests for the embedder module.

Sub-task 4.1 — Property 4: Every embedding vector has exactly 384 dimensions

Validates: Requirements 3.1

The embedder's embed_chunks is tested by mocking _get_model and the
torch-dependent internals to produce 384-dim outputs, bypassing real model
loading. The module is imported at module level to ensure torch is loaded
before any test stubs can replace it.
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

# Import torch and embedder at module level so torch is in sys.modules before
# any test file can stub it (test_ollama_preflight.py stubs torch at import time).
import torch
import python_pipeline.embedder as embedder_mod

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

# Allow running tests from the project root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Sub-task 4.1 — Property 4: Every embedding vector has exactly 384 dimensions
# Feature: rust-vs-python-rag-benchmark, Property 4: Every embedding vector has exactly 384 dimensions
# ---------------------------------------------------------------------------

# Feature: rust-vs-python-rag-benchmark, Property 4: Every embedding vector has exactly 384 dimensions


@given(
    chunks=st.lists(
        st.text(min_size=1, max_size=200),
        min_size=1,
        max_size=20,
    )
)
@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_every_embedding_has_384_dimensions(chunks: list[str]) -> None:
    """Property 4: Every embedding vector returned by embed_chunks has exactly
    384 dimensions, for any non-empty list of input strings.

    embed_chunks is tested by mocking _get_model and torch internals so the
    test verifies the output shape contract without requiring real model weights.

    # Feature: rust-vs-python-rag-benchmark, Property 4: Every embedding vector has exactly 384 dimensions
    Validates: Requirements 3.1
    """
    n = len(chunks)
    # Produce a (n, 384) float32 numpy array — the shape the real model returns
    expected_output = np.zeros((n, 384), dtype=np.float32)

    # Mock tokenizer: returns a dict-like object
    mock_encoded = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
    mock_tokenizer = MagicMock(return_value=mock_encoded)

    # Mock model outputs
    mock_outputs = MagicMock()
    mock_model = MagicMock(return_value=mock_outputs)

    # Mock _mean_pool to return a mock with .cpu().numpy() returning expected_output
    mock_pooled = MagicMock()
    mock_pooled.cpu.return_value.numpy.return_value = expected_output

    # no_grad context manager mock (in case torch is stubbed in the test environment)
    @contextlib.contextmanager
    def mock_no_grad():
        yield

    with (
        patch.object(embedder_mod, "_get_model", return_value=(mock_tokenizer, mock_model)),
        patch.object(embedder_mod, "_mean_pool", return_value=mock_pooled),
        patch.object(torch, "no_grad", mock_no_grad),
    ):
        embeddings = embedder_mod.embed_chunks(chunks)

    assert len(embeddings) == len(chunks), (
        f"Expected {len(chunks)} embeddings, got {len(embeddings)}"
    )
    for i, vec in enumerate(embeddings):
        assert len(vec) == 384, (
            f"Embedding {i} has {len(vec)} dimensions, expected 384"
        )
