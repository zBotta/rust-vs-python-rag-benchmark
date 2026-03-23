"""Property-based tests for the embedder module.

Sub-task 4.1 — Property 4: Every embedding vector has exactly 384 dimensions

Validates: Requirements 3.1

The SentenceTransformer model is mocked to avoid downloading model weights
during testing. The mock returns a numpy array of shape (n, 384) for any
input list of n strings, which is the contract the real model satisfies.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
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
@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
def test_every_embedding_has_384_dimensions(chunks: list[str]) -> None:
    """Property 4: Every embedding vector returned by embed_chunks has exactly
    384 dimensions, for any non-empty list of input strings.

    The SentenceTransformer is mocked to return a numpy array of shape
    (len(chunks), 384), which is the contract the real model satisfies.
    This test verifies that embed_chunks correctly propagates the 384-dim
    vectors from the model to the caller.

    # Feature: rust-vs-python-rag-benchmark, Property 4: Every embedding vector has exactly 384 dimensions
    Validates: Requirements 3.1
    """
    mock_model = MagicMock()
    # The real SentenceTransformer.encode returns a numpy array of shape (n, 384).
    mock_model.encode.return_value = np.zeros((len(chunks), 384), dtype=np.float32)

    mock_st_class = MagicMock(return_value=mock_model)

    with patch.dict("sys.modules", {"sentence_transformers": MagicMock(SentenceTransformer=mock_st_class)}):
        # Re-import inside the patch context so the mock is picked up.
        import importlib
        import python_pipeline.embedder as embedder_mod
        importlib.reload(embedder_mod)

        embeddings = embedder_mod.embed_chunks(chunks)

    assert len(embeddings) == len(chunks), (
        f"Expected {len(chunks)} embeddings, got {len(embeddings)}"
    )
    for i, vec in enumerate(embeddings):
        assert len(vec) == 384, (
            f"Embedding {i} has {len(vec)} dimensions, expected 384"
        )

