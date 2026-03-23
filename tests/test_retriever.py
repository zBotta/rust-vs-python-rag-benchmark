# Feature: rust-vs-python-rag-benchmark, Property 5: Retriever returns exactly top_k results when index has >= top_k entries
"""Property tests for the Retriever component.

Validates: Requirements 4.4
"""
from __future__ import annotations

import random

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from python_pipeline.vector_store import VectorStore
from python_pipeline.retriever import Retriever


def _make_fake_embedding(seed: int, dim: int = 384) -> list[float]:
    """Return a deterministic fake embedding vector."""
    rng = random.Random(seed)
    vec = [rng.gauss(0, 1) for _ in range(dim)]
    norm = sum(x * x for x in vec) ** 0.5 or 1.0
    return [x / norm for x in vec]


def _fake_embedder_fn(seed_offset: int = 9999) -> callable:
    """Return an embedder function that ignores the query and returns a fixed fake vector."""
    def embedder(query: str) -> list[float]:
        return _make_fake_embedding(seed_offset)
    return embedder


@settings(max_examples=20)
@given(
    top_k=st.integers(min_value=1, max_value=10),
    n_extra=st.integers(min_value=0, max_value=20),
    query_seed=st.integers(min_value=0, max_value=999),
)
def test_retriever_returns_exactly_top_k(top_k: int, n_extra: int, query_seed: int) -> None:
    """Property 5: Retriever returns exactly top_k results when index has >= top_k entries."""
    n = top_k + n_extra  # n >= top_k

    # Build fake embeddings and chunks
    embeddings = [_make_fake_embedding(i) for i in range(n)]
    chunks = [f"chunk_{i}" for i in range(n)]

    # Build vector store
    vs = VectorStore(dim=384, space="cosine")
    vs.build_index(embeddings)

    # Embedder returns a fixed fake query embedding
    embedder_fn = _fake_embedder_fn(seed_offset=query_seed + 10000)

    retriever = Retriever(chunks=chunks, vector_store=vs, embedder_fn=embedder_fn)
    results = retriever.retrieve(query="some query", top_k=top_k)

    assert len(results) == top_k, (
        f"Expected exactly {top_k} results, got {len(results)} "
        f"(index size={n})"
    )

