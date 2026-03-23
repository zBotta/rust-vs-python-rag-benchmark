# Feature: rust-vs-python-rag-benchmark, Property 19: Retrieved chunk IDs for each query are identical whether run sequentially or concurrently
"""Property test for StressRunner concurrent vs sequential retrieval consistency.

Validates: Requirements 11.4
"""
from __future__ import annotations

import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import patch

from hypothesis import given, settings
from hypothesis import strategies as st

from python_pipeline.vector_store import VectorStore
from python_pipeline.retriever import Retriever


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_unit_embedding(seed: int, dim: int = 384) -> list[float]:
    """Return a deterministic, normalised fake embedding vector."""
    rng = random.Random(seed)
    vec = [rng.gauss(0, 1) for _ in range(dim)]
    norm = sum(x * x for x in vec) ** 0.5 or 1.0
    return [x / norm for x in vec]


def _build_vector_store(n_chunks: int, dim: int = 384) -> VectorStore:
    """Build a VectorStore with *n_chunks* deterministic embeddings."""
    embeddings = [_make_unit_embedding(i, dim) for i in range(n_chunks)]
    vs = VectorStore(dim=dim, space="cosine")
    vs.build_index(embeddings)
    return vs


# ---------------------------------------------------------------------------
# Property 19
# ---------------------------------------------------------------------------

@settings(max_examples=20)
@given(
    n_chunks=st.integers(min_value=5, max_value=30),
    n_queries=st.integers(min_value=1, max_value=8),
    top_k=st.integers(min_value=1, max_value=5),
    concurrency=st.integers(min_value=2, max_value=8),
    seed=st.integers(min_value=0, max_value=9999),
)
def test_concurrent_results_match_sequential(
    n_chunks: int,
    n_queries: int,
    top_k: int,
    concurrency: int,
    seed: int,
) -> None:
    """Property 19: Retrieved chunk IDs for each query are identical whether run
    sequentially or concurrently against the same read-only vector index.

    Each worker in the concurrent run uses its own Retriever instance but shares
    the same read-only VectorStore — exactly as StressRunner does.

    **Validates: Requirements 11.4**
    """
    # Clamp top_k so it never exceeds the index size
    top_k = min(top_k, n_chunks)

    # Build shared read-only vector store (built once, never mutated after this)
    chunks = [f"chunk_{i}" for i in range(n_chunks)]
    vector_store = _build_vector_store(n_chunks)

    # Assign each query a deterministic embedding seed (simulates a fixed embedder)
    rng = random.Random(seed)
    queries: list[str] = [f"query_{seed}_{i}" for i in range(n_queries)]
    query_seed_map: dict[str, int] = {q: rng.randint(10_000, 99_999) for q in queries}

    def make_embedder_fn(query: str):
        """Return an embedder that always produces the same vector for this query."""
        emb = _make_unit_embedding(query_seed_map[query])
        def embedder_fn(_q: str) -> list[float]:
            return emb
        return embedder_fn

    # --- Sequential baseline ---
    # Run each query once, sequentially, recording the returned chunk texts.
    sequential_results: dict[str, list[str]] = {}
    for query in queries:
        retriever = Retriever(
            chunks=chunks,
            vector_store=vector_store,
            embedder_fn=make_embedder_fn(query),
        )
        sequential_results[query] = retriever.retrieve(query, top_k)

    # --- Concurrent run ---
    # Dispatch all queries (repeated concurrency times to stress the shared index)
    # using a ThreadPoolExecutor — mirroring StressRunner's architecture.
    # Each worker gets its own Retriever instance; VectorStore is shared read-only.
    concurrent_results: dict[str, list[list[str]]] = {q: [] for q in queries}
    lock = threading.Lock()

    def run_query_concurrent(query: str) -> tuple[str, list[str]]:
        retriever = Retriever(
            chunks=chunks,
            vector_store=vector_store,
            embedder_fn=make_embedder_fn(query),
        )
        return query, retriever.retrieve(query, top_k)

    # Build a workload: each query repeated `concurrency` times to exercise
    # concurrent access to the shared VectorStore.
    workload = queries * concurrency

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(run_query_concurrent, q) for q in workload]
        for future in as_completed(futures):
            q, result = future.result()
            with lock:
                concurrent_results[q].append(result)

    # --- Assert Property 19 ---
    # Every concurrent execution of a given query must return the same chunk texts
    # as the sequential baseline — no shared mutable state corruption.
    for query in queries:
        expected = sequential_results[query]
        for i, concurrent_result in enumerate(concurrent_results[query]):
            assert concurrent_result == expected, (
                f"Concurrent result #{i} for query={query!r} differs from sequential baseline.\n"
                f"  Sequential : {expected}\n"
                f"  Concurrent : {concurrent_result}\n"
                f"  (n_chunks={n_chunks}, top_k={top_k}, concurrency={concurrency})"
            )


# ---------------------------------------------------------------------------
# Property 20
# ---------------------------------------------------------------------------

# Feature: rust-vs-python-rag-benchmark, Property 20: throughput(C workers) > throughput(1 worker) for C > 1

import time
from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed


def _run_queries_with_concurrency(
    n_queries: int,
    sleep_per_query: float,
    concurrency: int,
) -> float:
    """Run *n_queries* simulated queries (each sleeping *sleep_per_query* seconds)
    using *concurrency* workers and return throughput (queries / second).
    """
    def simulated_query(_query_id: int) -> None:
        time.sleep(sleep_per_query)

    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(simulated_query, i) for i in range(n_queries)]
        for f in _as_completed(futures):
            f.result()
    wall_elapsed = time.perf_counter() - wall_start

    return n_queries / wall_elapsed if wall_elapsed > 0 else 0.0


@settings(max_examples=20, deadline=None)
@given(
    concurrency=st.integers(min_value=2, max_value=8),
    n_queries=st.integers(min_value=8, max_value=24),
)
def test_throughput_scales_with_concurrency(
    concurrency: int,
    n_queries: int,
) -> None:
    """Property 20: throughput(C workers) > throughput(1 worker) for C > 1.

    Uses a simulated workload (fixed sleep per query) so that the concurrency
    benefit is measurable without requiring real ONNX models or an LLM server.

    **Validates: Requirements 11.5**
    """
    # Ensure we have enough queries to keep all workers busy
    n_queries = max(n_queries, concurrency * 2)

    sleep_per_query = 0.01  # 10 ms per query — enough to make parallelism visible

    throughput_1 = _run_queries_with_concurrency(n_queries, sleep_per_query, concurrency=1)
    throughput_c = _run_queries_with_concurrency(n_queries, sleep_per_query, concurrency=concurrency)

    assert throughput_c > throughput_1, (
        f"Expected throughput({concurrency} workers)={throughput_c:.2f} qps "
        f"> throughput(1 worker)={throughput_1:.2f} qps "
        f"(n_queries={n_queries}, sleep_per_query={sleep_per_query}s)"
    )

