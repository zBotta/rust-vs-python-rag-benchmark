"""Stress test mode — concurrent query dispatcher (Task 18).

Dispatches query_repetitions × len(query_set) queries across a thread pool,
each worker using its own Embedder and Retriever instance against a shared
read-only VectorStore.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from python_pipeline import embedder as embedder_module
from python_pipeline.retriever import Retriever
from python_pipeline.metrics_collector import QueryMetrics


class StressRunner:
    """Concurrent query dispatcher for stress test mode."""

    def __init__(
        self,
        chunks: list[str],
        vector_store,  # shared read-only VectorStore
        llm_generate_fn: Callable,
        query_set: list[str],
        concurrency: int,
        query_repetitions: int,
    ) -> None:
        self._chunks = chunks
        self._vector_store = vector_store
        self._llm_generate_fn = llm_generate_fn
        self._query_set = query_set
        self._concurrency = concurrency
        self._query_repetitions = query_repetitions
        self.last_wall_clock_s: float = 0.0

    def run(self) -> list[QueryMetrics]:
        """Dispatch all queries concurrently and return per-query metrics.

        Total queries = query_repetitions × len(query_set).
        Each worker gets its own Embedder and Retriever instance.
        """
        # Build full query list: query_set repeated query_repetitions times
        all_queries: list[tuple[int, str]] = []
        for rep in range(self._query_repetitions):
            for idx, question in enumerate(self._query_set):
                query_id = rep * len(self._query_set) + idx
                all_queries.append((query_id, question))

        results: list[QueryMetrics] = []
        lock = threading.Lock()

        def run_query(query_id: int, question: str) -> QueryMetrics:
            # Each worker gets its own embedder_fn and Retriever
            def embedder_fn(q: str) -> list[float]:
                return embedder_module.embed_chunks([q])[0]

            retriever = Retriever(
                chunks=self._chunks,
                vector_store=self._vector_store,
                embedder_fn=embedder_fn,
            )

            e2e_start = time.perf_counter()
            try:
                retrieval_start = time.perf_counter()
                retrieved_chunks = retriever.retrieve(question, top_k=5)
                retrieval_ms = (time.perf_counter() - retrieval_start) * 1000.0

                response = self._llm_generate_fn(
                    query=question,
                    chunks=retrieved_chunks,
                )
                end_to_end_ms = (time.perf_counter() - e2e_start) * 1000.0

                if response.failed:
                    return QueryMetrics(
                        query_id=query_id,
                        end_to_end_ms=end_to_end_ms,
                        retrieval_ms=retrieval_ms,
                        ttft_ms=0.0,
                        generation_ms=0.0,
                        total_tokens=0,
                        failed=True,
                        failure_reason=response.failure_reason,
                    )
                return QueryMetrics(
                    query_id=query_id,
                    end_to_end_ms=end_to_end_ms,
                    retrieval_ms=retrieval_ms,
                    ttft_ms=response.ttft_ms,
                    generation_ms=response.generation_ms,
                    total_tokens=response.total_tokens,
                    failed=False,
                    failure_reason=None,
                )
            except Exception as exc:
                end_to_end_ms = (time.perf_counter() - e2e_start) * 1000.0
                return QueryMetrics(
                    query_id=query_id,
                    end_to_end_ms=end_to_end_ms,
                    retrieval_ms=0.0,
                    ttft_ms=0.0,
                    generation_ms=0.0,
                    total_tokens=0,
                    failed=True,
                    failure_reason=str(exc),
                )

        with ThreadPoolExecutor(max_workers=self._concurrency) as executor:
            futures = {
                executor.submit(run_query, qid, question): (qid, question)
                for qid, question in all_queries
            }
            wall_start = time.perf_counter()
            for future in as_completed(futures):
                qm = future.result()
                with lock:
                    results.append(qm)
                status = "FAIL" if qm.failed else "OK"
                reason = f" — {qm.failure_reason}" if qm.failed else ""
                print(f"  Stress query {qm.query_id}: {status}{reason}")
        self.last_wall_clock_s = time.perf_counter() - wall_start

        return results
