"""Structured per-backend benchmark logger.

Writes log records to {output_dir}/{backend}-{language}.log.
No global logging configuration is modified.
"""

from __future__ import annotations

import enum
import os
from datetime import datetime, timezone
from pathlib import Path


class LogLevel(enum.IntEnum):
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3


class BenchmarkLogger:
    def __init__(
        self,
        output_dir: str,
        backend: str,
        language: str = "python",
        min_level: LogLevel = LogLevel.INFO,
    ) -> None:
        self._min_level = min_level
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        log_path = path / f"{backend}-{language}.log"
        self._file = open(log_path, "w", encoding="utf-8")  # truncate on open

    # ------------------------------------------------------------------
    # Core primitive
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush and close the underlying log file."""
        if not self._file.closed:
            self._file.flush()
            self._file.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def log(self, level: LogLevel, stage: str, message: str) -> None:
        if level < self._min_level:
            return
        ts = datetime.now(timezone.utc)
        ms = ts.microsecond // 1000
        timestamp = ts.strftime("%Y-%m-%dT%H:%M:%S") + f".{ms:03d}Z"
        line = f"{timestamp} [{level.name}] [{stage}] {message}\n"
        self._file.write(line)
        self._file.flush()

    # ------------------------------------------------------------------
    # Loading stage
    # ------------------------------------------------------------------

    def log_loading_start(self, dataset_name: str, subset: str, num_docs: int) -> None:
        self.log(
            LogLevel.INFO,
            "loading",
            f"Starting dataset load: {dataset_name} / {subset}, requested={num_docs} docs",
        )

    def log_loading_complete(self, num_docs: int, elapsed_ms: float) -> None:
        self.log(
            LogLevel.INFO,
            "loading",
            f"Loading complete: {num_docs} docs loaded in {elapsed_ms:.1f} ms",
        )

    def log_loading_batch(self, batch_size: int, cumulative: int) -> None:
        self.log(
            LogLevel.DEBUG,
            "loading",
            f"Batch fetched: batch_size={batch_size}, cumulative={cumulative}",
        )

    def log_loading_error(self, error: str) -> None:
        self.log(LogLevel.ERROR, "loading", f"Loading error: {error}")

    # ------------------------------------------------------------------
    # Chunking stage
    # ------------------------------------------------------------------

    def log_chunking_start(self, chunk_size: int, overlap: int) -> None:
        self.log(
            LogLevel.INFO,
            "chunking",
            f"Starting chunking: chunk_size={chunk_size}, overlap={overlap}",
        )

    def log_chunking_complete(self, num_chunks: int, elapsed_ms: float) -> None:
        self.log(
            LogLevel.INFO,
            "chunking",
            f"Chunking complete: {num_chunks} chunks produced in {elapsed_ms:.1f} ms",
        )

    def log_chunking_zero_warning(self) -> None:
        self.log(
            LogLevel.WARNING,
            "chunking",
            "Chunking produced zero chunks from a non-empty document set",
        )

    # ------------------------------------------------------------------
    # Embedding stage
    # ------------------------------------------------------------------

    def log_embedding_start(self, model: str, num_chunks: int) -> None:
        self.log(
            LogLevel.INFO,
            "embedding",
            f"Starting embedding: model={model}, num_chunks={num_chunks}",
        )

    def log_embedding_complete(self, elapsed_ms: float) -> None:
        self.log(
            LogLevel.INFO,
            "embedding",
            f"Embedding complete in {elapsed_ms:.1f} ms",
        )

    def log_embedding_progress(self, embedded_so_far: int) -> None:
        self.log(
            LogLevel.DEBUG,
            "embedding",
            f"Embedded so far: {embedded_so_far}",
        )

    def log_embedding_error(self, error: str) -> None:
        self.log(LogLevel.ERROR, "embedding", f"Embedding error: {error}")

    # ------------------------------------------------------------------
    # Index build stage
    # ------------------------------------------------------------------

    def log_index_build_start(self, num_embeddings: int) -> None:
        self.log(
            LogLevel.INFO,
            "index_build",
            f"Starting index build: num_embeddings={num_embeddings}",
        )

    def log_index_build_complete(self, elapsed_ms: float) -> None:
        self.log(
            LogLevel.INFO,
            "index_build",
            f"Index build complete in {elapsed_ms:.1f} ms",
        )

    def log_index_build_error(self, error: str) -> None:
        self.log(LogLevel.ERROR, "index_build", f"Index build error: {error}")

    # ------------------------------------------------------------------
    # Retrieval stage
    # ------------------------------------------------------------------

    def log_retrieval_start(self, query_id: int) -> None:
        self.log(LogLevel.DEBUG, "retrieval", f"Query {query_id} start")

    def log_retrieval_complete(
        self, query_id: int, num_chunks: int, elapsed_ms: float
    ) -> None:
        self.log(
            LogLevel.DEBUG,
            "retrieval",
            f"Query {query_id} complete: {num_chunks} chunks retrieved in {elapsed_ms:.1f} ms",
        )

    def log_retrieval_error(self, query_id: int, error: str) -> None:
        self.log(
            LogLevel.ERROR,
            "retrieval",
            f"Query {query_id} error: {error}",
        )

    # ------------------------------------------------------------------
    # Generation stage
    # ------------------------------------------------------------------

    def log_generation_start(self, query_id: int, num_chunks: int) -> None:
        self.log(
            LogLevel.DEBUG,
            "generation",
            f"Query {query_id} start: num_chunks={num_chunks}",
        )

    def log_generation_complete(
        self,
        query_id: int,
        total_tokens: int,
        ttft_ms: float,
        generation_ms: float,
    ) -> None:
        self.log(
            LogLevel.DEBUG,
            "generation",
            f"Query {query_id} complete: total_tokens={total_tokens},"
            f" ttft_ms={ttft_ms:.1f}, generation_ms={generation_ms:.1f}",
        )

    def log_generation_failed_response(self, query_id: int, reason: str) -> None:
        self.log(
            LogLevel.WARNING,
            "generation",
            f"Query {query_id} failed response: {reason}",
        )

    def log_generation_error(self, query_id: int, error: str) -> None:
        self.log(
            LogLevel.ERROR,
            "generation",
            f"Query {query_id} error: {error}",
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def log_run_summary(
        self,
        total_queries: int,
        failures: int,
        p50_ms: float,
        p95_ms: float,
        output_path: str,
    ) -> None:
        self.log(
            LogLevel.INFO,
            "summary",
            f"Run complete: total_queries={total_queries}, failures={failures},"
            f" p50_ms={p50_ms:.1f}, p95_ms={p95_ms:.1f}, output={output_path}",
        )

    def log_stress_summary(
        self, qps: float, peak_rss_mb: float, p99_ms: float
    ) -> None:
        self.log(
            LogLevel.INFO,
            "summary",
            f"Stress test complete: qps={qps:.2f}, peak_rss_mb={peak_rss_mb:.1f},"
            f" p99_ms={p99_ms:.1f}",
        )
