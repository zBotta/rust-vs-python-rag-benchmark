"""Metrics collector — Task 6 implementation.

Provides QueryMetrics and PipelineMetrics dataclasses, percentile computation,
and JSONL serialization/deserialization.
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class QueryMetrics:
    query_id: int
    end_to_end_ms: float
    retrieval_ms: float
    ttft_ms: float
    generation_ms: float
    total_tokens: int
    failed: bool
    failure_reason: Optional[str]


@dataclass
class PipelineMetrics:
    embedding_phase_ms: float
    index_build_ms: float
    queries: list[QueryMetrics] = field(default_factory=list)
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0


def compute_percentiles(latencies: list[float]) -> tuple[float, float]:
    """Return (p50, p95) for the given list of latency values.

    Uses statistics.median for p50 and a linear-interpolation percentile
    for p95 (equivalent to numpy.percentile with interpolation='linear').

    Returns (0.0, 0.0) for an empty list.
    """
    if not latencies:
        return 0.0, 0.0

    sorted_vals = sorted(latencies)
    n = len(sorted_vals)

    # p50 — standard median
    p50 = statistics.median(sorted_vals)

    # p95 — linear interpolation (matches numpy default)
    p95 = _percentile(sorted_vals, 95.0)

    return p50, p95


def _percentile(sorted_vals: list[float], pct: float) -> float:
    """Compute a percentile using linear interpolation (numpy-compatible)."""
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]

    # numpy uses: index = pct/100 * (n-1)
    idx = pct / 100.0 * (n - 1)
    lo = int(idx)
    hi = lo + 1
    frac = idx - lo

    if hi >= n:
        return sorted_vals[-1]

    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])


def serialize_to_jsonl(metrics: PipelineMetrics, output_path: str) -> None:
    """Write PipelineMetrics to a JSONL file.

    Each query is written as one JSON object per line with ``"type": "query"``.
    The final line is a summary object with ``"type": "summary"``.
    """
    successful_latencies = [
        q.end_to_end_ms for q in metrics.queries if not q.failed
    ]
    p50, p95 = compute_percentiles(successful_latencies)
    failure_count = sum(1 for q in metrics.queries if q.failed)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as fh:
        for q in metrics.queries:
            record = {
                "type": "query",
                "query_id": q.query_id,
                "end_to_end_ms": q.end_to_end_ms,
                "retrieval_ms": q.retrieval_ms,
                "ttft_ms": q.ttft_ms,
                "generation_ms": q.generation_ms,
                "total_tokens": q.total_tokens,
                "failed": q.failed,
                "failure_reason": q.failure_reason,
            }
            fh.write(json.dumps(record) + "\n")

        summary = {
            "type": "summary",
            "embedding_phase_ms": metrics.embedding_phase_ms,
            "index_build_ms": metrics.index_build_ms,
            "p50_latency_ms": p50,
            "p95_latency_ms": p95,
            "failure_count": failure_count,
        }
        fh.write(json.dumps(summary) + "\n")


def deserialize_from_jsonl(input_path: str) -> PipelineMetrics:
    """Read a JSONL file produced by serialize_to_jsonl and return PipelineMetrics.

    Expects query records followed by a single summary record as the last line.
    """
    path = Path(input_path)
    queries: list[QueryMetrics] = []
    summary: dict = {}

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("type") == "query":
                queries.append(
                    QueryMetrics(
                        query_id=obj["query_id"],
                        end_to_end_ms=obj["end_to_end_ms"],
                        retrieval_ms=obj["retrieval_ms"],
                        ttft_ms=obj["ttft_ms"],
                        generation_ms=obj["generation_ms"],
                        total_tokens=obj["total_tokens"],
                        failed=obj["failed"],
                        failure_reason=obj.get("failure_reason"),
                    )
                )
            elif obj.get("type") == "summary":
                summary = obj

    return PipelineMetrics(
        embedding_phase_ms=summary.get("embedding_phase_ms", 0.0),
        index_build_ms=summary.get("index_build_ms", 0.0),
        queries=queries,
        p50_latency_ms=summary.get("p50_latency_ms", 0.0),
        p95_latency_ms=summary.get("p95_latency_ms", 0.0),
    )


# ---------------------------------------------------------------------------
# Stress test metrics — Task 19
# ---------------------------------------------------------------------------

@dataclass
class StressSummary:
    concurrency: int
    total_queries: int
    queries_per_second: float
    peak_rss_mb: float
    p99_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    failure_count: int


def compute_stress_summary(
    query_metrics: list[QueryMetrics],
    concurrency: int,
    total_wall_clock_s: float,
) -> StressSummary:
    """Compute stress test summary metrics from a list of QueryMetrics.

    Uses psutil to get peak RSS memory.
    queries_per_second = len(query_metrics) / total_wall_clock_s
    p50/p95/p99 computed from successful query latencies only.
    """
    import psutil  # imported here to avoid hard dependency at module level

    queries_per_second = len(query_metrics) / total_wall_clock_s if total_wall_clock_s > 0 else 0.0
    peak_rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)

    successful_latencies = sorted(
        q.end_to_end_ms for q in query_metrics if not q.failed
    )
    p50, p95 = compute_percentiles(successful_latencies)
    p99 = _percentile(successful_latencies, 99.0) if successful_latencies else 0.0
    failure_count = sum(1 for q in query_metrics if q.failed)

    return StressSummary(
        concurrency=concurrency,
        total_queries=len(query_metrics),
        queries_per_second=queries_per_second,
        peak_rss_mb=peak_rss_mb,
        p99_latency_ms=p99,
        p50_latency_ms=p50,
        p95_latency_ms=p95,
        failure_count=failure_count,
    )


def append_stress_summary_to_jsonl(summary: StressSummary, output_path: str) -> None:
    """Append a stress_summary JSON record to an existing JSONL file."""
    record = {
        "type": "stress_summary",
        "concurrency": summary.concurrency,
        "total_queries": summary.total_queries,
        "queries_per_second": summary.queries_per_second,
        "peak_rss_mb": summary.peak_rss_mb,
        "p99_latency_ms": summary.p99_latency_ms,
        "p50_latency_ms": summary.p50_latency_ms,
        "p95_latency_ms": summary.p95_latency_ms,
        "failure_count": summary.failure_count,
    }
    with open(output_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def read_stress_summary_from_jsonl(input_path: str) -> Optional[StressSummary]:
    """Read a JSONL file and return the stress_summary record if present, or None."""
    path = Path(input_path)
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("type") == "stress_summary":
                return StressSummary(
                    concurrency=obj["concurrency"],
                    total_queries=obj["total_queries"],
                    queries_per_second=obj["queries_per_second"],
                    peak_rss_mb=obj["peak_rss_mb"],
                    p99_latency_ms=obj["p99_latency_ms"],
                    p50_latency_ms=obj["p50_latency_ms"],
                    p95_latency_ms=obj["p95_latency_ms"],
                    failure_count=obj["failure_count"],
                )
    return None
