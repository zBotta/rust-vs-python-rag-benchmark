"""Property-based tests for the metrics_collector module.

Sub-task 6.1 — Property 9: All metric fields present and non-negative
Sub-task 6.2 — Property 10: p50 = median, p95 = 95th percentile
Sub-task 6.3 — Property 11: Failed queries excluded from percentile calculations
Sub-task 6.4 — Property 12: JSONL metrics round-trip

Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5
"""
from __future__ import annotations

import statistics
import sys
import tempfile
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from python_pipeline.metrics_collector import (
    PipelineMetrics,
    QueryMetrics,
    compute_percentiles,
    deserialize_from_jsonl,
    serialize_to_jsonl,
)

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

non_negative_float = st.floats(min_value=0.0, max_value=1_000_000.0, allow_nan=False, allow_infinity=False)
non_negative_int = st.integers(min_value=0, max_value=100_000)


def query_metrics_strategy(query_id: int = 0, failed: bool = False):
    return st.builds(
        QueryMetrics,
        query_id=st.just(query_id),
        end_to_end_ms=non_negative_float,
        retrieval_ms=non_negative_float,
        ttft_ms=non_negative_float,
        generation_ms=non_negative_float,
        total_tokens=non_negative_int,
        failed=st.just(failed),
        failure_reason=st.just("test failure" if failed else None),
    )


# ---------------------------------------------------------------------------
# Sub-task 6.1 — Property 9: All metric fields present and non-negative
# Feature: rust-vs-python-rag-benchmark, Property 9: All metric fields present and non-negative
# ---------------------------------------------------------------------------

# Feature: rust-vs-python-rag-benchmark, Property 9: All metric fields present and non-negative


@given(
    end_to_end_ms=non_negative_float,
    retrieval_ms=non_negative_float,
    ttft_ms=non_negative_float,
    generation_ms=non_negative_float,
    total_tokens=non_negative_int,
    embedding_phase_ms=non_negative_float,
    index_build_ms=non_negative_float,
)
@settings(max_examples=20)
def test_all_metric_fields_non_negative(
    end_to_end_ms: float,
    retrieval_ms: float,
    ttft_ms: float,
    generation_ms: float,
    total_tokens: int,
    embedding_phase_ms: float,
    index_build_ms: float,
) -> None:
    """Property 9: All metric fields present and non-negative.

    # Feature: rust-vs-python-rag-benchmark, Property 9: All metric fields present and non-negative
    Validates: Requirements 6.1, 6.2
    """
    q = QueryMetrics(
        query_id=0,
        end_to_end_ms=end_to_end_ms,
        retrieval_ms=retrieval_ms,
        ttft_ms=ttft_ms,
        generation_ms=generation_ms,
        total_tokens=total_tokens,
        failed=False,
        failure_reason=None,
    )

    assert q.end_to_end_ms >= 0.0, "end_to_end_ms must be non-negative"
    assert q.retrieval_ms >= 0.0, "retrieval_ms must be non-negative"
    assert q.ttft_ms >= 0.0, "ttft_ms must be non-negative"
    assert q.generation_ms >= 0.0, "generation_ms must be non-negative"
    assert q.total_tokens >= 0, "total_tokens must be non-negative"

    pipeline = PipelineMetrics(
        embedding_phase_ms=embedding_phase_ms,
        index_build_ms=index_build_ms,
        queries=[q],
    )

    assert pipeline.embedding_phase_ms >= 0.0, "embedding_phase_ms must be non-negative"
    assert pipeline.index_build_ms >= 0.0, "index_build_ms must be non-negative"


# ---------------------------------------------------------------------------
# Sub-task 6.2 — Property 10: p50 = median, p95 = 95th percentile
# Feature: rust-vs-python-rag-benchmark, Property 10: p50 = median, p95 = 95th percentile of end-to-end latency values
# ---------------------------------------------------------------------------

# Feature: rust-vs-python-rag-benchmark, Property 10: p50 = median, p95 = 95th percentile of end-to-end latency values


def _numpy_percentile(sorted_vals: list[float], pct: float) -> float:
    """Linear-interpolation percentile matching numpy.percentile default."""
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    idx = pct / 100.0 * (n - 1)
    lo = int(idx)
    hi = lo + 1
    frac = idx - lo
    if hi >= n:
        return sorted_vals[-1]
    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])


@given(
    latencies=st.lists(
        st.floats(min_value=0.0, max_value=100_000.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=200,
    )
)
@settings(max_examples=20)
def test_p50_is_median_p95_is_95th_percentile(latencies: list[float]) -> None:
    """Property 10: p50 = median, p95 = 95th percentile of end-to-end latency values.

    # Feature: rust-vs-python-rag-benchmark, Property 10: p50 = median, p95 = 95th percentile of end-to-end latency values
    Validates: Requirements 6.3
    """
    p50, p95 = compute_percentiles(latencies)

    sorted_vals = sorted(latencies)
    expected_p50 = _numpy_percentile(sorted_vals, 50.0)
    expected_p95 = _numpy_percentile(sorted_vals, 95.0)

    assert abs(p50 - expected_p50) < 1e-9, f"p50={p50} expected={expected_p50}"
    assert abs(p95 - expected_p95) < 1e-9, f"p95={p95} expected={expected_p95}"

    # Sanity: p50 and p95 within [min, max]
    assert p50 >= sorted_vals[0] and p50 <= sorted_vals[-1], "p50 out of range"
    assert p95 >= sorted_vals[0] and p95 <= sorted_vals[-1], "p95 out of range"
    assert p95 >= p50, "p95 must be >= p50"


# ---------------------------------------------------------------------------
# Sub-task 6.3 — Property 11: Failed queries excluded from percentile calculations
# Feature: rust-vs-python-rag-benchmark, Property 11: Percentiles computed only from successful queries; failure count = count of failed=true records
# ---------------------------------------------------------------------------

# Feature: rust-vs-python-rag-benchmark, Property 11: Percentiles computed only from successful queries; failure count = count of failed=true records


@given(
    successful_latencies=st.lists(
        st.floats(min_value=0.0, max_value=100_000.0, allow_nan=False, allow_infinity=False),
        min_size=0,
        max_size=50,
    ),
    failed_latencies=st.lists(
        st.floats(min_value=0.0, max_value=100_000.0, allow_nan=False, allow_infinity=False),
        min_size=0,
        max_size=20,
    ),
)
@settings(max_examples=20)
def test_failed_queries_excluded_from_percentiles(
    successful_latencies: list[float],
    failed_latencies: list[float],
) -> None:
    """Property 11: Percentiles computed only from successful queries; failure count = count of failed=true records.

    # Feature: rust-vs-python-rag-benchmark, Property 11: Percentiles computed only from successful queries; failure count = count of failed=true records
    Validates: Requirements 6.5
    """
    queries: list[QueryMetrics] = []
    qid = 0

    for lat in successful_latencies:
        queries.append(QueryMetrics(
            query_id=qid, end_to_end_ms=lat, retrieval_ms=0.0,
            ttft_ms=0.0, generation_ms=0.0, total_tokens=0,
            failed=False, failure_reason=None,
        ))
        qid += 1

    for lat in failed_latencies:
        queries.append(QueryMetrics(
            query_id=qid, end_to_end_ms=lat, retrieval_ms=0.0,
            ttft_ms=0.0, generation_ms=0.0, total_tokens=0,
            failed=True, failure_reason="simulated failure",
        ))
        qid += 1

    # failure_count must equal number of failed queries
    failure_count = sum(1 for q in queries if q.failed)
    assert failure_count == len(failed_latencies), (
        f"failure_count={failure_count} != len(failed_latencies)={len(failed_latencies)}"
    )

    # Percentiles must be computed from successful latencies only
    expected_p50, expected_p95 = compute_percentiles(successful_latencies)

    actual_success_lats = [q.end_to_end_ms for q in queries if not q.failed]
    actual_p50, actual_p95 = compute_percentiles(actual_success_lats)

    assert abs(actual_p50 - expected_p50) < 1e-9, (
        f"p50 mismatch: actual={actual_p50} expected={expected_p50}"
    )
    assert abs(actual_p95 - expected_p95) < 1e-9, (
        f"p95 mismatch: actual={actual_p95} expected={expected_p95}"
    )


# ---------------------------------------------------------------------------
# Sub-task 6.4 — Property 12: JSONL metrics round-trip
# Feature: rust-vs-python-rag-benchmark, Property 12: serialize(PipelineMetrics) → JSONL → deserialize == original
# ---------------------------------------------------------------------------

# Feature: rust-vs-python-rag-benchmark, Property 12: serialize(PipelineMetrics) → JSONL → deserialize == original


@given(
    embedding_phase_ms=non_negative_float,
    index_build_ms=non_negative_float,
    query_data=st.lists(
        st.tuples(
            st.floats(min_value=0.0, max_value=100_000.0, allow_nan=False, allow_infinity=False),
            st.booleans(),
        ),
        min_size=0,
        max_size=20,
    ),
)
@settings(max_examples=20)
def test_jsonl_round_trip(
    embedding_phase_ms: float,
    index_build_ms: float,
    query_data: list[tuple[float, bool]],
) -> None:
    """Property 12: serialize(PipelineMetrics) → JSONL → deserialize == original.

    # Feature: rust-vs-python-rag-benchmark, Property 12: serialize(PipelineMetrics) → JSONL → deserialize == original
    Validates: Requirements 6.4
    """
    queries = [
        QueryMetrics(
            query_id=i,
            end_to_end_ms=lat,
            retrieval_ms=1.0,
            ttft_ms=2.0,
            generation_ms=3.0,
            total_tokens=10,
            failed=failed,
            failure_reason="err" if failed else None,
        )
        for i, (lat, failed) in enumerate(query_data)
    ]

    # Compute p50/p95 from successful queries (as serialize_to_jsonl does)
    success_lats = [q.end_to_end_ms for q in queries if not q.failed]
    p50, p95 = compute_percentiles(success_lats)

    original = PipelineMetrics(
        embedding_phase_ms=embedding_phase_ms,
        index_build_ms=index_build_ms,
        queries=queries,
        p50_latency_ms=p50,
        p95_latency_ms=p95,
    )

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        serialize_to_jsonl(original, tmp_path)
        restored = deserialize_from_jsonl(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    assert restored.embedding_phase_ms == original.embedding_phase_ms
    assert restored.index_build_ms == original.index_build_ms
    assert abs(restored.p50_latency_ms - original.p50_latency_ms) < 1e-9
    assert abs(restored.p95_latency_ms - original.p95_latency_ms) < 1e-9
    assert len(restored.queries) == len(original.queries)

    for r, o in zip(restored.queries, original.queries):
        assert r.query_id == o.query_id
        assert r.end_to_end_ms == o.end_to_end_ms
        assert r.retrieval_ms == o.retrieval_ms
        assert r.ttft_ms == o.ttft_ms
        assert r.generation_ms == o.generation_ms
        assert r.total_tokens == o.total_tokens
        assert r.failed == o.failed
        assert r.failure_reason == o.failure_reason

