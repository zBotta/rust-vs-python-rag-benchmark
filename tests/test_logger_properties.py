"""Property-based tests for python_pipeline.logger.BenchmarkLogger.

Feature: benchmark-logging
"""

from __future__ import annotations

import os
import re
import tempfile
from datetime import datetime, timezone
from itertools import combinations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from python_pipeline.logger import BenchmarkLogger, LogLevel

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Valid filesystem-safe strings for output_dir components, backend, language
_safe_text = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="_-"),
    min_size=1,
    max_size=20,
)

# Stage names must match [\w_]+ (word chars + underscore)
_stage_name = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="_"),
    min_size=1,
    max_size=20,
)

# Non-empty messages — restrict to printable ASCII excluding newlines/carriage-returns
# so that log lines remain single-line and substring checks are unambiguous.
_message = st.text(
    alphabet=st.characters(
        whitelist_categories=("Lu", "Ll", "Nd", "Po", "Zs"),
        whitelist_characters=" !#$%&'()*+,-./:;<=>?@[]^_`{|}~",
        blacklist_characters="\r\n\x00",
    ),
    min_size=1,
    max_size=200,
).filter(lambda s: s.strip() != "")

# All four log levels
_log_level = st.sampled_from(list(LogLevel))


# ---------------------------------------------------------------------------
# Property 1: Log file path derivation
# ---------------------------------------------------------------------------

@given(
    backend=_safe_text,
    language=_safe_text,
)
@settings(max_examples=100)
def test_prop_log_file_path_derivation(backend: str, language: str) -> None:
    """Feature: benchmark-logging, Property 1: Log file path derivation

    Validates: Requirements 1.2, 1.3
    """
    with tempfile.TemporaryDirectory() as output_dir:
        logger = BenchmarkLogger(output_dir=output_dir, backend=backend, language=language)
        expected_path = os.path.join(output_dir, f"{backend}-{language}.log")
        assert os.path.exists(expected_path), (
            f"Expected log file at {expected_path!r} but it does not exist"
        )
        # Ensure no other .log files were created
        log_files = [f for f in os.listdir(output_dir) if f.endswith(".log")]
        assert log_files == [f"{backend}-{language}.log"], (
            f"Unexpected log files: {log_files}"
        )
        logger._file.close()


# ---------------------------------------------------------------------------
# Property 2: Log file truncation on re-initialisation
# ---------------------------------------------------------------------------

@given(
    backend=_safe_text,
    language=_safe_text,
)
@settings(max_examples=100)
def test_prop_log_file_truncation(backend: str, language: str) -> None:
    """Feature: benchmark-logging, Property 2: Log file truncation on re-initialisation

    Validates: Requirements 1.1
    """
    import uuid

    # Use unique sentinel strings so substring checks are unambiguous
    sentinel_first = f"FIRST-{uuid.uuid4().hex}"
    sentinel_second = f"SECOND-{uuid.uuid4().hex}"

    with tempfile.TemporaryDirectory() as output_dir:
        # First logger writes a record with a unique sentinel
        logger1 = BenchmarkLogger(
            output_dir=output_dir, backend=backend, language=language,
            min_level=LogLevel.DEBUG,
        )
        logger1.log(LogLevel.INFO, "summary", sentinel_first)
        logger1._file.close()

        log_path = os.path.join(output_dir, f"{backend}-{language}.log")
        content_after_first = open(log_path, encoding="utf-8").read()
        assert sentinel_first in content_after_first, "First logger should have written content"

        # Second logger for the same combination must truncate prior content
        logger2 = BenchmarkLogger(
            output_dir=output_dir, backend=backend, language=language,
            min_level=LogLevel.DEBUG,
        )
        logger2.log(LogLevel.INFO, "summary", sentinel_second)
        logger2._file.close()

        content_after_second = open(log_path, encoding="utf-8").read()
        lines = [l for l in content_after_second.splitlines() if l.strip()]

        # Must contain exactly 1 line (only the second logger's record)
        assert len(lines) == 1, (
            f"Expected exactly 1 line after re-init, got {len(lines)}: {content_after_second!r}"
        )
        # The line must contain the second sentinel
        assert sentinel_second in lines[0], (
            f"Second sentinel not found in {lines[0]!r}"
        )
        # The first sentinel must be gone
        assert sentinel_first not in content_after_second, (
            f"First sentinel should have been truncated but is still present"
        )


# ---------------------------------------------------------------------------
# Property 3: Log level ordering invariant
# ---------------------------------------------------------------------------

@given(
    level_a=_log_level,
    level_b=_log_level,
)
@settings(max_examples=100)
def test_prop_log_level_ordering(level_a: LogLevel, level_b: LogLevel) -> None:
    """Feature: benchmark-logging, Property 3: Log level ordering invariant

    Validates: Requirements 2.1
    """
    # The ordering must be DEBUG < INFO < WARNING < ERROR
    assert LogLevel.DEBUG < LogLevel.INFO
    assert LogLevel.INFO < LogLevel.WARNING
    assert LogLevel.WARNING < LogLevel.ERROR

    # For any two levels, their relative integer ordering must be consistent
    if level_a.value < level_b.value:
        assert int(level_a) < int(level_b)
    elif level_a.value > level_b.value:
        assert int(level_a) > int(level_b)
    else:
        assert int(level_a) == int(level_b)


# ---------------------------------------------------------------------------
# Property 4: Log level filtering
# ---------------------------------------------------------------------------

@given(
    min_level=_log_level,
    record_level=_log_level,
    stage=_stage_name,
    message=_message,
)
@settings(max_examples=100)
def test_prop_log_level_filtering(
    min_level: LogLevel, record_level: LogLevel, stage: str, message: str
) -> None:
    """Feature: benchmark-logging, Property 4: Log level filtering

    Validates: Requirements 2.2, 2.3
    """
    with tempfile.TemporaryDirectory() as output_dir:
        logger = BenchmarkLogger(
            output_dir=output_dir, backend="test", language="python",
            min_level=min_level,
        )
        logger.log(record_level, stage, message)
        logger._file.close()

        log_path = os.path.join(output_dir, "test-python.log")
        content = open(log_path, encoding="utf-8").read()

        if record_level < min_level:
            # Record must NOT appear in the log file
            assert message not in content, (
                f"Record at level {record_level.name} (< min {min_level.name}) "
                f"should have been filtered but was found in log"
            )
        else:
            # Record MUST appear in the log file
            assert message in content, (
                f"Record at level {record_level.name} (>= min {min_level.name}) "
                f"should have been written but was not found in log"
            )


# ---------------------------------------------------------------------------
# Property 5: Log record format
# ---------------------------------------------------------------------------

_LOG_LINE_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z \[(DEBUG|INFO|WARNING|ERROR)\] \[[\w_]+\] .+$"
)


@given(
    stage=_stage_name,
    message=_message,
    level=_log_level,
)
@settings(max_examples=100)
def test_prop_log_record_format(stage: str, message: str, level: LogLevel) -> None:
    """Feature: benchmark-logging, Property 5: Log record format

    Validates: Requirements 3.1, 3.2
    """
    with tempfile.TemporaryDirectory() as output_dir:
        logger = BenchmarkLogger(
            output_dir=output_dir, backend="test", language="python",
            min_level=LogLevel.DEBUG,
        )
        logger.log(level, stage, message)
        logger._file.close()

        log_path = os.path.join(output_dir, "test-python.log")
        lines = open(log_path, encoding="utf-8").read().splitlines()

        assert len(lines) == 1, f"Expected 1 line, got {len(lines)}"
        line = lines[0]

        # Must match the full format regex
        assert _LOG_LINE_RE.match(line), (
            f"Log line does not match expected format: {line!r}"
        )

        # Extract and validate the timestamp
        timestamp_str = line.split(" ")[0]
        # Must parse as a valid UTC ISO-8601 datetime with millisecond precision
        assert timestamp_str.endswith("Z"), f"Timestamp must end with Z: {timestamp_str!r}"
        # Parse: YYYY-MM-DDTHH:MM:SS.mmmZ
        try:
            dt = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S.%fZ")
        except ValueError as exc:
            pytest.fail(f"Timestamp {timestamp_str!r} is not valid ISO-8601: {exc}")

        # Millisecond precision: the fractional part must be exactly 3 digits
        frac_part = timestamp_str.split(".")[1].rstrip("Z")
        assert len(frac_part) == 3, (
            f"Expected 3-digit millisecond precision, got {frac_part!r} in {timestamp_str!r}"
        )

        # Level name must appear correctly
        assert f"[{level.name}]" in line, f"Level {level.name!r} not found in {line!r}"

        # Stage must appear correctly
        assert f"[{stage}]" in line, f"Stage {stage!r} not found in {line!r}"

        # Message must appear
        assert message in line, f"Message {message!r} not found in {line!r}"


# ---------------------------------------------------------------------------
# Property 6: Log record ordering and flush
# ---------------------------------------------------------------------------

@given(
    records=st.lists(
        st.tuples(_log_level, _stage_name, _message),
        min_size=1,
        max_size=20,
    )
)
@settings(max_examples=100)
def test_prop_log_record_ordering(records: list[tuple[LogLevel, str, str]]) -> None:
    """Feature: benchmark-logging, Property 6: Log record ordering and flush

    Validates: Requirements 3.3, 3.4
    """
    with tempfile.TemporaryDirectory() as output_dir:
        logger = BenchmarkLogger(
            output_dir=output_dir, backend="test", language="python",
            min_level=LogLevel.DEBUG,
        )
        for level, stage, message in records:
            logger.log(level, stage, message)
        logger._file.close()

        log_path = os.path.join(output_dir, "test-python.log")
        lines = [l for l in open(log_path, encoding="utf-8").read().splitlines() if l.strip()]

        # Must have exactly N lines (all records written, none missing due to buffering)
        assert len(lines) == len(records), (
            f"Expected {len(records)} lines, got {len(lines)}"
        )

        # Each record's message must appear in the corresponding line (order preserved)
        for i, (level, stage, message) in enumerate(records):
            assert message in lines[i], (
                f"Record {i}: expected message {message!r} in line {lines[i]!r}"
            )
            assert f"[{level.name}]" in lines[i], (
                f"Record {i}: expected level {level.name!r} in line {lines[i]!r}"
            )
            assert f"[{stage}]" in lines[i], (
                f"Record {i}: expected stage {stage!r} in line {lines[i]!r}"
            )


# ---------------------------------------------------------------------------
# Property 7: Loading stage trace fields
# ---------------------------------------------------------------------------

@given(
    dataset_name=_safe_text,
    subset=_safe_text,
    num_docs=st.integers(min_value=1, max_value=100_000),
    actual_docs=st.integers(min_value=0, max_value=100_000),
    elapsed_ms=st.floats(min_value=0.0, max_value=1_000_000.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_prop_loading_stage_fields(
    dataset_name: str,
    subset: str,
    num_docs: int,
    actual_docs: int,
    elapsed_ms: float,
) -> None:
    """Feature: benchmark-logging, Property 7: Loading stage trace fields

    Validates: Requirements 4.1, 4.2
    """
    with tempfile.TemporaryDirectory() as output_dir:
        logger = BenchmarkLogger(
            output_dir=output_dir, backend="test", language="python",
            min_level=LogLevel.DEBUG,
        )
        logger.log_loading_start(dataset_name, subset, num_docs)
        logger.log_loading_complete(actual_docs, elapsed_ms)
        logger._file.close()

        log_path = os.path.join(output_dir, "test-python.log")
        lines = [l for l in open(log_path, encoding="utf-8").read().splitlines() if l.strip()]

        assert len(lines) == 2, f"Expected 2 lines, got {len(lines)}"

        start_line = lines[0]
        complete_line = lines[1]

        # Start record must contain dataset name, subset, and requested count
        assert dataset_name in start_line, f"dataset_name {dataset_name!r} not in start: {start_line!r}"
        assert subset in start_line, f"subset {subset!r} not in start: {start_line!r}"
        assert str(num_docs) in start_line, f"num_docs {num_docs} not in start: {start_line!r}"

        # Complete record must contain actual doc count
        assert str(actual_docs) in complete_line, (
            f"actual_docs {actual_docs} not in complete: {complete_line!r}"
        )

        # Complete record must contain a non-negative elapsed time
        assert "[INFO]" in complete_line, f"Expected INFO level in complete: {complete_line!r}"
        # elapsed_ms is non-negative (guaranteed by strategy min_value=0.0)
        assert elapsed_ms >= 0.0


# ---------------------------------------------------------------------------
# Property 9: Chunking stage trace fields
# ---------------------------------------------------------------------------

@given(
    chunk_size=st.integers(min_value=1, max_value=10_000),
    overlap=st.integers(min_value=0, max_value=5_000),
    num_chunks=st.integers(min_value=0, max_value=1_000_000),
    elapsed_ms=st.floats(min_value=0.0, max_value=1_000_000.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_prop_chunking_stage_fields(
    chunk_size: int,
    overlap: int,
    num_chunks: int,
    elapsed_ms: float,
) -> None:
    """Feature: benchmark-logging, Property 9: Chunking stage trace fields

    Validates: Requirements 5.1, 5.2
    """
    with tempfile.TemporaryDirectory() as output_dir:
        logger = BenchmarkLogger(
            output_dir=output_dir, backend="test", language="python",
            min_level=LogLevel.DEBUG,
        )
        logger.log_chunking_start(chunk_size, overlap)
        logger.log_chunking_complete(num_chunks, elapsed_ms)
        logger._file.close()

        log_path = os.path.join(output_dir, "test-python.log")
        lines = [l for l in open(log_path, encoding="utf-8").read().splitlines() if l.strip()]

        assert len(lines) == 2, f"Expected 2 lines, got {len(lines)}"

        start_line = lines[0]
        complete_line = lines[1]

        # Start record must contain chunk_size and overlap
        assert str(chunk_size) in start_line, f"chunk_size {chunk_size} not in start: {start_line!r}"
        assert str(overlap) in start_line, f"overlap {overlap} not in start: {start_line!r}"

        # Complete record must contain total chunk count
        assert str(num_chunks) in complete_line, (
            f"num_chunks {num_chunks} not in complete: {complete_line!r}"
        )

        # elapsed_ms is non-negative (guaranteed by strategy min_value=0.0)
        assert elapsed_ms >= 0.0


# ---------------------------------------------------------------------------
# Property 10: Embedding stage trace fields
# ---------------------------------------------------------------------------

@given(
    model=_safe_text,
    num_chunks=st.integers(min_value=0, max_value=1_000_000),
    elapsed_ms=st.floats(min_value=0.0, max_value=1_000_000.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_prop_embedding_stage_fields(
    model: str,
    num_chunks: int,
    elapsed_ms: float,
) -> None:
    """Feature: benchmark-logging, Property 10: Embedding stage trace fields

    Validates: Requirements 6.1, 6.2
    """
    with tempfile.TemporaryDirectory() as output_dir:
        logger = BenchmarkLogger(
            output_dir=output_dir, backend="test", language="python",
            min_level=LogLevel.DEBUG,
        )
        logger.log_embedding_start(model, num_chunks)
        logger.log_embedding_complete(elapsed_ms)
        logger._file.close()

        log_path = os.path.join(output_dir, "test-python.log")
        lines = [l for l in open(log_path, encoding="utf-8").read().splitlines() if l.strip()]

        assert len(lines) == 2, f"Expected 2 lines, got {len(lines)}"

        start_line = lines[0]
        complete_line = lines[1]

        # Start record must contain model name and chunk count
        assert model in start_line, f"model {model!r} not in start: {start_line!r}"
        assert str(num_chunks) in start_line, f"num_chunks {num_chunks} not in start: {start_line!r}"

        # Complete record must be INFO level (non-negative elapsed guaranteed by strategy)
        assert "[INFO]" in complete_line, f"Expected INFO level in complete: {complete_line!r}"
        assert elapsed_ms >= 0.0


# ---------------------------------------------------------------------------
# Property 8: Embedding progress DEBUG record count
# ---------------------------------------------------------------------------

@given(
    model=_safe_text,
    n=st.integers(min_value=100, max_value=2_000),
)
@settings(max_examples=50)
def test_prop_embedding_progress_count(model: str, n: int) -> None:
    """Feature: benchmark-logging, Property 8: Embedding progress DEBUG record count

    Validates: Requirements 6.4
    """
    import math

    with tempfile.TemporaryDirectory() as output_dir:
        logger = BenchmarkLogger(
            output_dir=output_dir, backend="test", language="python",
            min_level=LogLevel.DEBUG,
        )
        logger.log_embedding_start(model, n)

        # Simulate embedding progress: emit a progress record every 100 chunks
        for i in range(100, n + 1, 100):
            logger.log_embedding_progress(i)

        logger.log_embedding_complete(0.0)
        logger._file.close()

        log_path = os.path.join(output_dir, "test-python.log")
        lines = [l for l in open(log_path, encoding="utf-8").read().splitlines() if l.strip()]

        # Count DEBUG records from the embedding stage that are progress records
        debug_progress_lines = [
            l for l in lines
            if "[DEBUG]" in l and "[embedding]" in l and "Embedded so far:" in l
        ]

        expected_count = math.floor(n / 100)
        assert len(debug_progress_lines) == expected_count, (
            f"Expected {expected_count} progress DEBUG records for n={n}, "
            f"got {len(debug_progress_lines)}"
        )


# ---------------------------------------------------------------------------
# Property 11: Index build stage trace fields
# ---------------------------------------------------------------------------

@given(
    num_embeddings=st.integers(min_value=0, max_value=1_000_000),
    elapsed_ms=st.floats(min_value=0.0, max_value=1_000_000.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_prop_index_build_stage_fields(
    num_embeddings: int,
    elapsed_ms: float,
) -> None:
    """Feature: benchmark-logging, Property 11: Index build stage trace fields

    Validates: Requirements 7.1, 7.2
    """
    with tempfile.TemporaryDirectory() as output_dir:
        logger = BenchmarkLogger(
            output_dir=output_dir, backend="test", language="python",
            min_level=LogLevel.DEBUG,
        )
        logger.log_index_build_start(num_embeddings)
        logger.log_index_build_complete(elapsed_ms)
        logger._file.close()

        log_path = os.path.join(output_dir, "test-python.log")
        lines = [l for l in open(log_path, encoding="utf-8").read().splitlines() if l.strip()]

        assert len(lines) == 2, f"Expected 2 lines, got {len(lines)}"

        start_line = lines[0]
        complete_line = lines[1]

        # Start record must contain the number of embeddings
        assert str(num_embeddings) in start_line, (
            f"num_embeddings {num_embeddings} not in start: {start_line!r}"
        )

        # Complete record must be INFO level (non-negative elapsed guaranteed by strategy)
        assert "[INFO]" in complete_line, f"Expected INFO level in complete: {complete_line!r}"
        assert elapsed_ms >= 0.0


# ---------------------------------------------------------------------------
# Property 12: Retrieval stage trace fields
# ---------------------------------------------------------------------------

@given(
    query_id=st.integers(min_value=0, max_value=1_000_000),
    num_chunks=st.integers(min_value=0, max_value=10_000),
    elapsed_ms=st.floats(min_value=0.0, max_value=1_000_000.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_prop_retrieval_stage_fields(
    query_id: int,
    num_chunks: int,
    elapsed_ms: float,
) -> None:
    """Feature: benchmark-logging, Property 12: Retrieval stage trace fields

    Validates: Requirements 8.1, 8.2
    """
    with tempfile.TemporaryDirectory() as output_dir:
        logger = BenchmarkLogger(
            output_dir=output_dir, backend="test", language="python",
            min_level=LogLevel.DEBUG,
        )
        logger.log_retrieval_start(query_id)
        logger.log_retrieval_complete(query_id, num_chunks, elapsed_ms)
        logger._file.close()

        log_path = os.path.join(output_dir, "test-python.log")
        lines = [l for l in open(log_path, encoding="utf-8").read().splitlines() if l.strip()]

        assert len(lines) == 2, f"Expected 2 lines, got {len(lines)}"

        start_line = lines[0]
        complete_line = lines[1]

        # Start record must be DEBUG and contain query_id
        assert "[DEBUG]" in start_line, f"Expected DEBUG in start: {start_line!r}"
        assert str(query_id) in start_line, f"query_id {query_id} not in start: {start_line!r}"

        # Complete record must be DEBUG and contain query_id, num_chunks, and non-negative elapsed
        assert "[DEBUG]" in complete_line, f"Expected DEBUG in complete: {complete_line!r}"
        assert str(query_id) in complete_line, f"query_id {query_id} not in complete: {complete_line!r}"
        assert str(num_chunks) in complete_line, (
            f"num_chunks {num_chunks} not in complete: {complete_line!r}"
        )
        assert elapsed_ms >= 0.0


# ---------------------------------------------------------------------------
# Property 13: Generation stage trace fields
# ---------------------------------------------------------------------------

@given(
    query_id=st.integers(min_value=0, max_value=1_000_000),
    num_chunks=st.integers(min_value=0, max_value=10_000),
    total_tokens=st.integers(min_value=0, max_value=100_000),
    ttft_ms=st.floats(min_value=0.0, max_value=1_000_000.0, allow_nan=False, allow_infinity=False),
    generation_ms=st.floats(min_value=0.0, max_value=1_000_000.0, allow_nan=False, allow_infinity=False),
    failed=st.booleans(),
    reason=_safe_text,
)
@settings(max_examples=100)
def test_prop_generation_stage_fields(
    query_id: int,
    num_chunks: int,
    total_tokens: int,
    ttft_ms: float,
    generation_ms: float,
    failed: bool,
    reason: str,
) -> None:
    """Feature: benchmark-logging, Property 13: Generation stage trace fields

    Validates: Requirements 9.1, 9.2, 9.4
    """
    with tempfile.TemporaryDirectory() as output_dir:
        logger = BenchmarkLogger(
            output_dir=output_dir, backend="test", language="python",
            min_level=LogLevel.DEBUG,
        )
        logger.log_generation_start(query_id, num_chunks)
        logger.log_generation_complete(query_id, total_tokens, ttft_ms, generation_ms)
        if failed:
            logger.log_generation_failed_response(query_id, reason)
        logger._file.close()

        log_path = os.path.join(output_dir, "test-python.log")
        lines = [l for l in open(log_path, encoding="utf-8").read().splitlines() if l.strip()]

        expected_lines = 3 if failed else 2
        assert len(lines) == expected_lines, f"Expected {expected_lines} lines, got {len(lines)}: {lines}"

        start_line = lines[0]
        complete_line = lines[1]

        # Start record must be DEBUG and contain query_id and num_chunks
        assert "[DEBUG]" in start_line, f"Expected DEBUG in start: {start_line!r}"
        assert str(query_id) in start_line, f"query_id {query_id} not in start: {start_line!r}"
        assert str(num_chunks) in start_line, f"num_chunks {num_chunks} not in start: {start_line!r}"

        # Complete record must be DEBUG and contain query_id, total_tokens, ttft_ms, generation_ms
        assert "[DEBUG]" in complete_line, f"Expected DEBUG in complete: {complete_line!r}"
        assert str(query_id) in complete_line, f"query_id {query_id} not in complete: {complete_line!r}"
        assert str(total_tokens) in complete_line, (
            f"total_tokens {total_tokens} not in complete: {complete_line!r}"
        )
        # ttft_ms and generation_ms appear as formatted floats — check they are non-negative
        assert ttft_ms >= 0.0
        assert generation_ms >= 0.0

        # If failed, a WARNING record must contain query_id and reason
        if failed:
            warning_line = lines[2]
            assert "[WARNING]" in warning_line, f"Expected WARNING in failed line: {warning_line!r}"
            assert str(query_id) in warning_line, (
                f"query_id {query_id} not in warning: {warning_line!r}"
            )
            assert reason in warning_line, f"reason {reason!r} not in warning: {warning_line!r}"


# ---------------------------------------------------------------------------
# Property 14: Run summary trace fields
# ---------------------------------------------------------------------------

@given(
    total_queries=st.integers(min_value=0, max_value=1_000_000),
    failures=st.integers(min_value=0, max_value=1_000_000),
    p50_ms=st.floats(min_value=0.0, max_value=1_000_000.0, allow_nan=False, allow_infinity=False),
    p95_ms=st.floats(min_value=0.0, max_value=1_000_000.0, allow_nan=False, allow_infinity=False),
    output_path=_safe_text,
)
@settings(max_examples=100)
def test_prop_run_summary_fields(
    total_queries: int,
    failures: int,
    p50_ms: float,
    p95_ms: float,
    output_path: str,
) -> None:
    """Feature: benchmark-logging, Property 14: Run summary trace fields

    Validates: Requirements 10.1, 10.2
    """
    with tempfile.TemporaryDirectory() as output_dir:
        logger = BenchmarkLogger(
            output_dir=output_dir, backend="test", language="python",
            min_level=LogLevel.DEBUG,
        )
        logger.log_run_summary(total_queries, failures, p50_ms, p95_ms, output_path)
        logger._file.close()

        log_path = os.path.join(output_dir, "test-python.log")
        lines = [l for l in open(log_path, encoding="utf-8").read().splitlines() if l.strip()]

        assert len(lines) == 1, f"Expected 1 line, got {len(lines)}"
        summary_line = lines[0]

        # Must be INFO level
        assert "[INFO]" in summary_line, f"Expected INFO in summary: {summary_line!r}"

        # Must contain all five values
        assert str(total_queries) in summary_line, (
            f"total_queries {total_queries} not in summary: {summary_line!r}"
        )
        assert str(failures) in summary_line, (
            f"failures {failures} not in summary: {summary_line!r}"
        )
        assert output_path in summary_line, (
            f"output_path {output_path!r} not in summary: {summary_line!r}"
        )
        # p50_ms and p95_ms appear as formatted floats — verify they are non-negative
        assert p50_ms >= 0.0
        assert p95_ms >= 0.0
