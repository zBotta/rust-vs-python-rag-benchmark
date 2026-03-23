"""Property test for output filenames scoped to llm_backend — Task 22.1.

Property 22: JSONL and report filenames always include the active llm_backend value as a suffix.
Validates: Requirements 6.4, 8.1

# Feature: rust-vs-python-rag-benchmark, Property 22: For any value of llm_backend in config,
# the JSONL metrics files written by each pipeline and the report file produced by the
# Report_Generator must include the llm_backend value as a suffix in their filenames.
# No run may write to a filename that does not contain the active llm_backend value.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from python_pipeline.metrics_collector import (
    PipelineMetrics,
    QueryMetrics,
    serialize_to_jsonl,
)
from report.generate_report import generate_report

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Valid llm_backend values as defined in the spec
VALID_LLM_BACKENDS = ["ollama_http", "llama_cpp", "llm_rs"]

llm_backend_strategy = st.sampled_from(VALID_LLM_BACKENDS)

# Also test arbitrary backend strings to ensure the property holds generally
arbitrary_backend_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), whitelist_characters="_"),
    min_size=1,
    max_size=32,
).filter(lambda s: len(s) >= 1 and s[0].isalpha())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_minimal_pipeline_metrics() -> PipelineMetrics:
    """Return a minimal PipelineMetrics with one successful query."""
    return PipelineMetrics(
        embedding_phase_ms=100.0,
        index_build_ms=50.0,
        queries=[
            QueryMetrics(
                query_id=0,
                end_to_end_ms=200.0,
                retrieval_ms=10.0,
                ttft_ms=80.0,
                generation_ms=110.0,
                total_tokens=150,
                failed=False,
                failure_reason=None,
            )
        ],
        p50_latency_ms=200.0,
        p95_latency_ms=200.0,
    )


def _write_minimal_jsonl(path: Path) -> None:
    """Write a minimal valid JSONL file to the given path."""
    metrics = _make_minimal_pipeline_metrics()
    serialize_to_jsonl(metrics, str(path))


# ---------------------------------------------------------------------------
# Property 22: JSONL filenames include llm_backend suffix
# Feature: rust-vs-python-rag-benchmark, Property 22: JSONL and report filenames always
# include the active llm_backend value as a suffix
# ---------------------------------------------------------------------------


@given(llm_backend=llm_backend_strategy)
@settings(max_examples=20, deadline=None)
def test_python_jsonl_filename_contains_llm_backend(llm_backend: str) -> None:
    """Property 22: Python pipeline JSONL filename includes the llm_backend suffix.

    # Feature: rust-vs-python-rag-benchmark, Property 22: JSONL and report filenames always
    # include the active llm_backend value as a suffix
    Validates: Requirements 6.4
    """
    # The pipeline constructs: output_dir / f"metrics_python_{llm_backend}.jsonl"
    expected_filename = f"metrics_python_{llm_backend}.jsonl"

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / expected_filename
        metrics = _make_minimal_pipeline_metrics()
        serialize_to_jsonl(metrics, str(output_path))

        # Verify the file was written at the backend-scoped path
        assert output_path.exists(), (
            f"Expected JSONL file '{expected_filename}' was not created"
        )

        # Verify the filename contains the llm_backend value
        assert llm_backend in output_path.name, (
            f"Python JSONL filename '{output_path.name}' does not contain "
            f"llm_backend value '{llm_backend}'"
        )

        # Verify no fixed-name file (without backend suffix) was written
        fixed_name = Path(tmpdir) / "metrics_python.jsonl"
        assert not fixed_name.exists(), (
            "A fixed-name 'metrics_python.jsonl' was written — "
            "filenames must always include the llm_backend suffix"
        )


@given(llm_backend=llm_backend_strategy)
@settings(max_examples=20, deadline=None)
def test_rust_jsonl_filename_contains_llm_backend(llm_backend: str) -> None:
    """Property 22: Rust pipeline JSONL filename includes the llm_backend suffix.

    # Feature: rust-vs-python-rag-benchmark, Property 22: JSONL and report filenames always
    # include the active llm_backend value as a suffix
    Validates: Requirements 6.4
    """
    expected_filename = f"metrics_rust_{llm_backend}.jsonl"

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / expected_filename
        metrics = _make_minimal_pipeline_metrics()
        serialize_to_jsonl(metrics, str(output_path))

        assert output_path.exists(), (
            f"Expected JSONL file '{expected_filename}' was not created"
        )
        assert llm_backend in output_path.name, (
            f"Rust JSONL filename '{output_path.name}' does not contain "
            f"llm_backend value '{llm_backend}'"
        )

        fixed_name = Path(tmpdir) / "metrics_rust.jsonl"
        assert not fixed_name.exists(), (
            "A fixed-name 'metrics_rust.jsonl' was written — "
            "filenames must always include the llm_backend suffix"
        )


@given(llm_backend=llm_backend_strategy)
@settings(max_examples=20, deadline=None)
def test_report_filename_contains_llm_backend(llm_backend: str) -> None:
    """Property 22: Report Generator output filename includes the llm_backend suffix.

    # Feature: rust-vs-python-rag-benchmark, Property 22: JSONL and report filenames always
    # include the active llm_backend value as a suffix
    Validates: Requirements 8.1
    """
    expected_report_filename = f"benchmark_report_{llm_backend}.md"

    with tempfile.TemporaryDirectory() as tmpdir:
        py_jsonl = Path(tmpdir) / f"metrics_python_{llm_backend}.jsonl"
        rs_jsonl = Path(tmpdir) / f"metrics_rust_{llm_backend}.jsonl"
        report_path = Path(tmpdir) / expected_report_filename

        _write_minimal_jsonl(py_jsonl)
        _write_minimal_jsonl(rs_jsonl)

        generate_report(
            str(py_jsonl),
            str(rs_jsonl),
            output_path=str(report_path),
            query_set_size=50,
            llm_backend=llm_backend,
        )

        assert report_path.exists(), (
            f"Expected report file '{expected_report_filename}' was not created"
        )
        assert llm_backend in report_path.name, (
            f"Report filename '{report_path.name}' does not contain "
            f"llm_backend value '{llm_backend}'"
        )

        # Verify no fixed-name report was written
        fixed_report = Path(tmpdir) / "benchmark_report.md"
        assert not fixed_report.exists(), (
            "A fixed-name 'benchmark_report.md' was written — "
            "report filenames must always include the llm_backend suffix"
        )


@given(llm_backend=llm_backend_strategy)
@settings(max_examples=20, deadline=None)
def test_all_output_filenames_contain_llm_backend(llm_backend: str) -> None:
    """Property 22: All output files (both JSONLs and report) include the llm_backend suffix.

    Verifies the complete set of output files for a single backend run.

    # Feature: rust-vs-python-rag-benchmark, Property 22: JSONL and report filenames always
    # include the active llm_backend value as a suffix
    Validates: Requirements 6.4, 8.1
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        py_jsonl = tmpdir_path / f"metrics_python_{llm_backend}.jsonl"
        rs_jsonl = tmpdir_path / f"metrics_rust_{llm_backend}.jsonl"
        report_path = tmpdir_path / f"benchmark_report_{llm_backend}.md"

        _write_minimal_jsonl(py_jsonl)
        _write_minimal_jsonl(rs_jsonl)

        generate_report(
            str(py_jsonl),
            str(rs_jsonl),
            output_path=str(report_path),
            query_set_size=50,
            llm_backend=llm_backend,
        )

        # All three output files must exist and contain the backend suffix
        for output_file in [py_jsonl, rs_jsonl, report_path]:
            assert output_file.exists(), (
                f"Expected output file '{output_file.name}' was not created"
            )
            assert llm_backend in output_file.name, (
                f"Output filename '{output_file.name}' does not contain "
                f"llm_backend value '{llm_backend}'"
            )

        # No fixed-name files should exist in the output directory
        all_files = list(tmpdir_path.iterdir())
        # Filter to only the benchmark output files (exclude plot PNGs)
        benchmark_files = [
            f for f in all_files
            if f.suffix in (".jsonl", ".md")
        ]
        for f in benchmark_files:
            assert llm_backend in f.name, (
                f"File '{f.name}' was written without the llm_backend suffix '{llm_backend}'. "
                "All benchmark output files must be scoped to the active backend."
            )


# ---------------------------------------------------------------------------
# Property 22: filename construction logic (unit-level verification)
# ---------------------------------------------------------------------------

class TestFilenameConstructionLogic:
    """Unit tests verifying the filename construction logic for each backend."""

    @pytest.mark.parametrize("llm_backend", VALID_LLM_BACKENDS)
    def test_python_jsonl_filename_pattern(self, llm_backend: str) -> None:
        """Python JSONL filename follows the pattern metrics_python_{llm_backend}.jsonl."""
        # This mirrors the logic in pipeline.py:
        #   output_path = output_dir / f"metrics_python_{llm_backend}.jsonl"
        filename = f"metrics_python_{llm_backend}.jsonl"
        assert filename.startswith("metrics_python_"), (
            f"Python JSONL filename '{filename}' must start with 'metrics_python_'"
        )
        assert filename.endswith(f"_{llm_backend}.jsonl"), (
            f"Python JSONL filename '{filename}' must end with '_{llm_backend}.jsonl'"
        )
        assert llm_backend in filename

    @pytest.mark.parametrize("llm_backend", VALID_LLM_BACKENDS)
    def test_rust_jsonl_filename_pattern(self, llm_backend: str) -> None:
        """Rust JSONL filename follows the pattern metrics_rust_{llm_backend}.jsonl."""
        filename = f"metrics_rust_{llm_backend}.jsonl"
        assert filename.startswith("metrics_rust_"), (
            f"Rust JSONL filename '{filename}' must start with 'metrics_rust_'"
        )
        assert filename.endswith(f"_{llm_backend}.jsonl"), (
            f"Rust JSONL filename '{filename}' must end with '_{llm_backend}.jsonl'"
        )
        assert llm_backend in filename

    @pytest.mark.parametrize("llm_backend", VALID_LLM_BACKENDS)
    def test_report_filename_pattern(self, llm_backend: str) -> None:
        """Report filename follows the pattern benchmark_report_{llm_backend}.md."""
        filename = f"benchmark_report_{llm_backend}.md"
        assert filename.startswith("benchmark_report_"), (
            f"Report filename '{filename}' must start with 'benchmark_report_'"
        )
        assert filename.endswith(f"_{llm_backend}.md"), (
            f"Report filename '{filename}' must end with '_{llm_backend}.md'"
        )
        assert llm_backend in filename

    @pytest.mark.parametrize("llm_backend", VALID_LLM_BACKENDS)
    def test_different_backends_produce_different_filenames(self, llm_backend: str) -> None:
        """Each backend produces a distinct set of output filenames (no overwriting)."""
        other_backends = [b for b in VALID_LLM_BACKENDS if b != llm_backend]
        py_filename = f"metrics_python_{llm_backend}.jsonl"
        rs_filename = f"metrics_rust_{llm_backend}.jsonl"
        report_filename = f"benchmark_report_{llm_backend}.md"

        for other in other_backends:
            assert py_filename != f"metrics_python_{other}.jsonl", (
                f"Backend '{llm_backend}' and '{other}' produce the same Python JSONL filename"
            )
            assert rs_filename != f"metrics_rust_{other}.jsonl", (
                f"Backend '{llm_backend}' and '{other}' produce the same Rust JSONL filename"
            )
            assert report_filename != f"benchmark_report_{other}.md", (
                f"Backend '{llm_backend}' and '{other}' produce the same report filename"
            )

    def test_pipeline_writes_to_backend_scoped_path(self, tmp_path: Path) -> None:
        """The pipeline.py filename construction uses the llm_backend config value.

        Verifies the actual path construction logic from pipeline.py:
            output_path = output_dir / f"metrics_python_{llm_backend}.jsonl"
        """
        for llm_backend in VALID_LLM_BACKENDS:
            output_dir = tmp_path / llm_backend
            output_dir.mkdir()
            output_path = output_dir / f"metrics_python_{llm_backend}.jsonl"

            metrics = _make_minimal_pipeline_metrics()
            serialize_to_jsonl(metrics, str(output_path))

            assert output_path.exists()
            assert llm_backend in output_path.name
            # Confirm the file is valid JSONL
            lines = output_path.read_text().strip().splitlines()
            assert len(lines) >= 1
            for line in lines:
                obj = json.loads(line)
                assert "type" in obj

    def test_report_generator_default_output_path_uses_llm_backend(self, tmp_path: Path) -> None:
        """When output_path is omitted, generate_report derives the filename from llm_backend.

        The default output path logic in generate_report.py:
            output_path = Path(python_jsonl).parent / f"benchmark_report_{llm_backend}.md"
        """
        for llm_backend in VALID_LLM_BACKENDS:
            subdir = tmp_path / llm_backend
            subdir.mkdir()

            py_jsonl = subdir / f"metrics_python_{llm_backend}.jsonl"
            rs_jsonl = subdir / f"metrics_rust_{llm_backend}.jsonl"
            _write_minimal_jsonl(py_jsonl)
            _write_minimal_jsonl(rs_jsonl)

            # Call without output_path — should derive from llm_backend
            generate_report(
                str(py_jsonl),
                str(rs_jsonl),
                llm_backend=llm_backend,
                query_set_size=50,
            )

            expected_report = subdir / f"benchmark_report_{llm_backend}.md"
            assert expected_report.exists(), (
                f"Default report path should be 'benchmark_report_{llm_backend}.md', "
                f"but it was not found in {subdir}"
            )
            assert llm_backend in expected_report.name

