"""Report generator — Task 11 implementation.

Reads metrics_python.jsonl and metrics_rust.jsonl and produces benchmark_report.md
with a summary table, per-query latency plots, and optional high-failure-rate warnings.
"""
from __future__ import annotations

import json
import os
import statistics
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: str) -> tuple[list[dict], dict, dict]:
    """Load query records, summary record, and stress_summary record from a JSONL file.

    Returns (query_records, summary_record, stress_summary_record).
    Raises FileNotFoundError with a descriptive message if the file is missing.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Metrics file not found: '{path}'. "
            "Ensure the pipeline has been run before generating the report."
        )

    queries: list[dict] = []
    summary: dict = {}
    stress_summary: dict = {}

    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("type") == "query":
                queries.append(obj)
            elif obj.get("type") == "summary":
                summary = obj
            elif obj.get("type") == "stress_summary":
                stress_summary = obj

    return queries, summary, stress_summary


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _save_histogram(
    python_latencies: list[float],
    rust_latencies: list[float],
    output_dir: Path,
    filename: str = "latency_histogram.png",
) -> str:
    """Save a per-query latency histogram PNG and return the filename."""
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = 30
    if python_latencies:
        ax.hist(python_latencies, bins=bins, alpha=0.6, label="Python", color="steelblue")
    if rust_latencies:
        ax.hist(rust_latencies, bins=bins, alpha=0.6, label="Rust", color="darkorange")
    ax.set_xlabel("End-to-end latency (ms)")
    ax.set_ylabel("Query count")
    ax.set_title("Per-query latency distribution")
    ax.legend()
    fig.tight_layout()
    out_path = output_dir / filename
    fig.savefig(str(out_path), dpi=100)
    plt.close(fig)
    return filename


def _save_cdf(
    python_latencies: list[float],
    rust_latencies: list[float],
    output_dir: Path,
    filename: str = "latency_cdf.png",
) -> str:
    """Save a per-query latency CDF PNG and return the filename."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for latencies, label, color in [
        (python_latencies, "Python", "steelblue"),
        (rust_latencies, "Rust", "darkorange"),
    ]:
        if latencies:
            sorted_lats = sorted(latencies)
            cdf = np.arange(1, len(sorted_lats) + 1) / len(sorted_lats)
            ax.plot(sorted_lats, cdf, label=label, color=color)

    ax.set_xlabel("End-to-end latency (ms)")
    ax.set_ylabel("Cumulative probability")
    ax.set_title("Per-query latency CDF")
    ax.legend()
    fig.tight_layout()
    out_path = output_dir / filename
    fig.savefig(str(out_path), dpi=100)
    plt.close(fig)
    return filename


# ---------------------------------------------------------------------------
# Stress test section builder
# ---------------------------------------------------------------------------

def _build_stress_section(
    py_stress: dict,
    rs_stress: dict,
    llm_backend: str,
) -> list[str]:
    """Build the stress test section markdown lines.

    When only one side has stress data (e.g. llm_rs mode with only Rust),
    the table shows only that pipeline's column — no Delta/Delta% columns.
    """
    has_py = bool(py_stress)
    has_rs = bool(rs_stress)

    if not has_py and not has_rs:
        return []

    lines: list[str] = [
        "",
        "## Stress Test Results",
        "",
    ]

    if has_py and has_rs:
        # Two-pipeline table with Delta columns
        def _srow(label: str, py_val: float, rs_val: float, fmt: str = ".2f") -> str:
            delta = rs_val - py_val
            delta_pct = (delta / py_val * 100.0) if py_val != 0.0 else float("nan")
            delta_pct_str = f"{delta_pct:.2f}%" if delta_pct == delta_pct else "N/A"
            return (
                f"| {label} | {py_val:{fmt}} | {rs_val:{fmt}} "
                f"| {delta:+{fmt}} | {delta_pct_str} |"
            )

        def _srow_int(label: str, py_val: float, rs_val: float) -> str:
            delta = rs_val - py_val
            delta_pct = (delta / py_val * 100.0) if py_val != 0.0 else float("nan")
            delta_pct_str = f"{delta_pct:.2f}%" if delta_pct == delta_pct else "N/A"
            return (
                f"| {label} | {int(py_val)} | {int(rs_val)} "
                f"| {int(delta):+d} | {delta_pct_str} |"
            )

        lines += [
            "| Metric | Python value | Rust value | Delta | Delta % |",
            "| --- | --- | --- | --- | --- |",
            _srow("throughput (QPS)", py_stress.get("queries_per_second", 0.0), rs_stress.get("queries_per_second", 0.0)),
            _srow("peak RSS (MB)", py_stress.get("peak_rss_mb", 0.0), rs_stress.get("peak_rss_mb", 0.0)),
            _srow("p50 latency (ms)", py_stress.get("p50_latency_ms", 0.0), rs_stress.get("p50_latency_ms", 0.0)),
            _srow("p95 latency (ms)", py_stress.get("p95_latency_ms", 0.0), rs_stress.get("p95_latency_ms", 0.0)),
            _srow("p99 latency (ms)", py_stress.get("p99_latency_ms", 0.0), rs_stress.get("p99_latency_ms", 0.0)),
            _srow_int("failure count", float(py_stress.get("failure_count", 0)), float(rs_stress.get("failure_count", 0))),
        ]
    else:
        # Single-pipeline table (Rust-only for llm_rs mode, or Python-only)
        pipeline_label = "Rust value" if has_rs else "Python value"
        stress = rs_stress if has_rs else py_stress

        def _srow_single(label: str, val: float, fmt: str = ".2f") -> str:
            return f"| {label} | {val:{fmt}} |"

        lines += [
            f"| Metric | {pipeline_label} |",
            "| --- | --- |",
            _srow_single("throughput (QPS)", stress.get("queries_per_second", 0.0)),
            _srow_single("peak RSS (MB)", stress.get("peak_rss_mb", 0.0)),
            _srow_single("p50 latency (ms)", stress.get("p50_latency_ms", 0.0)),
            _srow_single("p95 latency (ms)", stress.get("p95_latency_ms", 0.0)),
            _srow_single("p99 latency (ms)", stress.get("p99_latency_ms", 0.0)),
            f"| failure count | {int(stress.get('failure_count', 0))} |",
        ]

    return lines


# ---------------------------------------------------------------------------
# Single-pipeline report (llm_rs mode — Python pipeline was skipped)
# ---------------------------------------------------------------------------

def _generate_single_pipeline_report(
    rust_jsonl: str,
    output_path: str,
    query_set_size: int = 50,
    llm_backend: str = "llm_rs",
) -> None:
    """Produce a Rust-only report when the Python pipeline was skipped (llm_rs mode).

    The report contains only Rust columns — no Delta or Delta % columns.
    This is an expected, valid state per Requirement 8.6.
    """
    rs_queries, rs_summary, rs_stress = _load_jsonl(rust_jsonl)

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    rs_lats = [q["end_to_end_ms"] for q in rs_queries if not q.get("failed", False)]
    rs_retrieval = [q["retrieval_ms"] for q in rs_queries if not q.get("failed", False)]
    rs_ttft = [q["ttft_ms"] for q in rs_queries if not q.get("failed", False)]
    rs_tokens = [q["total_tokens"] for q in rs_queries if not q.get("failed", False)]

    rs_p50 = rs_summary.get("p50_latency_ms", 0.0)
    rs_p95 = rs_summary.get("p95_latency_ms", 0.0)
    rs_embed = rs_summary.get("embedding_phase_ms", 0.0)
    rs_index = rs_summary.get("index_build_ms", 0.0)
    rs_failures = int(rs_summary.get("failure_count", 0))

    rs_mean_ttft = _mean(rs_ttft)
    rs_mean_retrieval = _mean(rs_retrieval)
    rs_mean_tokens = _mean([float(t) for t in rs_tokens])

    # Generate plots (Rust-only)
    hist_file = _save_histogram([], rs_lats, output_dir)
    cdf_file = _save_cdf([], rs_lats, output_dir)

    def _row_single(label: str, rs_val: float, fmt: str = ".2f") -> str:
        return f"| {label} | {rs_val:{fmt}} |"

    rows = [
        _row_single("p50 end-to-end latency", rs_p50),
        _row_single("p95 end-to-end latency", rs_p95),
        _row_single("mean TTFT", rs_mean_ttft),
        _row_single("mean retrieval latency", rs_mean_retrieval),
        _row_single("mean Total_Tokens", rs_mean_tokens),
        _row_single("embedding phase time", rs_embed),
        _row_single("index build time", rs_index),
        f"| failure count | {rs_failures} |",
    ]

    warnings: list[str] = []
    if query_set_size > 0 and rs_failures / query_set_size > 0.10:
        warnings.append(
            f"> ⚠️ **Warning**: Rust pipeline failure count ({rs_failures}) exceeds 10% of "
            f"query set size ({query_set_size}). Results may not be statistically reliable."
        )

    lines: list[str] = [
        f"# Rust vs Python RAG Benchmark Report — {llm_backend}",
        "",
        "> ℹ️ **Note**: Python pipeline was skipped (llm_rs backend is Rust-only). "
        "This is a single-pipeline report.",
        "",
        "## Summary Table",
        "",
        "| Metric | Rust value |",
        "| --- | --- |",
    ]
    lines.extend(rows)

    if warnings:
        lines.append("")
        lines.append("## Warnings")
        lines.append("")
        lines.extend(warnings)

    lines += [
        "",
        "## Per-Query Latency Distribution",
        "",
        f"![Latency Histogram]({hist_file})",
        "",
        f"![Latency CDF]({cdf_file})",
        "",
    ]

    stress_lines = _build_stress_section({}, rs_stress, llm_backend)
    if stress_lines:
        lines.extend(stress_lines)
        lines.append("")

    report_text = "\n".join(lines)
    Path(output_path).write_text(report_text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_report(
    python_jsonl: str,
    rust_jsonl: str,
    output_path: Optional[str] = None,
    query_set_size: int = 50,
    llm_backend: str = "ollama_http",
) -> None:
    """Read both JSONL files and produce benchmark_report_{llm_backend}.md.

    Output filenames are scoped to the active LLM backend so that successive
    runs with different backends never overwrite each other.

    When ``llm_backend = "llm_rs"`` and only the Rust JSONL is present (the
    Python pipeline was skipped), a single-pipeline report is produced without
    raising an error — this is an expected, valid state per Requirement 8.6.

    Parameters
    ----------
    python_jsonl:
        Path to the Python pipeline metrics JSONL file
        (e.g. ``metrics_python_ollama_http.jsonl``).
    rust_jsonl:
        Path to the Rust pipeline metrics JSONL file
        (e.g. ``metrics_rust_ollama_http.jsonl``).
    output_path:
        Destination path for the generated Markdown report.  When omitted the
        report is written to the same directory as ``rust_jsonl`` under the
        name ``benchmark_report_{llm_backend}.md``.
    query_set_size:
        Total number of queries in the query set (used for failure-rate check).
    llm_backend:
        The active LLM backend value from config (``ollama_http``, ``pytorch``,
        or ``llm_rs``).  Used to derive the default output filename.

    Raises
    ------
    FileNotFoundError
        If either JSONL file does not exist, except when ``llm_backend =
        "llm_rs"`` and only the Python JSONL is absent (single-pipeline mode).
    """
    # Determine single-pipeline mode: llm_rs backend with Python JSONL absent
    python_missing = not Path(python_jsonl).exists()
    single_pipeline_mode = llm_backend == "llm_rs" and python_missing

    if output_path is None:
        # Use the Rust JSONL directory as the base for the report path
        output_path = str(
            Path(rust_jsonl).parent / f"benchmark_report_{llm_backend}.md"
        )

    if single_pipeline_mode:
        return _generate_single_pipeline_report(
            rust_jsonl=rust_jsonl,
            output_path=output_path,
            query_set_size=query_set_size,
            llm_backend=llm_backend,
        )

    py_queries, py_summary, py_stress = _load_jsonl(python_jsonl)
    rs_queries, rs_summary, rs_stress = _load_jsonl(rust_jsonl)

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Extract per-query latencies (successful only for percentiles) ---
    py_lats = [q["end_to_end_ms"] for q in py_queries if not q.get("failed", False)]
    rs_lats = [q["end_to_end_ms"] for q in rs_queries if not q.get("failed", False)]

    py_retrieval = [q["retrieval_ms"] for q in py_queries if not q.get("failed", False)]
    rs_retrieval = [q["retrieval_ms"] for q in rs_queries if not q.get("failed", False)]

    py_ttft = [q["ttft_ms"] for q in py_queries if not q.get("failed", False)]
    rs_ttft = [q["ttft_ms"] for q in rs_queries if not q.get("failed", False)]

    py_tokens = [q["total_tokens"] for q in py_queries if not q.get("failed", False)]
    rs_tokens = [q["total_tokens"] for q in rs_queries if not q.get("failed", False)]

    # --- Summary values ---
    py_p50 = py_summary.get("p50_latency_ms", 0.0)
    py_p95 = py_summary.get("p95_latency_ms", 0.0)
    py_embed = py_summary.get("embedding_phase_ms", 0.0)
    py_index = py_summary.get("index_build_ms", 0.0)
    py_failures = int(py_summary.get("failure_count", 0))

    rs_p50 = rs_summary.get("p50_latency_ms", 0.0)
    rs_p95 = rs_summary.get("p95_latency_ms", 0.0)
    rs_embed = rs_summary.get("embedding_phase_ms", 0.0)
    rs_index = rs_summary.get("index_build_ms", 0.0)
    rs_failures = int(rs_summary.get("failure_count", 0))

    py_mean_ttft = _mean(py_ttft)
    rs_mean_ttft = _mean(rs_ttft)
    py_mean_retrieval = _mean(py_retrieval)
    rs_mean_retrieval = _mean(rs_retrieval)
    py_mean_tokens = _mean([float(t) for t in py_tokens])
    rs_mean_tokens = _mean([float(t) for t in rs_tokens])

    # --- Generate plots ---
    hist_file = _save_histogram(py_lats, rs_lats, output_dir)
    cdf_file = _save_cdf(py_lats, rs_lats, output_dir)

    # --- Build table rows ---
    def _row(label: str, py_val: float, rs_val: float, fmt: str = ".2f") -> str:
        delta = rs_val - py_val
        delta_pct = (delta / py_val * 100.0) if py_val != 0.0 else float("nan")
        delta_pct_str = f"{delta_pct:.2f}%" if not (delta_pct != delta_pct) else "N/A"
        return (
            f"| {label} | {py_val:{fmt}} | {rs_val:{fmt}} "
            f"| {delta:+{fmt}} | {delta_pct_str} |"
        )

    def _row_int(label: str, py_val: float, rs_val: float) -> str:
        delta = rs_val - py_val
        delta_pct = (delta / py_val * 100.0) if py_val != 0.0 else float("nan")
        delta_pct_str = f"{delta_pct:.2f}%" if not (delta_pct != delta_pct) else "N/A"
        return (
            f"| {label} | {int(py_val)} | {int(rs_val)} "
            f"| {int(delta):+d} | {delta_pct_str} |"
        )

    rows = [
        _row("p50 end-to-end latency", py_p50, rs_p50),
        _row("p95 end-to-end latency", py_p95, rs_p95),
        _row("mean TTFT", py_mean_ttft, rs_mean_ttft),
        _row("mean retrieval latency", py_mean_retrieval, rs_mean_retrieval),
        _row("mean Total_Tokens", py_mean_tokens, rs_mean_tokens),
        _row("embedding phase time", py_embed, rs_embed),
        _row("index build time", py_index, rs_index),
        _row_int("failure count", float(py_failures), float(rs_failures)),
    ]

    # --- Warnings ---
    warnings: list[str] = []
    if query_set_size > 0:
        if py_failures / query_set_size > 0.10:
            warnings.append(
                f"> ⚠️ **Warning**: Python pipeline failure count ({py_failures}) exceeds 10% of "
                f"query set size ({query_set_size}). Results may not be statistically reliable."
            )
        if rs_failures / query_set_size > 0.10:
            warnings.append(
                f"> ⚠️ **Warning**: Rust pipeline failure count ({rs_failures}) exceeds 10% of "
                f"query set size ({query_set_size}). Results may not be statistically reliable."
            )

    # --- Assemble report ---
    lines: list[str] = [
        f"# Rust vs Python RAG Benchmark Report — {llm_backend}",
        "",
        "## Summary Table",
        "",
        "| Metric | Python value | Rust value | Delta | Delta % |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines.extend(rows)

    if warnings:
        lines.append("")
        lines.append("## Warnings")
        lines.append("")
        lines.extend(warnings)

    lines += [
        "",
        "## Per-Query Latency Distribution",
        "",
        f"![Latency Histogram]({hist_file})",
        "",
        f"![Latency CDF]({cdf_file})",
        "",
    ]

    # --- Stress test section (appended after main table if stress data present) ---
    stress_lines = _build_stress_section(py_stress, rs_stress, llm_backend)
    if stress_lines:
        lines.extend(stress_lines)
        lines.append("")

    report_text = "\n".join(lines)
    Path(output_path).write_text(report_text, encoding="utf-8")


if __name__ == "__main__":
    import argparse
    import sys

    try:
        import tomllib  # Python 3.11+
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            tomllib = None  # type: ignore[assignment]

    parser = argparse.ArgumentParser(description="Generate benchmark report from JSONL metrics files.")
    parser.add_argument("--python-jsonl", default=None, help="Path to Python pipeline JSONL metrics file")
    parser.add_argument("--rust-jsonl", default=None, help="Path to Rust pipeline JSONL metrics file")
    parser.add_argument("--output", default=None, help="Output path for the Markdown report")
    parser.add_argument("--llm-backend", default=None, help="LLM backend name (e.g. ollama_http)")
    parser.add_argument("--query-set-size", type=int, default=50, help="Total number of queries in the query set")
    args = parser.parse_args()

    # Resolve llm_backend from config if not provided
    llm_backend = args.llm_backend
    if llm_backend is None:
        config_path = "benchmark_config.toml"
        if tomllib is not None and Path(config_path).exists():
            with open(config_path, "rb") as f:
                cfg = tomllib.load(f)
            llm_backend = cfg.get("llm_backend", "ollama_http")
        else:
            llm_backend = "ollama_http"

    # Resolve output_dir from config if JSONL paths not provided
    output_dir = "output/"
    if tomllib is not None and Path("benchmark_config.toml").exists():
        with open("benchmark_config.toml", "rb") as f:
            cfg = tomllib.load(f)
        output_dir = cfg.get("output_dir", "output/")

    python_jsonl = args.python_jsonl or f"{output_dir}metrics_python_{llm_backend}.jsonl"
    rust_jsonl = args.rust_jsonl or f"{output_dir}metrics_rust_{llm_backend}.jsonl"
    output_path = args.output or f"{output_dir}benchmark_report_{llm_backend}.md"

    try:
        generate_report(
            python_jsonl=python_jsonl,
            rust_jsonl=rust_jsonl,
            output_path=output_path,
            query_set_size=args.query_set_size,
            llm_backend=llm_backend,
        )
        print(f"Report written to {output_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
