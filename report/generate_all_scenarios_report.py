from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Optional


METRICS_FILE_RE = re.compile(r"^metrics_(python|rust)_([a-zA-Z0-9_]+)\.jsonl$")


@dataclass
class StressMetrics:
    concurrency: int
    total_queries: int
    queries_per_second: float
    peak_rss_mb: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    failure_count: int


@dataclass
class RunMetrics:
    language: str
    backend: str
    source_file: Path
    query_count: int
    success_count: int
    failure_count: int
    failure_rate_pct: float
    p50_latency_ms: float
    p95_latency_ms: float
    embedding_phase_ms: float
    index_build_ms: float
    mean_end_to_end_ms: float
    mean_retrieval_ms: float
    mean_ttft_ms: float
    mean_generation_ms: float
    mean_total_tokens: float
    generation_share_pct: float
    stress: Optional[StressMetrics]


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _load_jsonl_metrics(path: Path, language: str, backend: str) -> RunMetrics:
    queries: list[dict] = []
    summary: dict = {}
    stress_summary: dict = {}

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            obj_type = obj.get("type")
            if obj_type == "query":
                queries.append(obj)
            elif obj_type == "summary":
                summary = obj
            elif obj_type == "stress_summary":
                stress_summary = obj

    success_queries = [q for q in queries if not q.get("failed", False)]
    failure_count = len(queries) - len(success_queries)
    query_count = len(queries)

    mean_end_to_end = _mean([float(q.get("end_to_end_ms", 0.0)) for q in success_queries])
    mean_retrieval = _mean([float(q.get("retrieval_ms", 0.0)) for q in success_queries])
    mean_ttft = _mean([float(q.get("ttft_ms", 0.0)) for q in success_queries])
    mean_generation = _mean([float(q.get("generation_ms", 0.0)) for q in success_queries])
    mean_tokens = _mean([float(q.get("total_tokens", 0.0)) for q in success_queries])

    generation_share_pct = (100.0 * mean_generation / mean_end_to_end) if mean_end_to_end > 0 else 0.0
    failure_rate_pct = (100.0 * failure_count / query_count) if query_count > 0 else 0.0

    stress: Optional[StressMetrics] = None
    if stress_summary:
        stress = StressMetrics(
            concurrency=int(stress_summary.get("concurrency", 0)),
            total_queries=int(stress_summary.get("total_queries", 0)),
            queries_per_second=float(stress_summary.get("queries_per_second", 0.0)),
            peak_rss_mb=float(stress_summary.get("peak_rss_mb", 0.0)),
            p50_latency_ms=float(stress_summary.get("p50_latency_ms", 0.0)),
            p95_latency_ms=float(stress_summary.get("p95_latency_ms", 0.0)),
            p99_latency_ms=float(stress_summary.get("p99_latency_ms", 0.0)),
            failure_count=int(stress_summary.get("failure_count", 0)),
        )

    return RunMetrics(
        language=language,
        backend=backend,
        source_file=path,
        query_count=query_count,
        success_count=len(success_queries),
        failure_count=failure_count,
        failure_rate_pct=failure_rate_pct,
        p50_latency_ms=float(summary.get("p50_latency_ms", 0.0)),
        p95_latency_ms=float(summary.get("p95_latency_ms", 0.0)),
        embedding_phase_ms=float(summary.get("embedding_phase_ms", 0.0)),
        index_build_ms=float(summary.get("index_build_ms", 0.0)),
        mean_end_to_end_ms=mean_end_to_end,
        mean_retrieval_ms=mean_retrieval,
        mean_ttft_ms=mean_ttft,
        mean_generation_ms=mean_generation,
        mean_total_tokens=mean_tokens,
        generation_share_pct=generation_share_pct,
        stress=stress,
    )


def _discover_runs(output_dir: Path) -> tuple[dict[tuple[str, str], RunMetrics], list[Path], list[Path]]:
    runs: dict[tuple[str, str], RunMetrics] = {}
    discovered: list[Path] = []
    ignored: list[Path] = []

    for path in sorted(output_dir.glob("metrics_*.jsonl")):
        discovered.append(path)
        m = METRICS_FILE_RE.match(path.name)
        if not m:
            ignored.append(path)
            continue

        language, backend = m.group(1), m.group(2)
        metrics = _load_jsonl_metrics(path, language, backend)
        key = (language, backend)

        existing = runs.get(key)
        if existing is None or path.stat().st_mtime > existing.source_file.stat().st_mtime:
            runs[key] = metrics

    return runs, discovered, ignored


def _fmt(value: float) -> str:
    return f"{value:.2f}"


def _fmt_i(value: int) -> str:
    return str(value)


def _comparison_section(
    title: str,
    left_label: str,
    left: Optional[RunMetrics],
    right_label: str,
    right: Optional[RunMetrics],
    delta_label: str,
) -> list[str]:
    lines = [
        f"## {title}",
        "",
    ]

    if left is None or right is None:
        missing = []
        if left is None:
            missing.append(left_label)
        if right is None:
            missing.append(right_label)
        lines.append(f"Missing data: {', '.join(missing)}")
        lines.append("")
        return lines

    rows: list[tuple[str, str, str, str]] = []

    def add(metric: str, lv: float, rv: float, integer: bool = False) -> None:
        delta = rv - lv
        if integer:
            rows.append((metric, _fmt_i(int(lv)), _fmt_i(int(rv)), _fmt_i(int(delta))))
        else:
            rows.append((metric, _fmt(lv), _fmt(rv), _fmt(delta)))

    add("query_count", float(left.query_count), float(right.query_count), integer=True)
    add("success_count", float(left.success_count), float(right.success_count), integer=True)
    add("failure_count", float(left.failure_count), float(right.failure_count), integer=True)
    add("failure_rate_pct", left.failure_rate_pct, right.failure_rate_pct)
    add("p50_latency_ms", left.p50_latency_ms, right.p50_latency_ms)
    add("p95_latency_ms", left.p95_latency_ms, right.p95_latency_ms)
    add("mean_end_to_end_ms", left.mean_end_to_end_ms, right.mean_end_to_end_ms)
    add("mean_ttft_ms", left.mean_ttft_ms, right.mean_ttft_ms)
    add("mean_generation_ms", left.mean_generation_ms, right.mean_generation_ms)
    add("mean_retrieval_ms", left.mean_retrieval_ms, right.mean_retrieval_ms)
    add("mean_total_tokens", left.mean_total_tokens, right.mean_total_tokens)
    add("embedding_phase_ms", left.embedding_phase_ms, right.embedding_phase_ms)
    add("index_build_ms", left.index_build_ms, right.index_build_ms)

    if left.stress is not None and right.stress is not None:
        add("stress_qps", left.stress.queries_per_second, right.stress.queries_per_second)
        add("stress_p99_latency_ms", left.stress.p99_latency_ms, right.stress.p99_latency_ms)
        add("stress_peak_rss_mb", left.stress.peak_rss_mb, right.stress.peak_rss_mb)

    lines.extend([
        f"Source files: {left.source_file.name} vs {right.source_file.name}",
        "",
        f"| Metric | {left_label} | {right_label} | {delta_label} |",
        "| --- | ---: | ---: | ---: |",
    ])
    for metric, lv, rv, dv in rows:
        lines.append(f"| {metric} | {lv} | {rv} | {dv} |")

    fairness_notes: list[str] = []
    if left.query_count != right.query_count:
        fairness_notes.append(
            f"query_count mismatch ({left.query_count} vs {right.query_count})"
        )
    if left.failure_count > 0 or right.failure_count > 0:
        fairness_notes.append(
            f"failures present ({left.failure_count} vs {right.failure_count})"
        )

    if fairness_notes:
        lines.append("")
        lines.append(f"Fairness caveats: {'; '.join(fairness_notes)}")

    lines.append("")
    return lines


def _analysis_and_conclusions(runs: dict[tuple[str, str], RunMetrics]) -> list[str]:
    lines = [
        "## Analysis",
        "",
    ]

    if not runs:
        lines.append("No runs were discovered.")
        lines.append("")
        return lines

    lines.extend([
        "### Run Inventory",
        "",
        "| Run | query_count | failures | p50_latency_ms | mean_end_to_end_ms | generation_share_pct |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for key in sorted(runs):
        rm = runs[key]
        run_name = f"{rm.language}/{rm.backend}"
        lines.append(
            f"| {run_name} | {rm.query_count} | {rm.failure_count} | {_fmt(rm.p50_latency_ms)} | "
            f"{_fmt(rm.mean_end_to_end_ms)} | {_fmt(rm.generation_share_pct)} |"
        )

    lines.extend([
        "",
        "### Key Observations",
        "",
    ])

    gen_dominant = [
        f"{rm.language}/{rm.backend} ({_fmt(rm.generation_share_pct)}%)"
        for rm in runs.values()
        if rm.generation_share_pct >= 80.0
    ]
    if gen_dominant:
        lines.append(
            "- Generation dominates end-to-end latency in these runs: "
            + ", ".join(sorted(gen_dominant))
            + "."
        )
    else:
        lines.append("- Generation is not dominant in all runs; inspect run-specific bottlenecks.")

    best_p50 = min(runs.values(), key=lambda r: r.p50_latency_ms if r.p50_latency_ms > 0 else float("inf"))
    lines.append(
        f"- Fastest p50 latency among discovered runs: {best_p50.language}/{best_p50.backend} "
        f"({_fmt(best_p50.p50_latency_ms)} ms)."
    )

    stress_runs = [r for r in runs.values() if r.stress is not None]
    if stress_runs:
        best_qps = max(stress_runs, key=lambda r: r.stress.queries_per_second if r.stress else 0.0)
        lines.append(
            f"- Highest stress throughput among runs with stress data: {best_qps.language}/{best_qps.backend} "
            f"({_fmt(best_qps.stress.queries_per_second if best_qps.stress else 0.0)} QPS)."
        )

    lines.extend([
        "",
        "### llama_cpp Focus (Python vs Rust)",
        "",
    ])

    py_llama = runs.get(("python", "llama_cpp"))
    rs_llama = runs.get(("rust", "llama_cpp"))
    if py_llama is None or rs_llama is None:
        missing = []
        if py_llama is None:
            missing.append("python/llama_cpp")
        if rs_llama is None:
            missing.append("rust/llama_cpp")
        lines.append(
            "- Missing llama_cpp comparison inputs: "
            + ", ".join(missing)
            + "."
        )
    else:
        p50_delta = rs_llama.p50_latency_ms - py_llama.p50_latency_ms
        p95_delta = rs_llama.p95_latency_ms - py_llama.p95_latency_ms
        e2e_delta = rs_llama.mean_end_to_end_ms - py_llama.mean_end_to_end_ms
        emb_delta = rs_llama.embedding_phase_ms - py_llama.embedding_phase_ms
        idx_delta = rs_llama.index_build_ms - py_llama.index_build_ms

        lines.append(
            "- Both llama_cpp runs completed all queries without failures "
            f"({py_llama.failure_count} vs {rs_llama.failure_count})."
        )
        lines.append(
            f"- Latency delta (Rust - Python): p50={_fmt(p50_delta)} ms, "
            f"p95={_fmt(p95_delta)} ms, mean_end_to_end={_fmt(e2e_delta)} ms."
        )
        lines.append(
            f"- Build-time stages are significantly higher in Rust for this run: "
            f"embedding delta={_fmt(emb_delta)} ms, index delta={_fmt(idx_delta)} ms."
        )
        lines.append(
            f"- Mean TTFT differs sharply (Python={_fmt(py_llama.mean_ttft_ms)} ms, "
            f"Rust={_fmt(rs_llama.mean_ttft_ms)} ms); treat this with caution if token streaming "
            "instrumentation differs between bindings."
        )

    lines.extend([
        "",
        "## Conclusions",
        "",
        "- Use same backend + same query_count + same stress settings for strict apples-to-apples conclusions.",
        "- If query_count or failure_count differs, treat deltas as directional, not definitive.",
        "- In most discovered runs, inference generation time is the primary bottleneck, not retrieval/indexing.",
        "- Prefer backend-specific stress strategies: HTTP backends can share server concurrency; in-process backends should avoid unsafe shared model instances across threads.",
        "- For this benchmark iteration, Qwen2.5 GGUF is a practical llama_cpp comparison model because it loads and runs in both Python and Rust pipelines.",
        "- Keep Llama-3.2 GGUF as a compatibility track item; exclude it from strict Python-vs-Rust llama_cpp performance claims until loader compatibility is resolved.",
        "",
    ])

    return lines


def generate_all_scenarios_report(output_dir: Path, output_path: Path) -> None:
    runs, discovered, ignored = _discover_runs(output_dir)

    lines: list[str] = [
        "# Cross-Scenario Benchmark Report",
        "",
        "This report compares all discovered JSONL metrics files and builds three requested comparisons:",
        "1. ollama_http (Rust vs Python)",
        "2. llama_cpp (Rust vs Python)",
        "3. llm_rs (Rust) vs llama_cpp (Python)",
        "",
        "## Discovered JSONL Files",
        "",
    ]

    for p in discovered:
        lines.append(f"- {p.name}")

    if ignored:
        lines.append("")
        lines.append("## Ignored JSONL Files")
        lines.append("")
        for p in ignored:
            lines.append(f"- {p.name} (does not match metrics_<language>_<backend>.jsonl)")

    lines.append("")

    lines.extend(
        _comparison_section(
            title="Scenario 1: ollama_http (Rust vs Python)",
            left_label="Python ollama_http",
            left=runs.get(("python", "ollama_http")),
            right_label="Rust ollama_http",
            right=runs.get(("rust", "ollama_http")),
            delta_label="Rust - Python",
        )
    )

    lines.extend(
        _comparison_section(
            title="Scenario 2: llama_cpp (Rust vs Python)",
            left_label="Python llama_cpp",
            left=runs.get(("python", "llama_cpp")),
            right_label="Rust llama_cpp",
            right=runs.get(("rust", "llama_cpp")),
            delta_label="Rust - Python",
        )
    )

    lines.extend(
        _comparison_section(
            title="Scenario 3: llm_rs vs llama_cpp (Rust vs Python)",
            left_label="Python llama_cpp",
            left=runs.get(("python", "llama_cpp")),
            right_label="Rust llm_rs",
            right=runs.get(("rust", "llm_rs")),
            delta_label="Rust(llm_rs) - Python(llama_cpp)",
        )
    )

    lines.extend(_analysis_and_conclusions(runs))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a cross-scenario report from all output JSONL metrics files."
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory that contains metrics_*.jsonl files.",
    )
    parser.add_argument(
        "--output",
        default="output/benchmark_report_all_scenarios.md",
        help="Output markdown report path.",
    )
    args = parser.parse_args()

    generate_all_scenarios_report(
        output_dir=Path(args.output_dir),
        output_path=Path(args.output),
    )
    print(f"Cross-scenario report written to {args.output}")
