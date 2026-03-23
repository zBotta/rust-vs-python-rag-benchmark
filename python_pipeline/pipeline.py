"""Top-level Python RAG pipeline — Task 9.

Wires: dataset_loader → chunker → embedder → vector_store → retriever → llm_client → metrics_collector
Reads config from benchmark_config.toml, runs all queries sequentially, and writes
metrics_python.jsonl to output_dir.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# SSL bypass — must happen before any network library is imported.
# Set DISABLE_SSL_VERIFY=1 when behind a corporate proxy with SSL inspection.
# ---------------------------------------------------------------------------
import os
if os.environ.get("DISABLE_SSL_VERIFY", "").lower() in ("1", "true", "yes"):
    import ssl
    import urllib.request
    ssl._create_default_https_context = ssl._create_unverified_context
    os.environ["CURL_CA_BUNDLE"] = ""
    os.environ["REQUESTS_CA_BUNDLE"] = ""
    os.environ["SSL_CERT_FILE"] = ""
    os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
    os.environ["HTTPX_VERIFY"] = "0"  # httpx reads this at client creation time

import json
import sys
import time
from pathlib import Path

from python_pipeline import config as config_module
from python_pipeline import dataset_loader, chunker, embedder
from python_pipeline.vector_store import VectorStore
from python_pipeline.retriever import Retriever
from python_pipeline import llm_client
from python_pipeline import llm_client_llama_cpp
from python_pipeline.metrics_collector import (
    QueryMetrics,
    PipelineMetrics,
    compute_percentiles,
    serialize_to_jsonl,
    compute_stress_summary,
    append_stress_summary_to_jsonl,
)
from python_pipeline.stress_runner import StressRunner


def _preflight_ollama(llm_host: str, llm_model: str) -> None:
    """Check Ollama reachability and model availability before dispatching queries."""
    import httpx
    try:
        resp = httpx.get(f"{llm_host}/api/tags", timeout=5.0)
        resp.raise_for_status()
    except Exception as exc:
        print(f"ERROR: Ollama endpoint not reachable at {llm_host}: {exc}")
        sys.exit(1)

    models = [m["name"] for m in resp.json().get("models", [])]
    if llm_model not in models:
        print(f"ERROR: Model '{llm_model}' not found. Available: {models}")
        sys.exit(1)

    print(f"Preflight OK: model '{llm_model}' is available.")


def _preflight_gguf(gguf_model_path: str) -> None:
    """Check that the GGUF model file exists and is readable before loading."""
    path = Path(gguf_model_path)
    if not path.exists():
        print(f"ERROR: GGUF model file not found: {gguf_model_path}")
        sys.exit(1)
    if not path.is_file():
        print(f"ERROR: GGUF path is not a file: {gguf_model_path}")
        sys.exit(1)
    try:
        with path.open("rb") as fh:
            magic = fh.read(4)
        if magic != b"GGUF":
            print(f"WARNING: File does not start with GGUF magic bytes: {gguf_model_path}")
    except OSError as exc:
        print(f"ERROR: Cannot read GGUF model file: {exc}")
        sys.exit(1)
    print(f"Preflight OK: GGUF model file found at {gguf_model_path}")


def run_pipeline(config_path: str = "benchmark_config.toml") -> None:
    """Run the full Python RAG pipeline and write metrics_python.jsonl."""

    # 1. Load config
    cfg = config_module.load_config(config_path)

    # Select LLM backend — llm_rs is Rust-only; exit cleanly after validation.
    if cfg.llm_backend == "llm_rs":
        print("Python pipeline skipped: llm_rs backend is Rust-only")
        sys.exit(0)

    # Build a unified llm_generate_fn based on the selected backend.
    if cfg.llm_backend == "llama_cpp":
        def llm_generate_fn(query: str, chunks: list) -> object:
            return llm_client_llama_cpp.generate(
                query=query,
                chunks=chunks,
                gguf_model_path=cfg.gguf_model_path,
            )
    else:
        # Default: ollama_http
        def llm_generate_fn(query: str, chunks: list) -> object:
            return llm_client.generate(
                query=query,
                chunks=chunks,
                llm_host=cfg.llm_host,
                model=cfg.llm_model,
            )

    if cfg.llm_backend == "ollama_http":
        _preflight_ollama(cfg.llm_host, cfg.llm_model)
    elif cfg.llm_backend == "llama_cpp":
        _preflight_gguf(cfg.gguf_model_path)

    # Ensure output directory exists
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load documents
    print("Loading documents...")
    docs = dataset_loader.load_documents(cfg.dataset_name, cfg.dataset_subset, cfg.num_documents)
    print(f"Loaded {len(docs)} documents.")

    # 3. Chunk documents
    print("Chunking documents...")
    chunks = chunker.chunk_documents(docs, chunk_size=cfg.chunk_size, overlap=cfg.chunk_overlap)
    print(f"Produced {len(chunks)} chunks.")

    # 4. Embed all chunks — record embedding_phase_ms
    print("Embedding chunks...")
    embed_start = time.perf_counter()
    embeddings = embedder.embed_chunks(chunks)
    embedding_phase_ms = (time.perf_counter() - embed_start) * 1000.0
    print(f"Embedding done in {embedding_phase_ms:.1f} ms.")

    # 5. Build vector index — record index_build_ms
    print("Building vector index...")
    vs = VectorStore(dim=384, space="cosine")
    index_start = time.perf_counter()
    vs.build_index(embeddings)
    index_build_ms = (time.perf_counter() - index_start) * 1000.0
    print(f"Index built in {index_build_ms:.1f} ms.")

    # embedder_fn for Retriever: embed a single query string → list[float]
    def embedder_fn(query: str) -> list[float]:
        return embedder.embed_chunks([query])[0]

    retriever = Retriever(chunks=chunks, vector_store=vs, embedder_fn=embedder_fn)

    # 6. Load query set
    query_set_path = Path(cfg.query_set_path)
    with query_set_path.open("r", encoding="utf-8") as fh:
        queries = json.load(fh)
    total_queries = len(queries)
    print(f"Loaded {total_queries} queries.")

    # 7. Run each query sequentially
    query_metrics_list: list[QueryMetrics] = []

    for i, entry in enumerate(queries):
        query_id: int = entry["id"]
        question: str = entry["question"]
        print(f"Running query {i + 1}/{total_queries}...")

        e2e_start = time.perf_counter()

        try:
            # a. Retrieve top-k chunks — record retrieval_ms
            retrieval_start = time.perf_counter()
            retrieved_chunks = retriever.retrieve(question, top_k=cfg.top_k)
            retrieval_ms = (time.perf_counter() - retrieval_start) * 1000.0

            # b. Generate answer — record ttft_ms, generation_ms, total_tokens
            response = llm_generate_fn(
                query=question,
                chunks=retrieved_chunks,
            )

            # c. Record end-to-end latency
            end_to_end_ms = (time.perf_counter() - e2e_start) * 1000.0

            if response.failed:
                qm = QueryMetrics(
                    query_id=query_id,
                    end_to_end_ms=end_to_end_ms,
                    retrieval_ms=retrieval_ms,
                    ttft_ms=0.0,
                    generation_ms=0.0,
                    total_tokens=0,
                    failed=True,
                    failure_reason=response.failure_reason,
                )
            else:
                qm = QueryMetrics(
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
            qm = QueryMetrics(
                query_id=query_id,
                end_to_end_ms=end_to_end_ms,
                retrieval_ms=0.0,
                ttft_ms=0.0,
                generation_ms=0.0,
                total_tokens=0,
                failed=True,
                failure_reason=str(exc),
            )

        query_metrics_list.append(qm)
        status = "FAIL" if qm.failed else "OK"
        reason = f" — {qm.failure_reason}" if qm.failed else ""
        print(f"  Query {i + 1}/{total_queries}: {status}{reason}")

    # 8. Compute p50/p95 from successful queries
    successful_latencies = [q.end_to_end_ms for q in query_metrics_list if not q.failed]
    p50, p95 = compute_percentiles(successful_latencies)

    # 9. Create PipelineMetrics and serialize to {output_dir}/metrics_python_{llm_backend}.jsonl
    pipeline_metrics = PipelineMetrics(
        embedding_phase_ms=embedding_phase_ms,
        index_build_ms=index_build_ms,
        queries=query_metrics_list,
        p50_latency_ms=p50,
        p95_latency_ms=p95,
    )

    llm_backend = cfg.llm_backend
    output_path = output_dir / f"metrics_python_{llm_backend}.jsonl"
    serialize_to_jsonl(pipeline_metrics, str(output_path))
    print(f"Metrics written to {output_path}")
    print(f"p50={p50:.1f} ms  p95={p95:.1f} ms  failures={sum(1 for q in query_metrics_list if q.failed)}/{total_queries}")

    # 10. Run stress test phase if enabled
    if cfg.stress_test.enabled:
        print("\nStarting stress test phase...")
        # Reload documents using stress_test.num_documents
        stress_docs = dataset_loader.load_documents(
            cfg.dataset_name, cfg.dataset_subset, cfg.stress_test.num_documents
        )
        stress_chunks = chunker.chunk_documents(
            stress_docs, chunk_size=cfg.chunk_size, overlap=cfg.chunk_overlap
        )
        stress_embeddings = embedder.embed_chunks(stress_chunks)
        stress_vs = VectorStore(dim=384, space="cosine")
        stress_vs.build_index(stress_embeddings)

        query_strings = [entry["question"] for entry in queries]

        runner = StressRunner(
            chunks=stress_chunks,
            vector_store=stress_vs,
            llm_generate_fn=llm_generate_fn,
            query_set=query_strings,
            concurrency=cfg.stress_test.concurrency,
            query_repetitions=cfg.stress_test.query_repetitions,
        )
        stress_results = runner.run()
        print(f"Stress test complete: {len(stress_results)} queries dispatched.")

        # Compute and append stress summary to the JSONL file
        stress_summary = compute_stress_summary(
            query_metrics=stress_results,
            concurrency=cfg.stress_test.concurrency,
            total_wall_clock_s=runner.last_wall_clock_s,
        )
        append_stress_summary_to_jsonl(stress_summary, str(output_path))
        print(
            f"Stress summary: {stress_summary.queries_per_second:.2f} QPS  "
            f"peak_rss={stress_summary.peak_rss_mb:.1f} MB  "
            f"p99={stress_summary.p99_latency_ms:.1f} ms"
        )


if __name__ == "__main__":
    run_pipeline()
