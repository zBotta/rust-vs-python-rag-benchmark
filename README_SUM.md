# Rust vs Python RAG Benchmark Summary

## 1. What This Project Is

This project is a fair, side-by-side benchmark of the same RAG system implemented in two languages:

- Python
- Rust

The core question is simple:

When architecture and models are held constant, where does each language perform better for real AI workloads?

---

## 2. Why This Matters

The goal is not to declare one language universally better.

The goal is to produce reproducible evidence for engineering decisions, especially around:

- Development speed and ecosystem flexibility
- Runtime efficiency, latency, and concurrency behavior
- Production-readiness tradeoffs

---

## 3. What the Pipeline Does

Both implementations run the same sequence:

1. Load the same Wikipedia document dataset
2. Split documents into chunks
3. Convert chunks into embeddings
4. Build a vector index for semantic retrieval
5. Retrieve relevant context for each query
6. Send query + context to an LLM for answer generation
7. Collect performance and reliability metrics
8. Generate a comparison report

---

## 4. Technologies Used

## Data
- Hugging Face Wikipedia dataset
- Local parquet cache for reproducibility and speed

## Embeddings
- sentence-transformers/all-MiniLM-L6-v2

## Vector Search
- HNSW index
- Python: hnswlib
- Rust: hnsw_rs or instant-distance stack

## LLM Backends
- Ollama HTTP
- llama.cpp in-process
- llm-rs in-process (Rust-only mode)

## Metrics and Reporting
- JSONL raw metrics output
- Markdown benchmark report
- Main KPIs:
  - p50, p95, p99 latency
  - TTFT
  - Generation time
  - Throughput (QPS)
  - Peak RSS memory
  - Failure count

---

## 5. Project Vision

This benchmark is intended as a decision tool for real systems, helping teams answer:

- Is Python sufficient for our latency and throughput targets?
- Where does Rust meaningfully improve performance and stability?
- Which backend and architecture gives the best cost/performance balance?

---

## 6. Elevator Pitch

A reproducible benchmark platform that compares Python and Rust implementations of the same RAG pipeline to quantify latency, throughput, memory usage, and reliability tradeoffs for production AI systems.