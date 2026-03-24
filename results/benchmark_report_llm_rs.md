# Rust vs Python RAG Benchmark Report — llm_rs

> ℹ️ **Note**: Python pipeline was skipped (llm_rs backend is Rust-only). This is a single-pipeline report.

## Summary Table

| Metric | Rust value |
| --- | --- |
| p50 end-to-end latency | 31703.21 |
| p95 end-to-end latency | 34539.52 |
| mean TTFT | 6058.20 |
| mean retrieval latency | 3.65 |
| mean Total_Tokens | 371.14 |
| embedding phase time | 1802916.72 |
| index build time | 29397.31 |
| failure count | 0 |

## Per-Query Latency Distribution

![Latency Histogram](latency_histogram.png)

![Latency CDF](latency_cdf.png)
