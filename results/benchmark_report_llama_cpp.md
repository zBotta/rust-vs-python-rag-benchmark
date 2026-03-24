# Rust vs Python RAG Benchmark Report — llama_cpp

## Summary Table

| Metric | Python value | Rust value | Delta | Delta % |
| --- | --- | --- | --- | --- |
| p50 end-to-end latency | 6312.99 | 12422.12 | +6109.13 | 96.77% |
| p95 end-to-end latency | 7733.57 | 59507.35 | +51773.78 | 669.47% |
| mean TTFT | 1902.57 | 1.74 | -1900.83 | -99.91% |
| mean retrieval latency | 15.63 | 3.18 | -12.44 | -79.63% |
| mean Total_Tokens | 0.00 | 0.00 | +0.00 | N/A |
| embedding phase time | 275228.76 | 1496271.35 | +1221042.59 | 443.65% |
| index build time | 1406.50 | 49298.59 | +47892.09 | 3405.05% |
| failure count | 0 | 0 | +0 | N/A |

## Per-Query Latency Distribution

![Latency Histogram](latency_histogram.png)

![Latency CDF](latency_cdf.png)
