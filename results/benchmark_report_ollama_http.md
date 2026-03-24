# Rust vs Python RAG Benchmark Report — ollama_http

## Summary Table

| Metric | Python value | Rust value | Delta | Delta % |
| --- | --- | --- | --- | --- |
| p50 end-to-end latency | 8070.01 | 8497.59 | +427.59 | 5.30% |
| p95 end-to-end latency | 13988.27 | 12164.25 | -1824.01 | -13.04% |
| mean TTFT | 5177.62 | 0.03 | -5177.59 | -100.00% |
| mean retrieval latency | 12.41 | 6.44 | -5.97 | -48.11% |
| mean Total_Tokens | 508.64 | 556.68 | +48.04 | 9.44% |
| embedding phase time | 344522.16 | 1718839.47 | +1374317.31 | 398.91% |
| index build time | 2172.00 | 43209.31 | +41037.31 | 1889.38% |
| failure count | 0 | 0 | +0 | N/A |

## Per-Query Latency Distribution

![Latency Histogram](latency_histogram.png)

![Latency CDF](latency_cdf.png)


## Stress Test Results

| Metric | Python value | Rust value | Delta | Delta % |
| --- | --- | --- | --- | --- |
| throughput (QPS) | 0.13 | 0.14 | +0.01 | 4.48% |
| peak RSS (MB) | 365.07 | 125.47 | -239.60 | -65.63% |
| p50 latency (ms) | 60829.39 | 57112.03 | -3717.36 | -6.11% |
| p95 latency (ms) | 73220.84 | 73117.93 | -102.90 | -0.14% |
| p99 latency (ms) | 77774.02 | 81020.64 | +3246.62 | 4.17% |
| failure count | 0 | 0 | +0 | N/A |
