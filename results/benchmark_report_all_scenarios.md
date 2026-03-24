# Cross-Scenario Benchmark Report

This report compares all discovered JSONL metrics files and builds three requested comparisons:
1. ollama_http (Rust vs Python)
2. llama_cpp (Rust vs Python)
3. llm_rs (Rust) vs llama_cpp (Python)

## Discovered JSONL Files

- metrics_python.jsonl
- metrics_python_llama_cpp.jsonl
- metrics_python_ollama_http.jsonl
- metrics_rust.jsonl
- metrics_rust_llama_cpp.jsonl
- metrics_rust_llm_rs.jsonl
- metrics_rust_ollama_http.jsonl

## Ignored JSONL Files

- metrics_python.jsonl (does not match metrics_<language>_<backend>.jsonl)
- metrics_rust.jsonl (does not match metrics_<language>_<backend>.jsonl)

## Scenario 1: ollama_http (Rust vs Python)

Source files: metrics_python_ollama_http.jsonl vs metrics_rust_ollama_http.jsonl

| Metric | Python ollama_http | Rust ollama_http | Rust - Python |
| --- | ---: | ---: | ---: |
| query_count | 50 | 50 | 0 |
| success_count | 50 | 50 | 0 |
| failure_count | 0 | 0 | 0 |
| failure_rate_pct | 0.00 | 0.00 | 0.00 |
| p50_latency_ms | 8070.01 | 8497.59 | 427.59 |
| p95_latency_ms | 13988.27 | 12164.25 | -1824.01 |
| mean_end_to_end_ms | 8794.63 | 8021.30 | -773.33 |
| mean_ttft_ms | 5177.62 | 0.03 | -5177.59 |
| mean_generation_ms | 8782.13 | 4432.45 | -4349.69 |
| mean_retrieval_ms | 12.41 | 6.44 | -5.97 |
| mean_total_tokens | 508.64 | 556.68 | 48.04 |
| embedding_phase_ms | 344522.16 | 1718839.47 | 1374317.31 |
| index_build_ms | 2172.00 | 43209.31 | 41037.31 |
| stress_qps | 0.13 | 0.14 | 0.01 |
| stress_p99_latency_ms | 77774.02 | 81020.64 | 3246.62 |
| stress_peak_rss_mb | 365.07 | 125.47 | -239.60 |

## Scenario 2: llama_cpp (Rust vs Python)

Source files: metrics_python_llama_cpp.jsonl vs metrics_rust_llama_cpp.jsonl

| Metric | Python llama_cpp | Rust llama_cpp | Rust - Python |
| --- | ---: | ---: | ---: |
| query_count | 50 | 50 | 0 |
| success_count | 50 | 50 | 0 |
| failure_count | 0 | 0 | 0 |
| failure_rate_pct | 0.00 | 0.00 | 0.00 |
| p50_latency_ms | 6312.99 | 12422.12 | 6109.13 |
| p95_latency_ms | 7733.57 | 59507.35 | 51773.78 |
| mean_end_to_end_ms | 6164.26 | 23545.31 | 17381.05 |
| mean_ttft_ms | 1902.57 | 1.74 | -1900.83 |
| mean_generation_ms | 6135.86 | 20970.54 | 14834.68 |
| mean_retrieval_ms | 15.63 | 3.18 | -12.44 |
| mean_total_tokens | 0.00 | 0.00 | 0.00 |
| embedding_phase_ms | 275228.76 | 1496271.35 | 1221042.59 |
| index_build_ms | 1406.50 | 49298.59 | 47892.09 |

## Scenario 3: llm_rs vs llama_cpp (Rust vs Python)

Source files: metrics_python_llama_cpp.jsonl vs metrics_rust_llm_rs.jsonl

| Metric | Python llama_cpp | Rust llm_rs | Rust(llm_rs) - Python(llama_cpp) |
| --- | ---: | ---: | ---: |
| query_count | 50 | 50 | 0 |
| success_count | 50 | 50 | 0 |
| failure_count | 0 | 0 | 0 |
| failure_rate_pct | 0.00 | 0.00 | 0.00 |
| p50_latency_ms | 6312.99 | 31703.21 | 25390.22 |
| p95_latency_ms | 7733.57 | 34539.52 | 26805.95 |
| mean_end_to_end_ms | 6164.26 | 30936.97 | 24772.71 |
| mean_ttft_ms | 1902.57 | 6058.20 | 4155.63 |
| mean_generation_ms | 6135.86 | 30913.81 | 24777.94 |
| mean_retrieval_ms | 15.63 | 3.65 | -11.98 |
| mean_total_tokens | 0.00 | 371.14 | 371.14 |
| embedding_phase_ms | 275228.76 | 1802916.72 | 1527687.96 |
| index_build_ms | 1406.50 | 29397.31 | 27990.81 |

## Analysis

### Run Inventory

| Run | query_count | failures | p50_latency_ms | mean_end_to_end_ms | generation_share_pct |
| --- | ---: | ---: | ---: | ---: | ---: |
| python/llama_cpp | 50 | 0 | 6312.99 | 6164.26 | 99.54 |
| python/ollama_http | 50 | 0 | 8070.01 | 8794.63 | 99.86 |
| rust/llama_cpp | 50 | 0 | 12422.12 | 23545.31 | 89.06 |
| rust/llm_rs | 50 | 0 | 31703.21 | 30936.97 | 99.93 |
| rust/ollama_http | 50 | 0 | 8497.59 | 8021.30 | 55.26 |

### Key Observations

- Generation dominates end-to-end latency in these runs: python/llama_cpp (99.54%), python/ollama_http (99.86%), rust/llama_cpp (89.06%), rust/llm_rs (99.93%).
- Fastest p50 latency among discovered runs: python/llama_cpp (6312.99 ms).
- Highest stress throughput among runs with stress data: rust/ollama_http (0.14 QPS).

### llama_cpp Focus (Python vs Rust)

- Both llama_cpp runs completed all queries without failures (0 vs 0).
- Latency delta (Rust - Python): p50=6109.13 ms, p95=51773.78 ms, mean_end_to_end=17381.05 ms.
- Build-time stages are significantly higher in Rust for this run: embedding delta=1221042.59 ms, index delta=47892.09 ms.
- Mean TTFT differs sharply (Python=1902.57 ms, Rust=1.74 ms); treat this with caution if token streaming instrumentation differs between bindings.

## Conclusions

- Use same backend + same query_count + same stress settings for strict apples-to-apples conclusions.
- If query_count or failure_count differs, treat deltas as directional, not definitive.
- In most discovered runs, inference generation time is the primary bottleneck, not retrieval/indexing.
- Prefer backend-specific stress strategies: HTTP backends can share server concurrency; in-process backends should avoid unsafe shared model instances across threads.
- For this benchmark iteration, Qwen2.5 GGUF is a practical llama_cpp comparison model because it loads and runs in both Python and Rust pipelines.
- Keep Llama-3.2 GGUF as a compatibility track item; exclude it from strict Python-vs-Rust llama_cpp performance claims until loader compatibility is resolved.
