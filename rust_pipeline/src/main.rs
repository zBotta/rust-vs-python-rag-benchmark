//! Rust RAG pipeline entry point — Task 10.
//!
//! Wires: dataset_loader → chunker → embedder → vector_store → retriever
//!        → llm_client → metrics_collector
//!
//! Reads config from `benchmark_config.toml`, runs all queries sequentially,
//! and writes `metrics_rust.jsonl` to `output_dir`.

mod config;
pub mod logger;
mod dataset_loader;
mod chunker;
mod embedder;
mod vector_store;
mod retriever;
mod llm_client;
mod llm_client_llm_rs;
mod llm_client_llama_cpp;
mod metrics_collector;
mod stress_runner;

use std::path::Path;
use std::time::Instant;

use metrics_collector::{
    compute_percentiles, serialize_to_jsonl, PipelineMetrics, QueryMetrics,
};
use retriever::Retriever;
use vector_store::VectorStore;
use llm_client::LlmClient;

fn main() {
    // -----------------------------------------------------------------------
    // 1. Load config from benchmark_config.toml
    // -----------------------------------------------------------------------
    let config_path = "benchmark_config.toml";
    let cfg = match config::load_config(config_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Fatal: failed to load config '{}': {}", config_path, e);
            std::process::exit(1);
        }
    };

    // Ensure output directory exists
    if let Err(e) = std::fs::create_dir_all(&cfg.output_dir) {
        eprintln!("Fatal: cannot create output directory '{}': {}", cfg.output_dir, e);
        std::process::exit(1);
    }

    // -----------------------------------------------------------------------
    // 1a. Construct BenchmarkLogger
    // -----------------------------------------------------------------------
    let min_level = match cfg.log_level.to_uppercase().as_str() {
        "DEBUG"   => logger::LogLevel::Debug,
        "INFO"    => logger::LogLevel::Info,
        "WARNING" => logger::LogLevel::Warning,
        "ERROR"   => logger::LogLevel::Error,
        other => {
            eprintln!("Warning: unrecognised log_level '{}'; falling back to INFO", other);
            logger::LogLevel::Info
        }
    };
    let mut logger = logger::BenchmarkLogger::new(&cfg.output_dir, &cfg.llm_backend, "rust", min_level)
        .unwrap_or_else(|e| {
            eprintln!("Warning: failed to create logger: {}", e);
            // Fallback: create logger writing to a temp location
            logger::BenchmarkLogger::new("/tmp", &cfg.llm_backend, "rust", min_level)
                .expect("Failed to create fallback logger")
        });

    // -----------------------------------------------------------------------
    // 1b. Select LLM client based on llm_backend
    // -----------------------------------------------------------------------
    let llm_client_arc: std::sync::Arc<dyn LlmClient + Send + Sync> = match cfg.llm_backend.as_str() {
        "llama_cpp" => {
            match llm_client_llama_cpp::LlamaCppClient::new(&cfg.gguf_model_path) {
                Ok(c) => std::sync::Arc::new(c),
                Err(e) => {
                    eprintln!("Fatal: failed to load llama_cpp model: {}", e);
                    std::process::exit(1);
                }
            }
        }
        "llm_rs" => {
            match llm_client_llm_rs::LlmRsClient::new(&cfg.gguf_model_path) {
                Ok(c) => std::sync::Arc::new(c),
                Err(e) => {
                    eprintln!("Fatal: failed to load llm_rs model: {}", e);
                    std::process::exit(1);
                }
            }
        }
        _ => {
            // Default: ollama_http
            std::sync::Arc::new(llm_client::OllamaHttpClient::new(&cfg.llm_host, &cfg.llm_model))
        }
    };

    // -----------------------------------------------------------------------
    // 1c. Preflight checks — validate LLM backend before loading data
    // -----------------------------------------------------------------------
    match cfg.llm_backend.as_str() {
        "ollama_http" => {
            let tags_url = format!("{}/api/tags", cfg.llm_host);
            let disable_ssl = std::env::var("DISABLE_SSL_VERIFY")
                .map(|v| matches!(v.to_lowercase().as_str(), "1" | "true" | "yes"))
                .unwrap_or(false);
            let client = reqwest::blocking::Client::builder()
                .danger_accept_invalid_certs(disable_ssl)
                .timeout(std::time::Duration::from_secs(5))
                .build()
                .expect("Failed to build HTTP client");
            match client.get(&tags_url).send() {
                Err(e) => {
                    eprintln!("ERROR: Ollama endpoint not reachable at {}: {}", cfg.llm_host, e);
                    std::process::exit(1);
                }
                Ok(resp) => {
                    let body: serde_json::Value = resp.json().unwrap_or_default();
                    let models: Vec<String> = body["models"]
                        .as_array()
                        .unwrap_or(&vec![])
                        .iter()
                        .filter_map(|m| m["name"].as_str().map(|s| s.to_string()))
                        .collect();
                    if !models.contains(&cfg.llm_model) {
                        eprintln!(
                            "ERROR: Model '{}' not found. Available: {:?}",
                            cfg.llm_model, models
                        );
                        std::process::exit(1);
                    }
                    println!("Preflight OK: model '{}' is available.", cfg.llm_model);
                }
            }
        }
        "llama_cpp" | "llm_rs" => {
            let path = std::path::Path::new(&cfg.gguf_model_path);
            if !path.exists() {
                eprintln!("ERROR: GGUF model file not found: {}", cfg.gguf_model_path);
                std::process::exit(1);
            }
            if !path.is_file() {
                eprintln!("ERROR: GGUF path is not a file: {}", cfg.gguf_model_path);
                std::process::exit(1);
            }
            println!("Preflight OK: GGUF model file found at {}", cfg.gguf_model_path);
        }
        _ => {}
    }

    // -----------------------------------------------------------------------
    // 2. Load documents
    // -----------------------------------------------------------------------
    println!("Loading documents...");
    logger.log_loading_start(&cfg.dataset_name, &cfg.dataset_subset, cfg.num_documents);
    let load_start = Instant::now();
    let docs = match dataset_loader::load_documents(
        &cfg.dataset_name,
        &cfg.dataset_subset,
        cfg.num_documents,
    ) {
        Ok(d) => d,
        Err(e) => {
            logger.log_loading_error(&e.to_string());
            eprintln!("Fatal: failed to load dataset: {}", e);
            std::process::exit(1);
        }
    };
    let load_elapsed_ms = load_start.elapsed().as_secs_f64() * 1000.0;
    logger.log_loading_complete(docs.len(), load_elapsed_ms);
    println!("Loaded {} documents.", docs.len());

    // -----------------------------------------------------------------------
    // 3. Chunk documents
    // -----------------------------------------------------------------------
    println!("Chunking documents...");
    logger.log_chunking_start(cfg.chunk_size, cfg.chunk_overlap);
    let chunk_start = Instant::now();
    let chunks = chunker::chunk_documents(&docs, cfg.chunk_size, cfg.chunk_overlap);
    let chunk_elapsed_ms = chunk_start.elapsed().as_secs_f64() * 1000.0;
    logger.log_chunking_complete(chunks.len(), chunk_elapsed_ms);
    if chunks.is_empty() && !docs.is_empty() {
        logger.log_chunking_zero_warning();
    }
    println!("Produced {} chunks.", chunks.len());

    // -----------------------------------------------------------------------
    // 4. Embed all chunks — record embedding_phase_ms
    // -----------------------------------------------------------------------
    println!("Embedding chunks...");
    logger.log_embedding_start(&cfg.embedding_model, chunks.len());
    let embed_start = Instant::now();
    let embeddings: Vec<[f32; 384]> = match embedder::embed_chunks(&chunks) {
        Ok(e) => e,
        Err(e) => {
            logger.log_embedding_error(&e.to_string());
            eprintln!("Fatal: embedding failed: {}", e);
            std::process::exit(1);
        }
    };
    let embedding_phase_ms = embed_start.elapsed().as_secs_f64() * 1000.0;
    // Emit progress records every 100 chunks
    for i in (100..=chunks.len()).step_by(100) {
        logger.log_embedding_progress(i);
    }
    logger.log_embedding_complete(embedding_phase_ms);
    println!("Embedding done in {:.1} ms.", embedding_phase_ms);

    // -----------------------------------------------------------------------
    // 5. Build vector index — record index_build_ms
    // -----------------------------------------------------------------------
    println!("Building vector index...");
    logger.log_index_build_start(embeddings.len());
    let mut vs = VectorStore::new(384);
    let index_start = Instant::now();
    if let Err(e) = vs.build_index(&embeddings) {
        logger.log_index_build_error(&e.to_string());
        eprintln!("Fatal: failed to build vector index: {}", e);
        std::process::exit(1);
    }
    let index_build_ms = index_start.elapsed().as_secs_f64() * 1000.0;
    logger.log_index_build_complete(index_build_ms);
    println!("Index built in {:.1} ms.", index_build_ms);

    let retriever = Retriever::new(chunks.clone(), vs);

    // -----------------------------------------------------------------------
    // 6. Load query set from query_set.json
    // -----------------------------------------------------------------------
    let query_set_path = Path::new(&cfg.query_set_path);
    let query_set_raw = match std::fs::read_to_string(query_set_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Fatal: cannot read query set '{}': {}", query_set_path.display(), e);
            std::process::exit(1);
        }
    };
    let queries: Vec<serde_json::Value> = match serde_json::from_str(&query_set_raw) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Fatal: cannot parse query set JSON: {}", e);
            std::process::exit(1);
        }
    };
    let total_queries = queries.len();
    println!("Loaded {} queries.", total_queries);

    // -----------------------------------------------------------------------
    // 7. Run each query sequentially
    // -----------------------------------------------------------------------
    let mut query_metrics_list: Vec<QueryMetrics> = Vec::with_capacity(total_queries);

    for (i, entry) in queries.iter().enumerate() {
        let query_id = entry["id"].as_u64().unwrap_or(i as u64) as usize;
        let question = match entry["question"].as_str() {
            Some(q) => q.to_string(),
            None => {
                eprintln!("Warning: query {} has no 'question' field; skipping.", query_id);
                query_metrics_list.push(QueryMetrics {
                    query_id,
                    end_to_end_ms: 0.0,
                    retrieval_ms: 0.0,
                    ttft_ms: 0.0,
                    generation_ms: 0.0,
                    total_tokens: 0,
                    failed: true,
                    failure_reason: Some("Missing 'question' field".to_string()),
                });
                continue;
            }
        };

        println!("Running query {}/{}...", i + 1, total_queries);

        let e2e_start = Instant::now();

        // a. Embed query
        let query_embedding: [f32; 384] = match embedder::embed_chunks(&[question.clone()]) {
            Ok(mut e) if !e.is_empty() => e.remove(0),
            Err(e) => {
                let end_to_end_ms = e2e_start.elapsed().as_secs_f64() * 1000.0;
                eprintln!("Warning: query embedding failed for query {}: {}", query_id, e);
                query_metrics_list.push(QueryMetrics {
                    query_id,
                    end_to_end_ms,
                    retrieval_ms: 0.0,
                    ttft_ms: 0.0,
                    generation_ms: 0.0,
                    total_tokens: 0,
                    failed: true,
                    failure_reason: Some(format!("Embedding error: {}", e)),
                });
                continue;
            }
            _ => {
                eprintln!("Warning: empty embedding for query {}", query_id);
                continue;
            }
        };

        // a. Retrieve top-k chunks — record retrieval_ms
        let retrieval_start = Instant::now();
        logger.log_retrieval_start(query_id);
        let retrieved_chunks = match retriever.retrieve(&query_embedding, cfg.top_k) {
            Ok(c) => c,
            Err(e) => {
                let end_to_end_ms = e2e_start.elapsed().as_secs_f64() * 1000.0;
                logger.log_retrieval_error(query_id, &e.to_string());
                eprintln!("Warning: retrieval failed for query {}: {}", query_id, e);
                query_metrics_list.push(QueryMetrics {
                    query_id,
                    end_to_end_ms,
                    retrieval_ms: retrieval_start.elapsed().as_secs_f64() * 1000.0,
                    ttft_ms: 0.0,
                    generation_ms: 0.0,
                    total_tokens: 0,
                    failed: true,
                    failure_reason: Some(format!("Retrieval error: {}", e)),
                });
                continue;
            }
        };
        let retrieval_ms = retrieval_start.elapsed().as_secs_f64() * 1000.0;
        logger.log_retrieval_complete(query_id, retrieved_chunks.len(), retrieval_ms);

    // b. Generate answer — record ttft_ms, generation_ms, total_tokens
        logger.log_generation_start(query_id, retrieved_chunks.len());
        let response = match llm_client_arc.generate(&question, &retrieved_chunks) {
            Ok(r) => r,
            Err(e) => {
                let end_to_end_ms = e2e_start.elapsed().as_secs_f64() * 1000.0;
                logger.log_generation_error(query_id, &e.to_string());
                eprintln!("Warning: LLM call failed for query {}: {}", query_id, e);
                query_metrics_list.push(QueryMetrics {
                    query_id,
                    end_to_end_ms,
                    retrieval_ms,
                    ttft_ms: 0.0,
                    generation_ms: 0.0,
                    total_tokens: 0,
                    failed: true,
                    failure_reason: Some(format!("LLM error: {}", e)),
                });
                continue;
            }
        };

        // c. Record end-to-end latency
        let end_to_end_ms = e2e_start.elapsed().as_secs_f64() * 1000.0;

        // d. Create QueryMetrics record
        let qm = if response.failed {
            let reason = response.failure_reason.as_deref().unwrap_or("unknown");
            logger.log_generation_failed_response(query_id, reason);
            QueryMetrics {
                query_id,
                end_to_end_ms,
                retrieval_ms,
                ttft_ms: 0.0,
                generation_ms: 0.0,
                total_tokens: 0,
                failed: true,
                failure_reason: response.failure_reason,
            }
        } else {
            logger.log_generation_complete(query_id, response.total_tokens as usize, response.ttft_ms, response.generation_ms);
            QueryMetrics {
                query_id,
                end_to_end_ms,
                retrieval_ms,
                ttft_ms: response.ttft_ms,
                generation_ms: response.generation_ms,
                total_tokens: response.total_tokens,
                failed: false,
                failure_reason: None,
            }
        };

        query_metrics_list.push(qm);
    }

    // -----------------------------------------------------------------------
    // 8. Compute p50/p95 from successful queries
    // -----------------------------------------------------------------------
    let successful_latencies: Vec<f64> = query_metrics_list
        .iter()
        .filter(|q| !q.failed)
        .map(|q| q.end_to_end_ms)
        .collect();
    let (p50, p95) = compute_percentiles(&successful_latencies);

    let failure_count = query_metrics_list.iter().filter(|q| q.failed).count();

    // -----------------------------------------------------------------------
    // 9. Create PipelineMetrics and serialize to {output_dir}/metrics_rust.jsonl
    // -----------------------------------------------------------------------
    let pipeline_metrics = PipelineMetrics {
        embedding_phase_ms,
        index_build_ms,
        queries: query_metrics_list,
        p50_latency_ms: p50,
        p95_latency_ms: p95,
    };

    let output_path = Path::new(&cfg.output_dir).join(format!("metrics_rust_{}.jsonl", cfg.llm_backend));
    if let Err(e) = serialize_to_jsonl(&pipeline_metrics, &output_path) {
        eprintln!("Fatal: failed to write metrics to '{}': {}", output_path.display(), e);
        std::process::exit(1);
    }

    println!("Metrics written to {}", output_path.display());
    println!(
        "p50={:.1} ms  p95={:.1} ms  failures={}/{}",
        p50, p95, failure_count, total_queries
    );
    logger.log_run_summary(total_queries, failure_count, p50, p95, &output_path.display().to_string());

    // -----------------------------------------------------------------------
    // 10. Run stress test phase if enabled
    // -----------------------------------------------------------------------
    if cfg.stress_test.enabled {
        println!("\nStarting stress test phase...");

        // Reload documents using stress_test.num_documents
        let stress_docs = match dataset_loader::load_documents(
            &cfg.dataset_name,
            &cfg.dataset_subset,
            cfg.stress_test.num_documents,
        ) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Fatal: stress test dataset load failed: {}", e);
                std::process::exit(1);
            }
        };
        let stress_chunks = chunker::chunk_documents(&stress_docs, cfg.chunk_size, cfg.chunk_overlap);
        let stress_embeddings = match embedder::embed_chunks(&stress_chunks) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("Fatal: stress test embedding failed: {}", e);
                std::process::exit(1);
            }
        };
        let mut stress_vs = VectorStore::new(384);
        if let Err(e) = stress_vs.build_index(&stress_embeddings) {
            eprintln!("Fatal: stress test index build failed: {}", e);
            std::process::exit(1);
        }

        let query_strings: Vec<String> = queries
            .iter()
            .filter_map(|e| e["question"].as_str().map(|s| s.to_string()))
            .collect();

        let stress_runner = stress_runner::StressRunner {
            chunks: std::sync::Arc::new(stress_chunks),
            vector_store: std::sync::Arc::new(stress_vs),
            llm_client: std::sync::Arc::clone(&llm_client_arc),
            query_set: query_strings,
            concurrency: cfg.stress_test.concurrency,
            query_repetitions: cfg.stress_test.query_repetitions,
        };

        let stress_wall_start = Instant::now();
        let stress_results = stress_runner.run();
        let stress_wall_clock_s = stress_wall_start.elapsed().as_secs_f64();
        println!("Stress test complete: {} queries dispatched.", stress_results.len());

        // Compute and append stress summary to the JSONL file
        let stress_summary = metrics_collector::compute_stress_summary(
            &stress_results,
            cfg.stress_test.concurrency,
            stress_wall_clock_s,
        );
        if let Err(e) = metrics_collector::append_stress_summary_to_jsonl(&stress_summary, &output_path) {
            eprintln!("Warning: failed to write stress summary: {}", e);
        } else {
            println!(
                "Stress summary: {:.2} QPS  peak_rss={:.1} MB  p99={:.1} ms",
                stress_summary.queries_per_second,
                stress_summary.peak_rss_mb,
                stress_summary.p99_latency_ms,
            );
            logger.log_stress_summary(stress_summary.queries_per_second, stress_summary.peak_rss_mb, stress_summary.p99_latency_ms);
        }
    }
}
