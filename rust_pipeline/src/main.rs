//! Rust RAG pipeline entry point — Task 10.
//!
//! Wires: dataset_loader → chunker → embedder → vector_store → retriever
//!        → llm_client → metrics_collector
//!
//! Reads config from `benchmark_config.toml`, runs all queries sequentially,
//! and writes `metrics_rust.jsonl` to `output_dir`.

mod config;
mod dataset_loader;
mod chunker;
mod embedder;
mod vector_store;
mod retriever;
mod llm_client;
mod metrics_collector;

use std::path::Path;
use std::time::Instant;

use metrics_collector::{
    compute_percentiles, serialize_to_jsonl, PipelineMetrics, QueryMetrics,
};
use retriever::Retriever;
use vector_store::VectorStore;

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
    // 2. Load documents
    // -----------------------------------------------------------------------
    println!("Loading documents...");
    let docs = match dataset_loader::load_documents(
        &cfg.dataset_name,
        &cfg.dataset_subset,
        cfg.num_documents,
    ) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Fatal: failed to load dataset: {}", e);
            std::process::exit(1);
        }
    };
    println!("Loaded {} documents.", docs.len());

    // -----------------------------------------------------------------------
    // 3. Chunk documents
    // -----------------------------------------------------------------------
    println!("Chunking documents...");
    let chunks = chunker::chunk_documents(&docs, cfg.chunk_size, cfg.chunk_overlap);
    println!("Produced {} chunks.", chunks.len());

    // -----------------------------------------------------------------------
    // 4. Embed all chunks — record embedding_phase_ms
    // -----------------------------------------------------------------------
    println!("Embedding chunks...");
    let embed_start = Instant::now();
    let embeddings: Vec<[f32; 384]> = match embedder::embed_chunks(&chunks) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Fatal: embedding failed: {}", e);
            std::process::exit(1);
        }
    };
    let embedding_phase_ms = embed_start.elapsed().as_secs_f64() * 1000.0;
    println!("Embedding done in {:.1} ms.", embedding_phase_ms);

    // -----------------------------------------------------------------------
    // 5. Build vector index — record index_build_ms
    // -----------------------------------------------------------------------
    println!("Building vector index...");
    let mut vs = VectorStore::new(384);
    let index_start = Instant::now();
    if let Err(e) = vs.build_index(&embeddings) {
        eprintln!("Fatal: failed to build vector index: {}", e);
        std::process::exit(1);
    }
    let index_build_ms = index_start.elapsed().as_secs_f64() * 1000.0;
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
        let retrieved_chunks = match retriever.retrieve(&query_embedding, cfg.top_k) {
            Ok(c) => c,
            Err(e) => {
                let end_to_end_ms = e2e_start.elapsed().as_secs_f64() * 1000.0;
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

    // b. Generate answer — record ttft_ms, generation_ms, total_tokens
        let response = match llm_client::generate(&question, &retrieved_chunks, &cfg.llm_host, &cfg.llm_model, 3) {
            Ok(r) => r,
            Err(e) => {
                let end_to_end_ms = e2e_start.elapsed().as_secs_f64() * 1000.0;
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

    let output_path = Path::new(&cfg.output_dir).join("metrics_rust.jsonl");
    if let Err(e) = serialize_to_jsonl(&pipeline_metrics, &output_path) {
        eprintln!("Fatal: failed to write metrics to '{}': {}", output_path.display(), e);
        std::process::exit(1);
    }

    println!("Metrics written to {}", output_path.display());
    println!(
        "p50={:.1} ms  p95={:.1} ms  failures={}/{}",
        p50, p95, failure_count, total_queries
    );
}
