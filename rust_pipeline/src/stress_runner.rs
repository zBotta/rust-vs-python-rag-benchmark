//! Stress test mode — concurrent query dispatcher (Task 18).
//!
//! Dispatches `query_repetitions × query_set.len()` queries across
//! `concurrency` threads, each worker using its own `embedder::embed_chunks`
//! call and querying the shared read-only `Arc<VectorStore>`.

use std::sync::{Arc, Mutex};
use std::thread;

use crate::embedder;
use crate::llm_client::{LlmClient, LLMResponse};
use crate::metrics_collector::QueryMetrics;
use crate::vector_store::VectorStore;

pub struct StressRunner {
    pub chunks: Arc<Vec<String>>,
    pub vector_store: Arc<VectorStore>,
    pub llm_client: Arc<dyn LlmClient + Send + Sync>,
    pub query_set: Vec<String>,
    pub concurrency: usize,
    pub query_repetitions: usize,
}

impl StressRunner {
    /// Dispatch all queries concurrently and return per-query metrics.
    ///
    /// Total queries = `query_repetitions × query_set.len()`.
    /// Each worker gets its own `embedder::embed_chunks` call and its own
    /// view of the shared read-only `VectorStore`.
    pub fn run(&self) -> Vec<QueryMetrics> {
        // Build full query list: query_set repeated query_repetitions times
        let mut all_queries: Vec<(usize, String)> = Vec::new();
        for rep in 0..self.query_repetitions {
            for (idx, question) in self.query_set.iter().enumerate() {
                let query_id = rep * self.query_set.len() + idx;
                all_queries.push((query_id, question.clone()));
            }
        }

        let total = all_queries.len();
        let results: Arc<Mutex<Vec<QueryMetrics>>> = Arc::new(Mutex::new(Vec::with_capacity(total)));

        // Partition queries across concurrency workers
        let concurrency = self.concurrency.max(1);
        let chunks_per_worker = (total + concurrency - 1) / concurrency;

        let mut handles = Vec::new();

        for worker_queries in all_queries.chunks(chunks_per_worker) {
            let worker_queries: Vec<(usize, String)> = worker_queries.to_vec();
            let chunks = Arc::clone(&self.chunks);
            let vector_store = Arc::clone(&self.vector_store);
            let llm_client = Arc::clone(&self.llm_client);
            let results = Arc::clone(&results);

            let handle = thread::spawn(move || {
                for (query_id, question) in worker_queries {
                    let qm = run_single_query(
                        query_id,
                        &question,
                        &chunks,
                        &vector_store,
                        &*llm_client,
                    );
                    let mut guard = results.lock().unwrap();
                    guard.push(qm);
                }
            });

            handles.push(handle);
        }

        for handle in handles {
            let _ = handle.join();
        }

        Arc::try_unwrap(results)
            .expect("all threads finished")
            .into_inner()
            .unwrap()
    }
}

/// Run a single query: embed → retrieve → generate → record metrics.
fn run_single_query(
    query_id: usize,
    question: &str,
    chunks: &[String],
    vector_store: &VectorStore,
    llm_client: &dyn LlmClient,
) -> QueryMetrics {
    use std::time::Instant;

    let e2e_start = Instant::now();

    // Embed query — each worker calls embed_chunks independently
    let query_embedding: [f32; 384] = match embedder::embed_chunks(&[question.to_string()]) {
        Ok(mut e) if !e.is_empty() => e.remove(0),
        Ok(_) => {
            return QueryMetrics {
                query_id,
                end_to_end_ms: e2e_start.elapsed().as_secs_f64() * 1000.0,
                retrieval_ms: 0.0,
                ttft_ms: 0.0,
                generation_ms: 0.0,
                total_tokens: 0,
                failed: true,
                failure_reason: Some("Empty embedding result".to_string()),
            };
        }
        Err(e) => {
            return QueryMetrics {
                query_id,
                end_to_end_ms: e2e_start.elapsed().as_secs_f64() * 1000.0,
                retrieval_ms: 0.0,
                ttft_ms: 0.0,
                generation_ms: 0.0,
                total_tokens: 0,
                failed: true,
                failure_reason: Some(format!("Embedding error: {}", e)),
            };
        }
    };

    // Retrieve top-k chunks from shared read-only VectorStore
    let retrieval_start = Instant::now();
    let retrieved_chunks = match vector_store.query(&query_embedding, 5) {
        Ok(results) => results
            .into_iter()
            .map(|(id, _score)| chunks[id].clone())
            .collect::<Vec<_>>(),
        Err(e) => {
            return QueryMetrics {
                query_id,
                end_to_end_ms: e2e_start.elapsed().as_secs_f64() * 1000.0,
                retrieval_ms: retrieval_start.elapsed().as_secs_f64() * 1000.0,
                ttft_ms: 0.0,
                generation_ms: 0.0,
                total_tokens: 0,
                failed: true,
                failure_reason: Some(format!("Retrieval error: {}", e)),
            };
        }
    };
    let retrieval_ms = retrieval_start.elapsed().as_secs_f64() * 1000.0;

    // Generate answer
    let response = match llm_client.generate(question, &retrieved_chunks) {
        Ok(r) => r,
        Err(e) => {
            return QueryMetrics {
                query_id,
                end_to_end_ms: e2e_start.elapsed().as_secs_f64() * 1000.0,
                retrieval_ms,
                ttft_ms: 0.0,
                generation_ms: 0.0,
                total_tokens: 0,
                failed: true,
                failure_reason: Some(format!("LLM error: {}", e)),
            };
        }
    };

    let end_to_end_ms = e2e_start.elapsed().as_secs_f64() * 1000.0;

    if response.failed {
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
    }
}
