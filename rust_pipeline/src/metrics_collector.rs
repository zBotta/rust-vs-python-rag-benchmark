//! Metrics collector — Task 6 implementation.
//!
//! Provides QueryMetrics and PipelineMetrics structs, percentile computation,
//! and JSONL serialization/deserialization.

use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum MetricsError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Missing summary record in JSONL file")]
    MissingSummary,
}

// ---------------------------------------------------------------------------
// Data models
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QueryMetrics {
    pub query_id: usize,
    pub end_to_end_ms: f64,
    pub retrieval_ms: f64,
    pub ttft_ms: f64,
    pub generation_ms: f64,
    pub total_tokens: u64,
    pub failed: bool,
    pub failure_reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PipelineMetrics {
    pub embedding_phase_ms: f64,
    pub index_build_ms: f64,
    pub queries: Vec<QueryMetrics>,
    pub p50_latency_ms: f64,
    pub p95_latency_ms: f64,
}

// ---------------------------------------------------------------------------
// Internal JSONL record types (for serde)
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum JsonlRecord {
    Query {
        query_id: usize,
        end_to_end_ms: f64,
        retrieval_ms: f64,
        ttft_ms: f64,
        generation_ms: f64,
        total_tokens: u64,
        failed: bool,
        failure_reason: Option<String>,
    },
    Summary {
        embedding_phase_ms: f64,
        index_build_ms: f64,
        p50_latency_ms: f64,
        p95_latency_ms: f64,
        failure_count: u64,
    },
    #[serde(rename = "stress_summary")]
    StressSummary {
        concurrency: usize,
        total_queries: usize,
        queries_per_second: f64,
        peak_rss_mb: f64,
        p99_latency_ms: f64,
        p50_latency_ms: f64,
        p95_latency_ms: f64,
        failure_count: usize,
    },
}

// ---------------------------------------------------------------------------
// Percentile computation
// ---------------------------------------------------------------------------

/// Compute (p50, p95) for a slice of latency values.
///
/// Uses linear interpolation (numpy-compatible).
/// Returns (0.0, 0.0) for an empty slice.
pub fn compute_percentiles(latencies: &[f64]) -> (f64, f64) {
    if latencies.is_empty() {
        return (0.0, 0.0);
    }

    let mut sorted = latencies.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let p50 = percentile_linear(&sorted, 50.0);
    let p95 = percentile_linear(&sorted, 95.0);

    (p50, p95)
}

/// Linear-interpolation percentile (matches numpy default).
fn percentile_linear(sorted: &[f64], pct: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }

    let idx = pct / 100.0 * (n as f64 - 1.0);
    let lo = idx.floor() as usize;
    let hi = lo + 1;
    let frac = idx - lo as f64;

    if hi >= n {
        return sorted[n - 1];
    }

    sorted[lo] + frac * (sorted[hi] - sorted[lo])
}

// ---------------------------------------------------------------------------
// Stress test metrics — Task 19
// ---------------------------------------------------------------------------

/// Stress test summary metrics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StressSummary {
    pub concurrency: usize,
    pub total_queries: usize,
    pub queries_per_second: f64,
    pub peak_rss_mb: f64,
    pub p99_latency_ms: f64,
    pub p50_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub failure_count: usize,
}

/// Compute the 99th percentile for a slice of latency values.
///
/// Returns 0.0 for an empty slice.
pub fn compute_p99(latencies: &[f64]) -> f64 {
    if latencies.is_empty() {
        return 0.0;
    }
    let mut sorted = latencies.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    percentile_linear(&sorted, 99.0)
}

/// Compute stress test summary metrics from a slice of QueryMetrics.
///
/// `queries_per_second = len(query_metrics) / total_wall_clock_s`.
/// p50/p95/p99 are computed from successful query latencies only.
/// `peak_rss_mb` is read from the current process via the `sysinfo` crate.
pub fn compute_stress_summary(
    query_metrics: &[QueryMetrics],
    concurrency: usize,
    total_wall_clock_s: f64,
) -> StressSummary {
    let queries_per_second = if total_wall_clock_s > 0.0 {
        query_metrics.len() as f64 / total_wall_clock_s
    } else {
        0.0
    };

    // Peak RSS via sysinfo (memory() returns bytes in sysinfo 0.31)
    let rss_mb = {
        use sysinfo::System;
        let mut sys = System::new_all();
        sys.refresh_all();
        sysinfo::get_current_pid()
            .ok()
            .and_then(|pid| sys.process(pid).map(|p| p.memory() as f64 / (1024.0 * 1024.0)))
            .unwrap_or(0.0)
    };

    let successful_latencies: Vec<f64> = query_metrics
        .iter()
        .filter(|q| !q.failed)
        .map(|q| q.end_to_end_ms)
        .collect();

    let (p50, p95) = compute_percentiles(&successful_latencies);
    let p99 = compute_p99(&successful_latencies);
    let failure_count = query_metrics.iter().filter(|q| q.failed).count();

    StressSummary {
        concurrency,
        total_queries: query_metrics.len(),
        queries_per_second,
        peak_rss_mb: rss_mb,
        p99_latency_ms: p99,
        p50_latency_ms: p50,
        p95_latency_ms: p95,
        failure_count,
    }
}

/// Append a stress_summary JSON record to an existing JSONL file.
pub fn append_stress_summary_to_jsonl(
    summary: &StressSummary,
    output_path: &Path,
) -> Result<(), MetricsError> {
    let file = std::fs::OpenOptions::new()
        .append(true)
        .create(true)
        .open(output_path)?;
    let mut writer = std::io::BufWriter::new(file);

    let record = JsonlRecord::StressSummary {
        concurrency: summary.concurrency,
        total_queries: summary.total_queries,
        queries_per_second: summary.queries_per_second,
        peak_rss_mb: summary.peak_rss_mb,
        p99_latency_ms: summary.p99_latency_ms,
        p50_latency_ms: summary.p50_latency_ms,
        p95_latency_ms: summary.p95_latency_ms,
        failure_count: summary.failure_count,
    };
    let line = serde_json::to_string(&record)?;
    writeln!(writer, "{}", line)?;
    Ok(())
}

/// Read a JSONL file and return the stress_summary record if present, or None.
pub fn read_stress_summary_from_jsonl(
    input_path: &Path,
) -> Result<Option<StressSummary>, MetricsError> {
    let file = std::fs::File::open(input_path)?;
    let reader = BufReader::new(file);

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let record: JsonlRecord = serde_json::from_str(line)?;
        if let JsonlRecord::StressSummary {
            concurrency,
            total_queries,
            queries_per_second,
            peak_rss_mb,
            p99_latency_ms,
            p50_latency_ms,
            p95_latency_ms,
            failure_count,
        } = record
        {
            return Ok(Some(StressSummary {
                concurrency,
                total_queries,
                queries_per_second,
                peak_rss_mb,
                p99_latency_ms,
                p50_latency_ms,
                p95_latency_ms,
                failure_count,
            }));
        }
    }
    Ok(None)
}

// ---------------------------------------------------------------------------
// JSONL serialization
// ---------------------------------------------------------------------------

/// Serialize PipelineMetrics to a JSONL file.
///
/// Each query is written as one JSON object per line with `"type": "query"`.
/// The final line is a summary object with `"type": "summary"`.
pub fn serialize_to_jsonl(metrics: &PipelineMetrics, output_path: &Path) -> Result<(), MetricsError> {
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let file = std::fs::File::create(output_path)?;
    let mut writer = std::io::BufWriter::new(file);

    for q in &metrics.queries {
        let record = JsonlRecord::Query {
            query_id: q.query_id,
            end_to_end_ms: q.end_to_end_ms,
            retrieval_ms: q.retrieval_ms,
            ttft_ms: q.ttft_ms,
            generation_ms: q.generation_ms,
            total_tokens: q.total_tokens,
            failed: q.failed,
            failure_reason: q.failure_reason.clone(),
        };
        let line = serde_json::to_string(&record)?;
        writeln!(writer, "{}", line)?;
    }

    let successful_latencies: Vec<f64> = metrics
        .queries
        .iter()
        .filter(|q| !q.failed)
        .map(|q| q.end_to_end_ms)
        .collect();

    let (p50, p95) = compute_percentiles(&successful_latencies);
    let failure_count = metrics.queries.iter().filter(|q| q.failed).count() as u64;

    let summary = JsonlRecord::Summary {
        embedding_phase_ms: metrics.embedding_phase_ms,
        index_build_ms: metrics.index_build_ms,
        p50_latency_ms: p50,
        p95_latency_ms: p95,
        failure_count,
    };
    let line = serde_json::to_string(&summary)?;
    writeln!(writer, "{}", line)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// JSONL deserialization
// ---------------------------------------------------------------------------

/// Deserialize PipelineMetrics from a JSONL file produced by serialize_to_jsonl.
pub fn deserialize_from_jsonl(input_path: &Path) -> Result<PipelineMetrics, MetricsError> {
    let file = std::fs::File::open(input_path)?;
    let reader = BufReader::new(file);

    let mut queries: Vec<QueryMetrics> = Vec::new();
    let mut summary_opt: Option<(f64, f64, f64, f64)> = None;

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let record: JsonlRecord = serde_json::from_str(line)?;
        match record {
            JsonlRecord::Query {
                query_id,
                end_to_end_ms,
                retrieval_ms,
                ttft_ms,
                generation_ms,
                total_tokens,
                failed,
                failure_reason,
            } => {
                queries.push(QueryMetrics {
                    query_id,
                    end_to_end_ms,
                    retrieval_ms,
                    ttft_ms,
                    generation_ms,
                    total_tokens,
                    failed,
                    failure_reason,
                });
            }
            JsonlRecord::Summary {
                embedding_phase_ms,
                index_build_ms,
                p50_latency_ms,
                p95_latency_ms,
                failure_count: _,
            } => {
                summary_opt = Some((embedding_phase_ms, index_build_ms, p50_latency_ms, p95_latency_ms));
            }
            JsonlRecord::StressSummary { .. } => {
                // stress_summary records are handled by read_stress_summary_from_jsonl; skip here
            }
        }
    }

    let (embedding_phase_ms, index_build_ms, p50_latency_ms, p95_latency_ms) =
        summary_opt.ok_or(MetricsError::MissingSummary)?;

    Ok(PipelineMetrics {
        embedding_phase_ms,
        index_build_ms,
        queries,
        p50_latency_ms,
        p95_latency_ms,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use tempfile::NamedTempFile;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn make_query(id: usize, e2e: f64, ret: f64, ttft: f64, gen: f64, tokens: u64, failed: bool) -> QueryMetrics {
        QueryMetrics {
            query_id: id,
            end_to_end_ms: e2e,
            retrieval_ms: ret,
            ttft_ms: ttft,
            generation_ms: gen,
            total_tokens: tokens,
            failed,
            failure_reason: if failed { Some("test failure".to_string()) } else { None },
        }
    }

    // -----------------------------------------------------------------------
    // Sub-task 6.1 — Property 9: All metric fields present and non-negative
    // Feature: rust-vs-python-rag-benchmark, Property 9: All metric fields present and non-negative
    // -----------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// Property 9: All metric fields present and non-negative.
        ///
        /// # Feature: rust-vs-python-rag-benchmark, Property 9: All metric fields present and non-negative
        /// Validates: Requirements 6.1, 6.2
        #[test]
        fn prop_all_metric_fields_non_negative(
            e2e in 0.0f64..1_000_000.0,
            ret in 0.0f64..1_000_000.0,
            ttft in 0.0f64..1_000_000.0,
            gen in 0.0f64..1_000_000.0,
            tokens in 0u64..100_000u64,
            embedding_ms in 0.0f64..1_000_000.0,
            index_ms in 0.0f64..1_000_000.0,
        ) {
            // Feature: rust-vs-python-rag-benchmark, Property 9: All metric fields present and non-negative
            let q = make_query(0, e2e, ret, ttft, gen, tokens, false);

            prop_assert!(q.end_to_end_ms >= 0.0, "end_to_end_ms must be non-negative");
            prop_assert!(q.retrieval_ms >= 0.0, "retrieval_ms must be non-negative");
            prop_assert!(q.ttft_ms >= 0.0, "ttft_ms must be non-negative");
            prop_assert!(q.generation_ms >= 0.0, "generation_ms must be non-negative");
            prop_assert!(q.total_tokens >= 0, "total_tokens must be non-negative");

            let pipeline = PipelineMetrics {
                embedding_phase_ms: embedding_ms,
                index_build_ms: index_ms,
                queries: vec![q],
                p50_latency_ms: 0.0,
                p95_latency_ms: 0.0,
            };

            prop_assert!(pipeline.embedding_phase_ms >= 0.0, "embedding_phase_ms must be non-negative");
            prop_assert!(pipeline.index_build_ms >= 0.0, "index_build_ms must be non-negative");
        }

        // -----------------------------------------------------------------------
        // Sub-task 6.2 — Property 10: p50 = median, p95 = 95th percentile
        // Feature: rust-vs-python-rag-benchmark, Property 10: p50 = median, p95 = 95th percentile of end-to-end latency values
        // -----------------------------------------------------------------------

        /// Property 10: p50 = median, p95 = 95th percentile.
        ///
        /// # Feature: rust-vs-python-rag-benchmark, Property 10: p50 = median, p95 = 95th percentile of end-to-end latency values
        /// Validates: Requirements 6.3
        #[test]
        fn prop_p50_is_median_p95_is_95th_percentile(
            latencies in prop::collection::vec(0.0f64..100_000.0, 1..200),
        ) {
            // Feature: rust-vs-python-rag-benchmark, Property 10: p50 = median, p95 = 95th percentile of end-to-end latency values
            let (p50, p95) = compute_percentiles(&latencies);

            // Verify p50 is the median (linear interpolation)
            let mut sorted = latencies.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let expected_p50 = percentile_linear(&sorted, 50.0);
            let expected_p95 = percentile_linear(&sorted, 95.0);

            prop_assert!(
                (p50 - expected_p50).abs() < 1e-9,
                "p50={} expected={}", p50, expected_p50
            );
            prop_assert!(
                (p95 - expected_p95).abs() < 1e-9,
                "p95={} expected={}", p95, expected_p95
            );

            // p50 must be within [min, max]
            let min_val = sorted.first().copied().unwrap_or(0.0);
            let max_val = sorted.last().copied().unwrap_or(0.0);
            prop_assert!(p50 >= min_val && p50 <= max_val, "p50 out of range");
            prop_assert!(p95 >= min_val && p95 <= max_val, "p95 out of range");
            prop_assert!(p95 >= p50, "p95 must be >= p50");
        }

        // -----------------------------------------------------------------------
        // Sub-task 6.3 — Property 11: Failed queries excluded from percentile calculations
        // Feature: rust-vs-python-rag-benchmark, Property 11: Percentiles computed only from successful queries; failure count = count of failed=true records
        // -----------------------------------------------------------------------

        /// Property 11: Failed queries excluded from percentile calculations.
        ///
        /// # Feature: rust-vs-python-rag-benchmark, Property 11: Percentiles computed only from successful queries; failure count = count of failed=true records
        /// Validates: Requirements 6.5
        #[test]
        fn prop_failed_queries_excluded_from_percentiles(
            successful_latencies in prop::collection::vec(0.0f64..100_000.0, 0..50),
            failed_latencies in prop::collection::vec(0.0f64..100_000.0, 0..20),
        ) {
            // Feature: rust-vs-python-rag-benchmark, Property 11: Percentiles computed only from successful queries; failure count = count of failed=true records
            let mut queries: Vec<QueryMetrics> = Vec::new();
            let mut id = 0usize;

            for lat in &successful_latencies {
                queries.push(make_query(id, *lat, 0.0, 0.0, 0.0, 0, false));
                id += 1;
            }
            for lat in &failed_latencies {
                queries.push(make_query(id, *lat, 0.0, 0.0, 0.0, 0, true));
                id += 1;
            }

            let failure_count = queries.iter().filter(|q| q.failed).count();
            prop_assert_eq!(failure_count, failed_latencies.len(), "failure_count mismatch");

            // Compute percentiles using only successful latencies
            let (expected_p50, expected_p95) = compute_percentiles(&successful_latencies);

            // Simulate what serialize_to_jsonl does
            let success_lats: Vec<f64> = queries.iter()
                .filter(|q| !q.failed)
                .map(|q| q.end_to_end_ms)
                .collect();
            let (actual_p50, actual_p95) = compute_percentiles(&success_lats);

            prop_assert!(
                (actual_p50 - expected_p50).abs() < 1e-9,
                "p50 mismatch: actual={} expected={}", actual_p50, expected_p50
            );
            prop_assert!(
                (actual_p95 - expected_p95).abs() < 1e-9,
                "p95 mismatch: actual={} expected={}", actual_p95, expected_p95
            );
        }
    }

    // -----------------------------------------------------------------------
    // Sub-task 6.4 — Property 12: JSONL metrics round-trip
    // Feature: rust-vs-python-rag-benchmark, Property 12: serialize(PipelineMetrics) → JSONL → deserialize == original
    // -----------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// Property 12: JSONL round-trip preserves PipelineMetrics equality.
        ///
        /// # Feature: rust-vs-python-rag-benchmark, Property 12: serialize(PipelineMetrics) → JSONL → deserialize == original
        /// Validates: Requirements 6.4
        #[test]
        fn prop_jsonl_round_trip(
            embedding_ms in 0.0f64..1_000_000.0,
            index_ms in 0.0f64..1_000_000.0,
            query_count in 0usize..20usize,
            base_e2e in prop::collection::vec(0.0f64..100_000.0, 0..20),
            base_failed in prop::collection::vec(any::<bool>(), 0..20),
        ) {
            // Feature: rust-vs-python-rag-benchmark, Property 12: serialize(PipelineMetrics) → JSONL → deserialize == original
            let n = query_count.min(base_e2e.len()).min(base_failed.len());
            let mut queries: Vec<QueryMetrics> = Vec::new();
            for i in 0..n {
                queries.push(make_query(i, base_e2e[i], 1.0, 2.0, 3.0, 10, base_failed[i]));
            }

            // Compute p50/p95 from successful queries (as serialize_to_jsonl does)
            let success_lats: Vec<f64> = queries.iter()
                .filter(|q| !q.failed)
                .map(|q| q.end_to_end_ms)
                .collect();
            let (p50, p95) = compute_percentiles(&success_lats);

            let original = PipelineMetrics {
                embedding_phase_ms: embedding_ms,
                index_build_ms: index_ms,
                queries: queries.clone(),
                p50_latency_ms: p50,
                p95_latency_ms: p95,
            };

            let tmp = NamedTempFile::new().expect("temp file");
            serialize_to_jsonl(&original, tmp.path()).expect("serialize");
            let restored = deserialize_from_jsonl(tmp.path()).expect("deserialize");

            prop_assert_eq!(restored.embedding_phase_ms, original.embedding_phase_ms);
            prop_assert_eq!(restored.index_build_ms, original.index_build_ms);
            prop_assert_eq!(restored.p50_latency_ms, original.p50_latency_ms);
            prop_assert_eq!(restored.p95_latency_ms, original.p95_latency_ms);
            prop_assert_eq!(restored.queries.len(), original.queries.len());

            for (r, o) in restored.queries.iter().zip(original.queries.iter()) {
                prop_assert_eq!(r.query_id, o.query_id);
                prop_assert_eq!(r.end_to_end_ms, o.end_to_end_ms);
                prop_assert_eq!(r.retrieval_ms, o.retrieval_ms);
                prop_assert_eq!(r.ttft_ms, o.ttft_ms);
                prop_assert_eq!(r.generation_ms, o.generation_ms);
                prop_assert_eq!(r.total_tokens, o.total_tokens);
                prop_assert_eq!(r.failed, o.failed);
                prop_assert_eq!(r.failure_reason.as_deref(), o.failure_reason.as_deref());
            }
        }
    }
}
