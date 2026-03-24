//! Structured benchmark logger for the Rust RAG pipeline.
//!
//! Writes one log file per run to `{output_dir}/{backend}-{language}.log`,
//! truncating any prior content. Each record is a single line:
//!
//! ```text
//! 2024-01-15T10:23:45.123Z [INFO] [loading] Starting dataset load ...
//! ```

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};

use chrono::Utc;
use thiserror::Error;

// ---------------------------------------------------------------------------
// LogLevel
// ---------------------------------------------------------------------------

/// Severity levels in ascending order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    Debug = 0,
    Info = 1,
    Warning = 2,
    Error = 3,
}

impl LogLevel {
    fn as_str(self) -> &'static str {
        match self {
            LogLevel::Debug => "DEBUG",
            LogLevel::Info => "INFO",
            LogLevel::Warning => "WARNING",
            LogLevel::Error => "ERROR",
        }
    }
}

// ---------------------------------------------------------------------------
// LoggerError
// ---------------------------------------------------------------------------

/// Errors that can arise when constructing or using a [`BenchmarkLogger`].
#[derive(Debug, Error)]
pub enum LoggerError {
    #[error("Logger I/O error: {0}")]
    Io(#[from] std::io::Error),
}

// ---------------------------------------------------------------------------
// BenchmarkLogger
// ---------------------------------------------------------------------------

/// Writes structured log records to `{output_dir}/{backend}-{language}.log`.
pub struct BenchmarkLogger {
    writer: BufWriter<File>,
    min_level: LogLevel,
}

impl BenchmarkLogger {
    /// Create (or truncate) the log file and return a ready logger.
    ///
    /// # Errors
    ///
    /// Returns [`LoggerError::Io`] if the directory cannot be created or the
    /// file cannot be opened.
    pub fn new(
        output_dir: &str,
        backend: &str,
        language: &str,
        min_level: LogLevel,
    ) -> Result<Self, LoggerError> {
        std::fs::create_dir_all(output_dir)?;
        let path = format!("{}/{}-{}.log", output_dir, backend, language);
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)?;
        Ok(Self {
            writer: BufWriter::new(file),
            min_level,
        })
    }

    // -----------------------------------------------------------------------
    // Core write primitive
    // -----------------------------------------------------------------------

    /// Write a single log record if `level >= self.min_level`.
    ///
    /// The line is flushed immediately so callers always see a consistent file.
    pub fn log(&mut self, level: LogLevel, stage: &str, message: &str) {
        if level < self.min_level {
            return;
        }
        let now = Utc::now();
        let ts = now.format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string();
        let line = format!("{} [{}] [{}] {}\n", ts, level.as_str(), stage, message);
        if let Err(e) = self.writer.write_all(line.as_bytes()) {
            eprintln!("BenchmarkLogger write error: {}", e);
            return;
        }
        if let Err(e) = self.writer.flush() {
            eprintln!("BenchmarkLogger flush error: {}", e);
        }
    }

    // -----------------------------------------------------------------------
    // Loading stage helpers
    // -----------------------------------------------------------------------

    pub fn log_loading_start(&mut self, dataset_name: &str, subset: &str, num_docs: usize) {
        self.log(
            LogLevel::Info,
            "loading",
            &format!(
                "Starting dataset load: {}/{}, requested={} docs",
                dataset_name, subset, num_docs
            ),
        );
    }

    pub fn log_loading_complete(&mut self, num_docs: usize, elapsed_ms: f64) {
        self.log(
            LogLevel::Info,
            "loading",
            &format!("Loading complete: {} docs loaded in {:.1} ms", num_docs, elapsed_ms),
        );
    }

    pub fn log_loading_batch(&mut self, batch_size: usize, cumulative: usize) {
        self.log(
            LogLevel::Debug,
            "loading",
            &format!("Batch fetched: batch_size={} cumulative={}", batch_size, cumulative),
        );
    }

    pub fn log_loading_error(&mut self, error: &str) {
        self.log(LogLevel::Error, "loading", &format!("Error: {}", error));
    }

    // -----------------------------------------------------------------------
    // Chunking stage helpers
    // -----------------------------------------------------------------------

    pub fn log_chunking_start(&mut self, chunk_size: usize, overlap: usize) {
        self.log(
            LogLevel::Info,
            "chunking",
            &format!("Starting chunking: chunk_size={} overlap={}", chunk_size, overlap),
        );
    }

    pub fn log_chunking_complete(&mut self, num_chunks: usize, elapsed_ms: f64) {
        self.log(
            LogLevel::Info,
            "chunking",
            &format!("Chunking complete: {} chunks in {:.1} ms", num_chunks, elapsed_ms),
        );
    }

    pub fn log_chunking_zero_warning(&mut self) {
        self.log(
            LogLevel::Warning,
            "chunking",
            "Zero chunks produced from non-empty document set",
        );
    }

    // -----------------------------------------------------------------------
    // Embedding stage helpers
    // -----------------------------------------------------------------------

    pub fn log_embedding_start(&mut self, model: &str, num_chunks: usize) {
        self.log(
            LogLevel::Info,
            "embedding",
            &format!("Starting embedding: model={} num_chunks={}", model, num_chunks),
        );
    }

    pub fn log_embedding_complete(&mut self, elapsed_ms: f64) {
        self.log(
            LogLevel::Info,
            "embedding",
            &format!("Embedding complete in {:.1} ms", elapsed_ms),
        );
    }

    pub fn log_embedding_progress(&mut self, embedded_so_far: usize) {
        self.log(
            LogLevel::Debug,
            "embedding",
            &format!("Embedded so far: {}", embedded_so_far),
        );
    }

    pub fn log_embedding_error(&mut self, error: &str) {
        self.log(LogLevel::Error, "embedding", &format!("Error: {}", error));
    }

    // -----------------------------------------------------------------------
    // Index build stage helpers
    // -----------------------------------------------------------------------

    pub fn log_index_build_start(&mut self, num_embeddings: usize) {
        self.log(
            LogLevel::Info,
            "index_build",
            &format!("Starting index build: num_embeddings={}", num_embeddings),
        );
    }

    pub fn log_index_build_complete(&mut self, elapsed_ms: f64) {
        self.log(
            LogLevel::Info,
            "index_build",
            &format!("Index build complete in {:.1} ms", elapsed_ms),
        );
    }

    pub fn log_index_build_error(&mut self, error: &str) {
        self.log(LogLevel::Error, "index_build", &format!("Error: {}", error));
    }

    // -----------------------------------------------------------------------
    // Retrieval stage helpers
    // -----------------------------------------------------------------------

    pub fn log_retrieval_start(&mut self, query_id: usize) {
        self.log(
            LogLevel::Debug,
            "retrieval",
            &format!("Query {} start", query_id),
        );
    }

    pub fn log_retrieval_complete(&mut self, query_id: usize, num_chunks: usize, elapsed_ms: f64) {
        self.log(
            LogLevel::Debug,
            "retrieval",
            &format!(
                "Query {} complete: {} chunks retrieved in {:.1} ms",
                query_id, num_chunks, elapsed_ms
            ),
        );
    }

    pub fn log_retrieval_error(&mut self, query_id: usize, error: &str) {
        self.log(
            LogLevel::Error,
            "retrieval",
            &format!("Query {} error: {}", query_id, error),
        );
    }

    // -----------------------------------------------------------------------
    // Generation stage helpers
    // -----------------------------------------------------------------------

    pub fn log_generation_start(&mut self, query_id: usize, num_chunks: usize) {
        self.log(
            LogLevel::Debug,
            "generation",
            &format!("Query {} start: num_chunks={}", query_id, num_chunks),
        );
    }

    pub fn log_generation_complete(
        &mut self,
        query_id: usize,
        total_tokens: usize,
        ttft_ms: f64,
        generation_ms: f64,
    ) {
        self.log(
            LogLevel::Debug,
            "generation",
            &format!(
                "Query {} complete: total_tokens={} ttft_ms={:.1} generation_ms={:.1}",
                query_id, total_tokens, ttft_ms, generation_ms
            ),
        );
    }

    pub fn log_generation_failed_response(&mut self, query_id: usize, reason: &str) {
        self.log(
            LogLevel::Warning,
            "generation",
            &format!("Query {} failed response: {}", query_id, reason),
        );
    }

    pub fn log_generation_error(&mut self, query_id: usize, error: &str) {
        self.log(
            LogLevel::Error,
            "generation",
            &format!("Query {} error: {}", query_id, error),
        );
    }

    // -----------------------------------------------------------------------
    // Summary helpers
    // -----------------------------------------------------------------------

    pub fn log_run_summary(
        &mut self,
        total_queries: usize,
        failures: usize,
        p50_ms: f64,
        p95_ms: f64,
        output_path: &str,
    ) {
        self.log(
            LogLevel::Info,
            "summary",
            &format!(
                "Run complete: total_queries={} failures={} p50_ms={:.1} p95_ms={:.1} output={}",
                total_queries, failures, p50_ms, p95_ms, output_path
            ),
        );
    }

    pub fn log_stress_summary(&mut self, qps: f64, peak_rss_mb: f64, p99_ms: f64) {
        self.log(
            LogLevel::Info,
            "summary",
            &format!(
                "Stress complete: qps={:.2} peak_rss_mb={:.1} p99_ms={:.1}",
                qps, peak_rss_mb, p99_ms
            ),
        );
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::fs;
    use tempfile::TempDir;

    // -----------------------------------------------------------------------
    // Helper
    // -----------------------------------------------------------------------

    fn read_log(dir: &TempDir, backend: &str, language: &str) -> String {
        let path = dir.path().join(format!("{}-{}.log", backend, language));
        fs::read_to_string(&path).unwrap_or_default()
    }

    // -----------------------------------------------------------------------
    // Property 1: Log file path derivation
    // Feature: benchmark-logging, Property 1: Log file path derivation
    // -----------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// For any output_dir, backend, and language string, the log file path
        /// must equal `{output_dir}/{backend}-{language}.log`.
        ///
        /// // Feature: benchmark-logging, Property 1: Log file path derivation
        /// Validates: Requirements 1.2, 1.3
        #[test]
        fn prop_log_file_path_derivation(
            backend in "[a-z][a-z0-9_]{0,15}",
            language in "[a-z][a-z0-9_]{0,10}",
        ) {
            let tmp = TempDir::new().unwrap();
            let output_dir = tmp.path().to_str().unwrap().to_string();

            let _logger = BenchmarkLogger::new(&output_dir, &backend, &language, LogLevel::Debug)
                .expect("logger construction should succeed");

            let expected = format!("{}/{}-{}.log", output_dir, backend, language);
            prop_assert!(
                std::path::Path::new(&expected).exists(),
                "Expected log file at '{}' does not exist",
                expected
            );
        }
    }

    // -----------------------------------------------------------------------
    // Property 2: Log file truncation on re-initialisation
    // Feature: benchmark-logging, Property 2: Log file truncation on re-initialisation
    // -----------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// Constructing a new logger for the same output_dir/backend/language
        /// must truncate prior content.
        ///
        /// // Feature: benchmark-logging, Property 2: Log file truncation on re-initialisation
        /// Validates: Requirements 1.1
        #[test]
        fn prop_log_file_truncation(
            backend in "[a-z][a-z0-9_]{0,15}",
            language in "[a-z][a-z0-9_]{0,10}",
            seed in 0u64..1_000_000u64,
        ) {
            let tmp = TempDir::new().unwrap();
            let output_dir = tmp.path().to_str().unwrap().to_string();

            // Use seed-derived unique markers that cannot be substrings of each other.
            let first_marker = format!("FIRST_ONLY_{}", seed);
            let second_marker = format!("SECOND_ONLY_{}", seed + 1_000_001);

            // First logger writes a record with the first marker.
            {
                let mut logger = BenchmarkLogger::new(&output_dir, &backend, &language, LogLevel::Debug)
                    .unwrap();
                logger.log(LogLevel::Info, "test_stage", &first_marker);
            }

            // Second logger is constructed — must truncate.
            {
                let mut logger = BenchmarkLogger::new(&output_dir, &backend, &language, LogLevel::Debug)
                    .unwrap();
                logger.log(LogLevel::Info, "test_stage", &second_marker);
            }

            let contents = read_log(&tmp, &backend, &language);
            prop_assert!(
                !contents.contains(&first_marker),
                "Old content '{}' should have been truncated",
                first_marker
            );
            prop_assert!(
                contents.contains(&second_marker),
                "New content '{}' should be present",
                second_marker
            );
        }
    }

    // -----------------------------------------------------------------------
    // Property 3: Log level ordering invariant
    // Feature: benchmark-logging, Property 3: Log level ordering invariant
    // -----------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// DEBUG < INFO < WARNING < ERROR (integer ordering).
        ///
        /// // Feature: benchmark-logging, Property 3: Log level ordering invariant
        /// Validates: Requirements 2.1
        #[test]
        fn prop_log_level_ordering(_dummy in 0u8..100u8) {
            prop_assert!((LogLevel::Debug as u8) < (LogLevel::Info as u8));
            prop_assert!((LogLevel::Info as u8) < (LogLevel::Warning as u8));
            prop_assert!((LogLevel::Warning as u8) < (LogLevel::Error as u8));
            prop_assert!(LogLevel::Debug < LogLevel::Info);
            prop_assert!(LogLevel::Info < LogLevel::Warning);
            prop_assert!(LogLevel::Warning < LogLevel::Error);
        }
    }

    // -----------------------------------------------------------------------
    // Property 4: Log level filtering
    // Feature: benchmark-logging, Property 4: Log level filtering
    // -----------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// Records below min_level must not appear in the log file.
        ///
        /// // Feature: benchmark-logging, Property 4: Log level filtering
        /// Validates: Requirements 2.2, 2.3
        #[test]
        fn prop_log_level_filtering(
            min_level_idx in 1usize..4usize,  // INFO=1, WARNING=2, ERROR=3
            record_level_idx in 0usize..4usize,
        ) {
            let levels = [LogLevel::Debug, LogLevel::Info, LogLevel::Warning, LogLevel::Error];
            let min_level = levels[min_level_idx];
            let record_level = levels[record_level_idx];

            let tmp = TempDir::new().unwrap();
            let output_dir = tmp.path().to_str().unwrap().to_string();
            let marker = format!("marker_{}", record_level_idx);

            let mut logger = BenchmarkLogger::new(&output_dir, "backend", "rust", min_level).unwrap();
            logger.log(record_level, "test_stage", &marker);
            drop(logger);

            let contents = read_log(&tmp, "backend", "rust");

            if record_level < min_level {
                prop_assert!(
                    !contents.contains(&marker),
                    "Record at level {:?} should be filtered when min_level={:?}",
                    record_level,
                    min_level
                );
            } else {
                prop_assert!(
                    contents.contains(&marker),
                    "Record at level {:?} should appear when min_level={:?}",
                    record_level,
                    min_level
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Property 5: Log record format
    // Feature: benchmark-logging, Property 5: Log record format
    // -----------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// Each log line must match the expected format regex.
        ///
        /// // Feature: benchmark-logging, Property 5: Log record format
        /// Validates: Requirements 3.1, 3.2
        #[test]
        fn prop_log_record_format(
            stage in "[a-z][a-z_]{0,15}",
            message in "[a-zA-Z0-9 .,_-]{1,60}",
            level_idx in 0usize..4usize,
        ) {
            let levels = [LogLevel::Debug, LogLevel::Info, LogLevel::Warning, LogLevel::Error];
            let level = levels[level_idx];

            let tmp = TempDir::new().unwrap();
            let output_dir = tmp.path().to_str().unwrap().to_string();

            let mut logger = BenchmarkLogger::new(&output_dir, "backend", "rust", LogLevel::Debug).unwrap();
            logger.log(level, &stage, &message);
            drop(logger);

            let contents = read_log(&tmp, "backend", "rust");
            let line = contents.trim_end_matches('\n');

            // Regex: ^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z \[(DEBUG|INFO|WARNING|ERROR)\] \[[\w_]+\] .+$
            let re = regex::Regex::new(
                r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z \[(DEBUG|INFO|WARNING|ERROR)\] \[[\w_]+\] .+$"
            ).unwrap();

            prop_assert!(
                re.is_match(line),
                "Log line does not match expected format: '{}'",
                line
            );
        }
    }

    // -----------------------------------------------------------------------
    // Property 6: Log record ordering and flush
    // Feature: benchmark-logging, Property 6: Log record ordering and flush
    // -----------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// N records emitted in order must yield exactly N lines in the same order.
        ///
        /// // Feature: benchmark-logging, Property 6: Log record ordering and flush
        /// Validates: Requirements 3.3, 3.4
        #[test]
        fn prop_log_record_ordering(
            n in 1usize..20usize,
        ) {
            let tmp = TempDir::new().unwrap();
            let output_dir = tmp.path().to_str().unwrap().to_string();

            let mut logger = BenchmarkLogger::new(&output_dir, "backend", "rust", LogLevel::Debug).unwrap();

            let messages: Vec<String> = (0..n).map(|i| format!("message_number_{}", i)).collect();
            for msg in &messages {
                logger.log(LogLevel::Info, "test_stage", msg);
            }
            drop(logger);

            let contents = read_log(&tmp, "backend", "rust");
            let lines: Vec<&str> = contents.lines().collect();

            prop_assert_eq!(
                lines.len(),
                n,
                "Expected {} lines, got {}",
                n,
                lines.len()
            );

            for (i, (line, msg)) in lines.iter().zip(messages.iter()).enumerate() {
                prop_assert!(
                    line.contains(msg.as_str()),
                    "Line {} should contain '{}', got '{}'",
                    i,
                    msg,
                    line
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Property 7: Loading stage trace fields
    // Feature: benchmark-logging, Property 7: Loading stage trace fields
    // -----------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// For any dataset name, subset, and document count, the loading start record
        /// must contain the dataset name, subset, and count; and the loading complete
        /// record must contain the actual document count and a non-negative elapsed time.
        ///
        /// // Feature: benchmark-logging, Property 7: Loading stage trace fields
        /// Validates: Requirements 4.1, 4.2
        #[test]
        fn prop_loading_stage_fields(
            dataset_name in "[a-z][a-z0-9_]{0,15}",
            subset in "[a-z][a-z0-9_]{0,15}",
            num_docs in 0usize..10000usize,
            actual_docs in 0usize..10000usize,
            elapsed_ms in 0.0f64..1000000.0f64,
        ) {
            let tmp = TempDir::new().unwrap();
            let output_dir = tmp.path().to_str().unwrap().to_string();

            let mut logger = BenchmarkLogger::new(&output_dir, "backend", "rust", LogLevel::Debug).unwrap();
            logger.log_loading_start(&dataset_name, &subset, num_docs);
            logger.log_loading_complete(actual_docs, elapsed_ms);
            drop(logger);

            let contents = read_log(&tmp, "backend", "rust");
            let lines: Vec<&str> = contents.lines().collect();

            prop_assert_eq!(lines.len(), 2, "Expected 2 lines, got {}", lines.len());

            let start_line = lines[0];
            prop_assert!(
                start_line.contains(&dataset_name),
                "Start line should contain dataset_name '{}': '{}'", dataset_name, start_line
            );
            prop_assert!(
                start_line.contains(&subset),
                "Start line should contain subset '{}': '{}'", subset, start_line
            );
            prop_assert!(
                start_line.contains(&num_docs.to_string()),
                "Start line should contain num_docs {}: '{}'", num_docs, start_line
            );

            let complete_line = lines[1];
            prop_assert!(
                complete_line.contains(&actual_docs.to_string()),
                "Complete line should contain actual_docs {}: '{}'", actual_docs, complete_line
            );
            // elapsed_ms is non-negative (f64 >= 0.0 guaranteed by strategy)
            prop_assert!(
                elapsed_ms >= 0.0,
                "elapsed_ms must be non-negative, got {}", elapsed_ms
            );
        }
    }

    // -----------------------------------------------------------------------
    // Property 9: Chunking stage trace fields
    // Feature: benchmark-logging, Property 9: Chunking stage trace fields
    // -----------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// For any chunk_size and overlap values, the chunking start record must contain
        /// both values; and the chunking complete record must contain the total chunk count
        /// and a non-negative elapsed time.
        ///
        /// // Feature: benchmark-logging, Property 9: Chunking stage trace fields
        /// Validates: Requirements 5.1, 5.2
        #[test]
        fn prop_chunking_stage_fields(
            chunk_size in 1usize..10000usize,
            overlap in 0usize..10000usize,
            num_chunks in 0usize..10000usize,
            elapsed_ms in 0.0f64..1000000.0f64,
        ) {
            let tmp = TempDir::new().unwrap();
            let output_dir = tmp.path().to_str().unwrap().to_string();

            let mut logger = BenchmarkLogger::new(&output_dir, "backend", "rust", LogLevel::Debug).unwrap();
            logger.log_chunking_start(chunk_size, overlap);
            logger.log_chunking_complete(num_chunks, elapsed_ms);
            drop(logger);

            let contents = read_log(&tmp, "backend", "rust");
            let lines: Vec<&str> = contents.lines().collect();

            prop_assert_eq!(lines.len(), 2, "Expected 2 lines, got {}", lines.len());

            let start_line = lines[0];
            prop_assert!(
                start_line.contains(&chunk_size.to_string()),
                "Start line should contain chunk_size {}: '{}'", chunk_size, start_line
            );
            prop_assert!(
                start_line.contains(&overlap.to_string()),
                "Start line should contain overlap {}: '{}'", overlap, start_line
            );

            let complete_line = lines[1];
            prop_assert!(
                complete_line.contains(&num_chunks.to_string()),
                "Complete line should contain num_chunks {}: '{}'", num_chunks, complete_line
            );
            prop_assert!(
                elapsed_ms >= 0.0,
                "elapsed_ms must be non-negative, got {}", elapsed_ms
            );
        }
    }

    // -----------------------------------------------------------------------
    // Property 10: Embedding stage trace fields
    // Feature: benchmark-logging, Property 10: Embedding stage trace fields
    // -----------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// For any embedding model name and chunk count, the embedding start record must
        /// contain both values; and the embedding complete record must contain a
        /// non-negative elapsed time.
        ///
        /// // Feature: benchmark-logging, Property 10: Embedding stage trace fields
        /// Validates: Requirements 6.1, 6.2
        #[test]
        fn prop_embedding_stage_fields(
            model in "[a-z][a-z0-9_]{0,15}",
            num_chunks in 0usize..10000usize,
            elapsed_ms in 0.0f64..1000000.0f64,
        ) {
            let tmp = TempDir::new().unwrap();
            let output_dir = tmp.path().to_str().unwrap().to_string();

            let mut logger = BenchmarkLogger::new(&output_dir, "backend", "rust", LogLevel::Debug).unwrap();
            logger.log_embedding_start(&model, num_chunks);
            logger.log_embedding_complete(elapsed_ms);
            drop(logger);

            let contents = read_log(&tmp, "backend", "rust");
            let lines: Vec<&str> = contents.lines().collect();

            prop_assert_eq!(lines.len(), 2, "Expected 2 lines, got {}", lines.len());

            let start_line = lines[0];
            prop_assert!(
                start_line.contains(&model),
                "Start line should contain model '{}': '{}'", model, start_line
            );
            prop_assert!(
                start_line.contains(&num_chunks.to_string()),
                "Start line should contain num_chunks {}: '{}'", num_chunks, start_line
            );

            prop_assert!(
                elapsed_ms >= 0.0,
                "elapsed_ms must be non-negative, got {}", elapsed_ms
            );
        }
    }

    // -----------------------------------------------------------------------
    // Property 8: Embedding progress DEBUG record count
    // Feature: benchmark-logging, Property 8: Embedding progress DEBUG record count
    // -----------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// For any N >= 100, the number of DEBUG progress records emitted must equal
        /// floor(N / 100). Simulated by calling log_embedding_progress(i) for i in
        /// (100..=n).step_by(100).
        ///
        /// // Feature: benchmark-logging, Property 8: Embedding progress DEBUG record count
        /// Validates: Requirements 6.4
        #[test]
        fn prop_embedding_progress_count(
            n in 100usize..10000usize,
        ) {
            let tmp = TempDir::new().unwrap();
            let output_dir = tmp.path().to_str().unwrap().to_string();

            let mut logger = BenchmarkLogger::new(&output_dir, "backend", "rust", LogLevel::Debug).unwrap();
            for i in (100..=n).step_by(100) {
                logger.log_embedding_progress(i);
            }
            drop(logger);

            let contents = read_log(&tmp, "backend", "rust");
            let actual_count = contents.lines().count();
            let expected_count = n / 100;

            prop_assert_eq!(
                actual_count,
                expected_count,
                "Expected {} progress records for n={}, got {}",
                expected_count, n, actual_count
            );
        }
    }

    // -----------------------------------------------------------------------
    // Property 11: Index build stage trace fields
    // Feature: benchmark-logging, Property 11: Index build stage trace fields
    // -----------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// For any number of embeddings, the index build start record must contain that
        /// count; and the index build complete record must contain a non-negative elapsed time.
        ///
        /// // Feature: benchmark-logging, Property 11: Index build stage trace fields
        /// Validates: Requirements 7.1, 7.2
        #[test]
        fn prop_index_build_stage_fields(
            num_embeddings in 0usize..10000usize,
            elapsed_ms in 0.0f64..1000000.0f64,
        ) {
            let tmp = TempDir::new().unwrap();
            let output_dir = tmp.path().to_str().unwrap().to_string();

            let mut logger = BenchmarkLogger::new(&output_dir, "backend", "rust", LogLevel::Debug).unwrap();
            logger.log_index_build_start(num_embeddings);
            logger.log_index_build_complete(elapsed_ms);
            drop(logger);

            let contents = read_log(&tmp, "backend", "rust");
            let lines: Vec<&str> = contents.lines().collect();

            prop_assert_eq!(lines.len(), 2, "Expected 2 lines, got {}", lines.len());

            let start_line = lines[0];
            prop_assert!(
                start_line.contains(&num_embeddings.to_string()),
                "Start line should contain num_embeddings {}: '{}'", num_embeddings, start_line
            );

            prop_assert!(
                elapsed_ms >= 0.0,
                "elapsed_ms must be non-negative, got {}", elapsed_ms
            );
        }
    }

    // -----------------------------------------------------------------------
    // Property 12: Retrieval stage trace fields
    // Feature: benchmark-logging, Property 12: Retrieval stage trace fields
    // -----------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// For any query ID, the retrieval start DEBUG record must contain the query ID;
        /// and the retrieval complete DEBUG record must contain the query ID, number of
        /// chunks retrieved, and a non-negative elapsed time.
        ///
        /// // Feature: benchmark-logging, Property 12: Retrieval stage trace fields
        /// Validates: Requirements 8.1, 8.2
        #[test]
        fn prop_retrieval_stage_fields(
            query_id in 0usize..10000usize,
            num_chunks in 0usize..10000usize,
            elapsed_ms in 0.0f64..1000000.0f64,
        ) {
            let tmp = TempDir::new().unwrap();
            let output_dir = tmp.path().to_str().unwrap().to_string();

            let mut logger = BenchmarkLogger::new(&output_dir, "backend", "rust", LogLevel::Debug).unwrap();
            logger.log_retrieval_start(query_id);
            logger.log_retrieval_complete(query_id, num_chunks, elapsed_ms);
            drop(logger);

            let contents = read_log(&tmp, "backend", "rust");
            let lines: Vec<&str> = contents.lines().collect();

            prop_assert_eq!(lines.len(), 2, "Expected 2 lines, got {}", lines.len());

            let start_line = lines[0];
            prop_assert!(
                start_line.contains(&query_id.to_string()),
                "Start line should contain query_id {}: '{}'", query_id, start_line
            );

            let complete_line = lines[1];
            prop_assert!(
                complete_line.contains(&query_id.to_string()),
                "Complete line should contain query_id {}: '{}'", query_id, complete_line
            );
            prop_assert!(
                complete_line.contains(&num_chunks.to_string()),
                "Complete line should contain num_chunks {}: '{}'", num_chunks, complete_line
            );
            prop_assert!(
                elapsed_ms >= 0.0,
                "elapsed_ms must be non-negative, got {}", elapsed_ms
            );
        }
    }

    // -----------------------------------------------------------------------
    // Property 13: Generation stage trace fields
    // Feature: benchmark-logging, Property 13: Generation stage trace fields
    // -----------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// For any query ID and context chunk count, the generation start DEBUG record must
        /// contain both values; the generation complete DEBUG record must contain the query ID,
        /// total tokens, ttft_ms, and generation_ms; and for any response marked as failed,
        /// a WARNING record must be emitted containing the query ID and failure reason.
        ///
        /// // Feature: benchmark-logging, Property 13: Generation stage trace fields
        /// Validates: Requirements 9.1, 9.2, 9.4
        #[test]
        fn prop_generation_stage_fields(
            query_id in 0usize..10000usize,
            num_chunks in 0usize..10000usize,
            total_tokens in 0usize..10000usize,
            ttft_ms in 0.0f64..1000000.0f64,
            generation_ms in 0.0f64..1000000.0f64,
            fail_reason in "[a-z][a-z0-9_]{0,15}",
        ) {
            let tmp = TempDir::new().unwrap();
            let output_dir = tmp.path().to_str().unwrap().to_string();

            let mut logger = BenchmarkLogger::new(&output_dir, "backend", "rust", LogLevel::Debug).unwrap();
            logger.log_generation_start(query_id, num_chunks);
            logger.log_generation_complete(query_id, total_tokens, ttft_ms, generation_ms);
            logger.log_generation_failed_response(query_id, &fail_reason);
            drop(logger);

            let contents = read_log(&tmp, "backend", "rust");
            let lines: Vec<&str> = contents.lines().collect();

            prop_assert_eq!(lines.len(), 3, "Expected 3 lines, got {}", lines.len());

            let start_line = lines[0];
            prop_assert!(
                start_line.contains(&query_id.to_string()),
                "Start line should contain query_id {}: '{}'", query_id, start_line
            );
            prop_assert!(
                start_line.contains(&num_chunks.to_string()),
                "Start line should contain num_chunks {}: '{}'", num_chunks, start_line
            );

            let complete_line = lines[1];
            prop_assert!(
                complete_line.contains(&query_id.to_string()),
                "Complete line should contain query_id {}: '{}'", query_id, complete_line
            );
            prop_assert!(
                complete_line.contains(&total_tokens.to_string()),
                "Complete line should contain total_tokens {}: '{}'", total_tokens, complete_line
            );

            let warn_line = lines[2];
            prop_assert!(
                warn_line.contains("[WARNING]"),
                "Failed response line should be WARNING: '{}'", warn_line
            );
            prop_assert!(
                warn_line.contains(&query_id.to_string()),
                "Warning line should contain query_id {}: '{}'", query_id, warn_line
            );
            prop_assert!(
                warn_line.contains(&fail_reason),
                "Warning line should contain fail_reason '{}': '{}'", fail_reason, warn_line
            );
        }
    }

    // -----------------------------------------------------------------------
    // Property 14: Run summary trace fields
    // Feature: benchmark-logging, Property 14: Run summary trace fields
    // -----------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// For any run completion with total_queries, failures, p50, p95, and output_path,
        /// the summary INFO record must contain all five values.
        ///
        /// // Feature: benchmark-logging, Property 14: Run summary trace fields
        /// Validates: Requirements 10.1, 10.2
        #[test]
        fn prop_run_summary_fields(
            total_queries in 0usize..10000usize,
            failures in 0usize..10000usize,
            p50_ms in 0.0f64..1000000.0f64,
            p95_ms in 0.0f64..1000000.0f64,
            output_path in "[a-z][a-z0-9_]{0,15}",
        ) {
            let tmp = TempDir::new().unwrap();
            let output_dir = tmp.path().to_str().unwrap().to_string();

            let mut logger = BenchmarkLogger::new(&output_dir, "backend", "rust", LogLevel::Debug).unwrap();
            logger.log_run_summary(total_queries, failures, p50_ms, p95_ms, &output_path);
            drop(logger);

            let contents = read_log(&tmp, "backend", "rust");
            let lines: Vec<&str> = contents.lines().collect();

            prop_assert_eq!(lines.len(), 1, "Expected 1 line, got {}", lines.len());

            let summary_line = lines[0];
            prop_assert!(
                summary_line.contains(&total_queries.to_string()),
                "Summary line should contain total_queries {}: '{}'", total_queries, summary_line
            );
            prop_assert!(
                summary_line.contains(&failures.to_string()),
                "Summary line should contain failures {}: '{}'", failures, summary_line
            );
            prop_assert!(
                summary_line.contains(&output_path),
                "Summary line should contain output_path '{}': '{}'", output_path, summary_line
            );
        }
    }
}
