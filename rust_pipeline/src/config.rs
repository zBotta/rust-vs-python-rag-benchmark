//! Configuration loader for the RAG benchmark.
//!
//! Reads `benchmark_config.toml`, validates required keys, and returns a
//! typed `BenchmarkConfig` struct. Missing required keys produce a
//! descriptive `ConfigError` that names the absent key.

use serde::Deserialize;
use std::path::Path;
use thiserror::Error;

/// All errors that can arise during configuration loading.
#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("Missing required configuration key: '{0}'")]
    MissingKey(String),

    #[error("Configuration validation error: {0}")]
    Validation(String),

    #[error("Failed to read config file '{path}': {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("Failed to parse config file: {0}")]
    Parse(#[from] toml::de::Error),
}

/// Raw TOML representation — all fields are `Option<T>` so we can detect
/// missing keys and emit a descriptive error rather than a generic parse error.
#[derive(Debug, Deserialize)]
struct RawConfig {
    dataset_name: Option<String>,
    dataset_subset: Option<String>,
    num_documents: Option<u64>,
    chunk_size: Option<u64>,
    chunk_overlap: Option<u64>,
    embedding_model: Option<String>,
    top_k: Option<u64>,
    llm_model: Option<String>,
    llm_host: Option<String>,
    query_set_path: Option<String>,
    output_dir: Option<String>,
    // Optional with defaults
    llm_backend: Option<String>,
    gguf_model_path: Option<String>,
    log_level: Option<String>,
    // Optional subsection
    stress_test: Option<RawStressTestConfig>,
}

/// Raw TOML representation of the `[stress_test]` subsection.
#[derive(Debug, Deserialize)]
struct RawStressTestConfig {
    enabled: Option<bool>,
    concurrency: Option<u64>,
    num_documents: Option<u64>,
    query_repetitions: Option<u64>,
}

/// Validated stress test configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct StressTestConfig {
    pub enabled: bool,
    pub concurrency: usize,
    pub num_documents: usize,
    pub query_repetitions: usize,
}

/// Validated benchmark configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct BenchmarkConfig {
    pub dataset_name: String,
    pub dataset_subset: String,
    pub num_documents: usize,
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub embedding_model: String,
    pub top_k: usize,
    pub llm_model: String,
    pub llm_host: String,
    pub query_set_path: String,
    pub output_dir: String,
    pub llm_backend: String,
    pub gguf_model_path: String,
    pub log_level: String,
    pub stress_test: StressTestConfig,
}

/// Helper: extract a required `Option<String>` field or return `ConfigError::MissingKey`.
macro_rules! require_str {
    ($raw:expr, $field:ident) => {
        $raw.$field
            .ok_or_else(|| ConfigError::MissingKey(stringify!($field).to_string()))?
    };
}

/// Helper: extract a required `Option<u64>` field or return `ConfigError::MissingKey`.
macro_rules! require_u64 {
    ($raw:expr, $field:ident) => {
        $raw.$field
            .ok_or_else(|| ConfigError::MissingKey(stringify!($field).to_string()))? as usize
    };
}

/// Load and validate benchmark configuration from a TOML file.
///
/// # Errors
///
/// Returns [`ConfigError::MissingKey`] (naming the absent key) when any
/// required key is absent from the file.
pub fn load_config(config_path: impl AsRef<Path>) -> Result<BenchmarkConfig, ConfigError> {
    let path = config_path.as_ref();
    let contents = std::fs::read_to_string(path).map_err(|e| ConfigError::Io {
        path: path.display().to_string(),
        source: e,
    })?;

    let raw: RawConfig = toml::from_str(&contents)?;

    // Apply defaults for optional fields.
    let llm_backend = raw.llm_backend.unwrap_or_else(|| "ollama_http".to_string());
    let gguf_model_path = raw.gguf_model_path.unwrap_or_default();
    let log_level = raw.log_level.unwrap_or_else(|| "INFO".to_string());

    // Validate: gguf_model_path is required for in-process backends.
    if (llm_backend == "llama_cpp" || llm_backend == "llm_rs") && gguf_model_path.is_empty() {
        return Err(ConfigError::MissingKey("gguf_model_path".to_string()));
    }

    // Parse optional [stress_test] subsection with documented defaults.
    let stress_raw = raw.stress_test.unwrap_or(RawStressTestConfig {
        enabled: None,
        concurrency: None,
        num_documents: None,
        query_repetitions: None,
    });
    let stress_concurrency = stress_raw.concurrency.unwrap_or(8) as usize;
    if stress_concurrency < 1 {
        return Err(ConfigError::Validation(
            "stress_test.concurrency must be >= 1".to_string(),
        ));
    }
    let stress_test = StressTestConfig {
        enabled: stress_raw.enabled.unwrap_or(false),
        concurrency: stress_concurrency,
        num_documents: stress_raw.num_documents.unwrap_or(10000) as usize,
        query_repetitions: stress_raw.query_repetitions.unwrap_or(10) as usize,
    };

    Ok(BenchmarkConfig {
        dataset_name: require_str!(raw, dataset_name),
        dataset_subset: require_str!(raw, dataset_subset),
        num_documents: require_u64!(raw, num_documents),
        chunk_size: require_u64!(raw, chunk_size),
        chunk_overlap: require_u64!(raw, chunk_overlap),
        embedding_model: require_str!(raw, embedding_model),
        top_k: require_u64!(raw, top_k),
        llm_model: require_str!(raw, llm_model),
        llm_host: require_str!(raw, llm_host),
        query_set_path: require_str!(raw, query_set_path),
        output_dir: require_str!(raw, output_dir),
        llm_backend,
        gguf_model_path,
        log_level,
        stress_test,
    })
}

// ---------------------------------------------------------------------------
// Property-based tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// All required configuration keys.
    const REQUIRED_KEYS: &[&str] = &[
        "dataset_name",
        "dataset_subset",
        "num_documents",
        "chunk_size",
        "chunk_overlap",
        "embedding_model",
        "top_k",
        "llm_model",
        "llm_host",
        "query_set_path",
        "output_dir",
    ];

    /// Optional keys with their documented default values (as TOML-formatted strings).
    const OPTIONAL_KEYS_AND_DEFAULTS: &[(&str, &str)] = &[
        ("llm_backend", "ollama_http"),
        ("gguf_model_path", ""),
        ("log_level", "INFO"),
    ];
    /// Build a complete valid TOML string.
    fn full_toml() -> String {
        r#"
dataset_name    = "wikipedia"
dataset_subset  = "20220301.simple"
num_documents   = 1000
chunk_size      = 512
chunk_overlap   = 64
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
top_k           = 5
llm_model       = "llama3.2:3b"
llm_host        = "http://localhost:11434"
query_set_path  = "query_set.json"
output_dir      = "output/"
"#
        .to_string()
    }

    /// Write `content` to a temp file and return the file handle.
    fn write_temp(content: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f
    }

    // -----------------------------------------------------------------------
    // Property 15: Missing required config key produces descriptive error
    // Feature: rust-vs-python-rag-benchmark, Property 15: Missing required config key produces descriptive error
    // -----------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// For any required key that is removed from the config, load_config must
        /// return a ConfigError::MissingKey whose message contains the key name.
        ///
        /// Feature: rust-vs-python-rag-benchmark, Property 15: Missing required config key produces descriptive error
        #[test]
        fn prop_missing_required_key_produces_descriptive_error(
            key_idx in 0usize..REQUIRED_KEYS.len()
        ) {
            let missing_key = REQUIRED_KEYS[key_idx];

            // Build a TOML that omits exactly one required key.
            let toml_content: String = full_toml()
                .lines()
                .filter(|line| {
                    // Drop the line that starts with the key name.
                    let trimmed = line.trim_start();
                    !trimmed.starts_with(missing_key)
                })
                .collect::<Vec<_>>()
                .join("\n");

            let f = write_temp(&toml_content);
            let result = load_config(f.path());

            prop_assert!(
                result.is_err(),
                "Expected error for missing key '{}', but got Ok",
                missing_key
            );

            let err = result.unwrap_err();
            let err_msg = err.to_string();
            prop_assert!(
                err_msg.contains(missing_key),
                "Error message '{}' does not mention missing key '{}'",
                err_msg,
                missing_key
            );
        }
    }

    // -----------------------------------------------------------------------
    // Property 16: Absent optional config key uses documented default
    // Feature: rust-vs-python-rag-benchmark, Property 16: Absent optional config key uses documented default
    // -----------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// For each optional key, omitting it from the config must result in the
        /// documented default value being applied.
        ///
        /// Feature: rust-vs-python-rag-benchmark, Property 16: Absent optional config key uses documented default
        #[test]
        fn prop_absent_optional_key_uses_default(
            key_idx in 0usize..OPTIONAL_KEYS_AND_DEFAULTS.len()
        ) {
            let (key, expected_default) = OPTIONAL_KEYS_AND_DEFAULTS[key_idx];

            // Build a TOML that omits the optional key.
            let toml_content: String = full_toml()
                .lines()
                .filter(|line| {
                    let trimmed = line.trim_start();
                    !trimmed.starts_with(key)
                })
                .collect::<Vec<_>>()
                .join("\n");

            let f = write_temp(&toml_content);
            let result = load_config(f.path());

            prop_assert!(
                result.is_ok(),
                "Config without optional key '{}' should load successfully: {:?}",
                key,
                result
            );

            let cfg = result.unwrap();
            let actual = match key {
                "llm_backend" => cfg.llm_backend.clone(),
                "gguf_model_path" => cfg.gguf_model_path.clone(),
                "log_level" => cfg.log_level.clone(),
                _ => panic!("Unknown optional key: {}", key),
            };
            prop_assert_eq!(
                actual.as_str(),
                expected_default,
                "Default for '{}' should be '{}', got '{}'",
                key,
                expected_default,
                actual
            );
        }
    }

    // -----------------------------------------------------------------------
    // Property 21: gguf_model_path required for in-process backends
    // Feature: rust-vs-python-rag-benchmark, Property 21: Config with llm_backend=llama_cpp or llm_rs and absent gguf_model_path → error naming gguf_model_path
    // -----------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// For any config where llm_backend is "llama_cpp" or "llm_rs" and
        /// gguf_model_path is absent or empty string, load_config must return
        /// a ConfigError whose message contains "gguf_model_path".
        ///
        /// Feature: rust-vs-python-rag-benchmark, Property 21: Config with llm_backend=llama_cpp or llm_rs and absent gguf_model_path → error naming gguf_model_path
        #[test]
        fn prop_gguf_model_path_required_for_in_process_backends(
            backend_idx in 0usize..2usize,
            gguf_present in proptest::bool::ANY,
        ) {
            let backend = if backend_idx == 0 { "llama_cpp" } else { "llm_rs" };

            // Build a TOML with the in-process backend and either absent or empty gguf_model_path.
            let gguf_line = if gguf_present {
                // Empty string — treated as absent.
                r#"gguf_model_path = """#.to_string()
            } else {
                // Key entirely absent — omit the line.
                String::new()
            };

            let toml_content = format!(
                r#"
dataset_name    = "wikipedia"
dataset_subset  = "20220301.simple"
num_documents   = 1000
chunk_size      = 512
chunk_overlap   = 64
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
top_k           = 5
llm_model       = "llama3.2:3b"
llm_host        = "http://localhost:11434"
query_set_path  = "query_set.json"
output_dir      = "output/"
llm_backend     = "{backend}"
{gguf_line}
"#
            );

            let f = write_temp(&toml_content);
            let result = load_config(f.path());

            prop_assert!(
                result.is_err(),
                "Expected error for backend '{}' with absent/empty gguf_model_path, but got Ok",
                backend
            );

            let err = result.unwrap_err();
            let err_msg = err.to_string();
            prop_assert!(
                err_msg.contains("gguf_model_path"),
                "Error message '{}' does not mention 'gguf_model_path'",
                err_msg
            );
        }
    }

    // -----------------------------------------------------------------------
    // Unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn full_config_loads_correctly() {
        let f = write_temp(&full_toml());
        let cfg = load_config(f.path()).unwrap();
        assert_eq!(cfg.dataset_name, "wikipedia");
        assert_eq!(cfg.num_documents, 1000);
        assert_eq!(cfg.chunk_size, 512);
        assert_eq!(cfg.top_k, 5);
        // stress_test defaults when section is absent
        assert!(!cfg.stress_test.enabled);
        assert_eq!(cfg.stress_test.concurrency, 8);
        assert_eq!(cfg.stress_test.num_documents, 10000);
        assert_eq!(cfg.stress_test.query_repetitions, 10);
    }

    #[test]
    fn stress_test_section_parsed_correctly() {
        let toml_content = format!(
            "{}\n[stress_test]\nenabled = true\nconcurrency = 4\nnum_documents = 500\nquery_repetitions = 3\n",
            full_toml()
        );
        let f = write_temp(&toml_content);
        let cfg = load_config(f.path()).unwrap();
        assert!(cfg.stress_test.enabled);
        assert_eq!(cfg.stress_test.concurrency, 4);
        assert_eq!(cfg.stress_test.num_documents, 500);
        assert_eq!(cfg.stress_test.query_repetitions, 3);
    }

    #[test]
    fn stress_test_concurrency_zero_returns_error() {
        let toml_content = format!(
            "{}\n[stress_test]\nconcurrency = 0\n",
            full_toml()
        );
        let f = write_temp(&toml_content);
        let err = load_config(f.path()).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("concurrency"),
            "Error should mention 'concurrency', got: {}",
            msg
        );
    }

    #[test]
    fn missing_each_required_key_names_it_in_error() {
        for &key in REQUIRED_KEYS {
            let toml_content: String = full_toml()
                .lines()
                .filter(|line| !line.trim_start().starts_with(key))
                .collect::<Vec<_>>()
                .join("\n");

            let f = write_temp(&toml_content);
            let err = load_config(f.path()).unwrap_err();
            assert!(
                err.to_string().contains(key),
                "Error for missing '{}' should mention the key; got: {}",
                key,
                err
            );
        }
    }
}
