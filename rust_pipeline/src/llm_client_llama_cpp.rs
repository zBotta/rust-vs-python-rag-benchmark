//! LLM client — Task 16: in-process GGUF inference via the `llama_cpp` crate.
//!
//! `LlamaCppClient` implements the `LlmClient` trait and runs inference
//! in-process using the `llama_cpp` crate (`edgenai/llama_cpp-rs`), which
//! provides high-level safe Rust bindings over the llama.cpp C API.
//!
//! The GGUF model file path is read from `config.gguf_model_path`.
//! The model is loaded once at construction time and reused across `generate()`
//! calls.
//!
//! TTFT is recorded as the elapsed time from the start of inference until the
//! first token string is received from the streaming iterator.
//! Total generation time covers the full token stream until completion.
//!
//! Token counting: the `llama_cpp` crate does not expose prompt/completion
//! token counts in the same way as Ollama; `total_tokens` is recorded as 0
//! for this backend (acceptable for in-process backends where token counting
//! differs).
//!
//! No retry logic is applied — in-process backends surface errors directly.
//!
//! _Requirements: 5.3, 5.4_

use std::time::Instant;

fn env_bool(key: &str, default: bool) -> bool {
    match std::env::var(key) {
        Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" => false,
            _ => default,
        },
        Err(_) => default,
    }
}

fn env_u32(key: &str) -> Option<u32> {
    std::env::var(key).ok().and_then(|v| v.trim().parse::<u32>().ok())
}

#[cfg(feature = "llama_cpp_backend")]
use llama_cpp::{
    standard_sampler::StandardSampler,
    LlamaModel,
    LlamaParams,
    SessionParams,
};

use crate::llm_client::{build_prompt, LlmClient, LlmError, LLMResponse};

// ---------------------------------------------------------------------------
// LlamaCppClient — only available when built with --features llama_cpp_backend
// ---------------------------------------------------------------------------

/// In-process LLM client backed by the `llama_cpp` crate (llama.cpp bindings).
///
/// Loads the GGUF model once at construction time and reuses it across calls.
#[cfg(feature = "llama_cpp_backend")]
pub struct LlamaCppClient {
    model: LlamaModel,
}

#[cfg(feature = "llama_cpp_backend")]
impl LlamaCppClient {
    /// Load the GGUF model from `model_path`.
    ///
    /// Returns an `LlmError` if the file does not exist or fails to load.
    pub fn new(model_path: &str) -> Result<Self, LlmError> {
        if !std::path::Path::new(model_path).exists() {
            return Err(LlmError::RequestFailed(format!(
                "GGUF model file not found: '{}'",
                model_path
            )));
        }

        let mut params = LlamaParams::default();

        // Conservative mode is default for compatibility. Set
        // RUST_LLAMA_CPP_CONSERVATIVE=0 to test less restrictive params.
        let conservative = env_bool("RUST_LLAMA_CPP_CONSERVATIVE", true);
        if conservative {
            params.n_gpu_layers = 0;
            params.use_mmap = false;
            params.use_mlock = false;
        } else {
            params.use_mmap = env_bool("RUST_LLAMA_CPP_USE_MMAP", params.use_mmap);
            params.use_mlock = env_bool("RUST_LLAMA_CPP_USE_MLOCK", params.use_mlock);
            if let Some(v) = env_u32("RUST_LLAMA_CPP_N_GPU_LAYERS") {
                params.n_gpu_layers = v;
            }
        }

        println!(
            "llama_cpp loader params: conservative={}, n_gpu_layers={}, use_mmap={}, use_mlock={}",
            conservative, params.n_gpu_layers, params.use_mmap, params.use_mlock
        );

        let model = LlamaModel::load_from_file(model_path, params).map_err(|e| {
            LlmError::RequestFailed(format!(
                "Failed to load GGUF model from '{}': {:?}",
                model_path, e
            ))
        })?;

        Ok(Self { model })
    }
}

#[cfg(feature = "llama_cpp_backend")]
impl LlmClient for LlamaCppClient {
    fn generate(&self, query: &str, chunks: &[String]) -> Result<LLMResponse, LlmError> {
        let prompt = build_prompt(chunks, query);

        let mut session_params = SessionParams::default();
        // Keep session context aligned with Python llama-cpp settings.
        session_params.n_ctx = 2048;
        session_params.n_batch = 2048;
        session_params.n_ubatch = 2048;

        let mut ctx = match self.model.create_session(session_params) {
            Ok(s) => s,
            Err(e) => {
                return Ok(LLMResponse {
                    text: String::new(),
                    total_tokens: 0,
                    ttft_ms: 0.0,
                    generation_ms: 0.0,
                    failed: true,
                    failure_reason: Some(format!("Failed to create llama_cpp session: {:?}", e)),
                });
            }
        };

        if let Err(e) = ctx.advance_context(&prompt) {
            return Ok(LLMResponse {
                text: String::new(),
                total_tokens: 0,
                ttft_ms: 0.0,
                generation_ms: 0.0,
                failed: true,
                failure_reason: Some(format!("Failed to advance context: {:?}", e)),
            });
        }

        let start = Instant::now();
        let mut text_parts: Vec<String> = Vec::new();
        let mut ttft_ms: f64 = 0.0;
        let mut first_token = true;

        let completions = match ctx.start_completing_with(StandardSampler::default(), 256) {
            Ok(c) => c,
            Err(e) => {
                return Ok(LLMResponse {
                    text: String::new(),
                    total_tokens: 0,
                    ttft_ms: 0.0,
                    generation_ms: 0.0,
                    failed: true,
                    failure_reason: Some(format!("Failed to start completion: {:?}", e)),
                });
            }
        };
        for token_str in completions.into_strings() {
            if first_token {
                ttft_ms = start.elapsed().as_secs_f64() * 1000.0;
                first_token = false;
            }
            text_parts.push(token_str);
        }

        let generation_ms = start.elapsed().as_secs_f64() * 1000.0;

        Ok(LLMResponse {
            text: text_parts.join(""),
            total_tokens: 0,
            ttft_ms,
            generation_ms,
            failed: false,
            failure_reason: None,
        })
    }
}

/// Stub used when the `llama_cpp_backend` feature is not enabled.
/// `main.rs` will never instantiate this — it exits with a clear error message.
#[cfg(not(feature = "llama_cpp_backend"))]
#[derive(Debug)]
pub struct LlamaCppClient;

#[cfg(not(feature = "llama_cpp_backend"))]
impl LlamaCppClient {
    pub fn new(model_path: &str) -> Result<Self, LlmError> {
        Err(LlmError::RequestFailed(format!(
            "llama_cpp backend is not compiled in. \
             Install LLVM (winget install --id LLVM.LLVM -e) then rebuild with \
             `cargo build --release --features llama_cpp_backend`. \
             Model path was: '{}'",
            model_path
        )))
    }
}

#[cfg(not(feature = "llama_cpp_backend"))]
impl LlmClient for LlamaCppClient {
    fn generate(&self, _query: &str, _chunks: &[String]) -> Result<LLMResponse, LlmError> {
        Err(LlmError::RequestFailed(
            "llama_cpp backend not compiled in".to_string(),
        ))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm_client::build_prompt;

    /// Verify that `LlamaCppClient::new` returns an error (not a panic) when
    /// the model file does not exist.
    #[test]
    fn test_llama_cpp_client_missing_model_returns_error() {
        let result = LlamaCppClient::new("/nonexistent/path/model.gguf");
        assert!(
            result.is_err(),
            "Expected error when model file does not exist"
        );
        // Both the real client and the stub return an error mentioning the path
        let msg = result.err().unwrap().to_string();
        assert!(
            msg.contains("model.gguf") || msg.contains("llama_cpp backend"),
            "Error message should mention the model path or missing feature; got: {}",
            msg
        );
    }

    /// Verify that the prompt built for the llama_cpp client matches the shared
    /// template — same as Property 6 but exercised through this module.
    #[test]
    fn test_prompt_template_via_llama_cpp_module() {
        let chunks = vec!["chunk A".to_string(), "chunk B".to_string()];
        let query = "What is X?";
        let prompt = build_prompt(&chunks, query);
        assert_eq!(
            prompt,
            "Context:\nchunk A\nchunk B\n\nQuestion: What is X?\nAnswer:"
        );
    }
}
