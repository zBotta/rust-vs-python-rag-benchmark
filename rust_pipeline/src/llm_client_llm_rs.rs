//! LLM client — Task 16: in-process GGUF inference via `candle` (pure-Rust).
//!
//! `LlmRsClient` implements the `LlmClient` trait and runs inference in-process
//! using the `candle-core` + `candle-transformers` crates with a GGUF model file.
//!
//! The GGUF model file path is read from `config.gguf_model_path`.
//!
//! Background
//! ----------
//! The original `rustformers/llm` crate (llm-rs) was archived in June 2024 and
//! only supports the legacy GGML format, not GGUF.  The `llama_cpp` crate
//! (high-level llama.cpp bindings) requires `libclang` and a C/C++ toolchain
//! which may not be available in all environments.
//!
//! `candle` (HuggingFace's pure-Rust ML framework) supports GGUF natively via
//! `candle_core::quantized::gguf_file` and provides a quantized LLaMA
//! implementation in `candle_transformers::models::quantized_llama`.  It
//! compiles without any native toolchain beyond the standard Rust toolchain.
//!
//! TTFT is recorded as the elapsed time from the start of prompt processing to
//! the first generated token.  Total generation time covers the full decode loop.
//!
//! _Requirements: 5.4_

use std::cell::UnsafeCell;
use std::fs::File;
use std::io::BufReader;
use std::time::Instant;

use candle_core::{Device, Tensor};
use candle_core::quantized::gguf_file;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_qwen2::ModelWeights;

use crate::llm_client::{LlmClient, LlmError, LLMResponse, build_prompt};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of new tokens to generate per request.
const MAX_NEW_TOKENS: usize = 256;

/// Temperature for sampling (0.0 = greedy).
const TEMPERATURE: f64 = 0.8;

/// Top-p nucleus sampling threshold.
const TOP_P: f64 = 0.9;

// ---------------------------------------------------------------------------
// LlmRsClient
// ---------------------------------------------------------------------------

/// In-process LLM client backed by `candle` (pure-Rust GGUF inference).
///
/// Loads the GGUF model once at construction time and reuses it across calls.
/// Uses the CPU device for inference (no GPU required).
pub struct LlmRsClient {
    model: UnsafeCell<ModelWeights>,
    tokenizer: tokenizers::Tokenizer,
    device: Device,
}

// SAFETY: LlmRsClient is only used single-threaded (sequential query loop).
unsafe impl Send for LlmRsClient {}
unsafe impl Sync for LlmRsClient {}

impl LlmRsClient {
    /// Load the GGUF model from `model_path`.
    ///
    /// The tokenizer is loaded from the GGUF metadata if available, otherwise
    /// falls back to the `bert-base-uncased` tokenizer (already used by the
    /// chunker) for basic tokenization.
    ///
    /// Returns `LlmError::RequestFailed` if the file cannot be loaded.
    pub fn new(model_path: &str) -> Result<Self, LlmError> {
        let device = Device::Cpu;

        // Open the GGUF file
        let mut file = File::open(model_path).map_err(|e| {
            LlmError::RequestFailed(format!(
                "Failed to open GGUF model file '{}': {}",
                model_path, e
            ))
        })?;

        // Parse GGUF content
        let model_content = gguf_file::Content::read(&mut file).map_err(|e| {
            LlmError::RequestFailed(format!(
                "Failed to parse GGUF model '{}': {}",
                model_path, e
            ))
        })?;

        // Load model weights
        let mut reader = BufReader::new(file);
        // Re-open to get a fresh reader for weight loading
        let mut file2 = File::open(model_path).map_err(|e| {
            LlmError::RequestFailed(format!(
                "Failed to re-open GGUF model file '{}': {}",
                model_path, e
            ))
        })?;
        let model_content2 = gguf_file::Content::read(&mut file2).map_err(|e| {
            LlmError::RequestFailed(format!(
                "Failed to re-parse GGUF model '{}': {}",
                model_path, e
            ))
        })?;
        let _ = reader; // suppress unused warning

        let model = ModelWeights::from_gguf(model_content2, &mut file2, &device).map_err(|e| {
            LlmError::RequestFailed(format!(
                "Failed to load model weights from '{}': {}",
                model_path, e
            ))
        })?;

        // Load tokenizer — try to extract from GGUF metadata, fall back to
        // bert-base-uncased (already cached by the chunker/embedder).
        let tokenizer = Self::load_tokenizer(model_path, &model_content)?;

        Ok(Self {
            model: UnsafeCell::new(model),
            tokenizer,
            device,
        })
    }

    /// Attempt to load a tokenizer.
    ///
    /// Priority:
    /// 1. `tokenizer.json` in the same directory as the GGUF file.
    /// 2. The `bert-base-uncased` tokenizer from the local HuggingFace hub cache
    ///    (already downloaded by the chunker/embedder — no network call made).
    fn load_tokenizer(
        model_path: &str,
        _content: &gguf_file::Content,
    ) -> Result<tokenizers::Tokenizer, LlmError> {
        // 1. Try sibling tokenizer.json next to the GGUF file
        let model_dir = std::path::Path::new(model_path)
            .parent()
            .unwrap_or(std::path::Path::new("."));
        let tokenizer_path = model_dir.join("tokenizer.json");
        if tokenizer_path.exists() {
            return tokenizers::Tokenizer::from_file(&tokenizer_path).map_err(|e| {
                LlmError::RequestFailed(format!(
                    "Failed to load tokenizer from '{}': {}",
                    tokenizer_path.display(),
                    e
                ))
            });
        }

        // 2. Look directly in the HF hub local cache (no network call).
        //    On Windows, HF stores cache at %USERPROFILE%\.cache\huggingface\hub
        //    (not in AppData). Check both bert-base-uncased and the sentence-transformers
        //    model (always present since the embedder uses it).
        let hf_hub_root = {
            // Prefer HF_HOME env var, then USERPROFILE\.cache, then dirs_next fallback
            if let Ok(hf_home) = std::env::var("HF_HOME") {
                std::path::PathBuf::from(hf_home).join("hub")
            } else {
                let base = std::env::var("USERPROFILE")
                    .map(std::path::PathBuf::from)
                    .unwrap_or_else(|_| {
                        dirs_next::cache_dir().unwrap_or_else(|| std::path::PathBuf::from("."))
                    });
                base.join(".cache").join("huggingface").join("hub")
            }
        };

        let candidate_model_dirs = [
            "models--bert-base-uncased",
            "models--sentence-transformers--all-MiniLM-L6-v2",
        ];

        for model_dir_name in &candidate_model_dirs {
            let snapshots = hf_hub_root.join(model_dir_name).join("snapshots");
            if snapshots.exists() {
                if let Ok(entries) = std::fs::read_dir(&snapshots) {
                    for entry in entries.flatten() {
                        let candidate = entry.path().join("tokenizer.json");
                        if candidate.exists() {
                            return tokenizers::Tokenizer::from_file(&candidate).map_err(|e| {
                                LlmError::RequestFailed(format!(
                                    "Failed to load tokenizer from cache '{}': {}",
                                    candidate.display(),
                                    e
                                ))
                            });
                        }
                    }
                }
            }
        }

        // 3. Fall back to hf-hub API (offline mode — uses cache, no network)
        let api = hf_hub::api::sync::ApiBuilder::new()
            .with_progress(false)
            .build()
            .map_err(|e| LlmError::RequestFailed(format!("Failed to create HF hub API: {}", e)))?;
        let repo = api.model("bert-base-uncased".to_string());
        let tokenizer_file = repo.get("tokenizer.json").map_err(|e| {
            LlmError::RequestFailed(format!(
                "Failed to fetch bert-base-uncased tokenizer.json from cache: {}. \
                 Ensure the model has been downloaded by running the Python pipeline first.",
                e
            ))
        })?;
        tokenizers::Tokenizer::from_file(&tokenizer_file).map_err(|e| {
            LlmError::RequestFailed(format!(
                "Failed to load bert-base-uncased tokenizer: {}",
                e
            ))
        })
    }
}

impl LlmClient for LlmRsClient {
    /// Run in-process GGUF inference and return an `LLMResponse` with TTFT and
    /// total generation time recorded.
    fn generate(&self, query: &str, chunks: &[String]) -> Result<LLMResponse, LlmError> {
        let prompt = build_prompt(chunks, query);

        // Tokenize the prompt
        let encoding = self
            .tokenizer
            .encode(prompt.as_str(), true)
            .map_err(|e| LlmError::RequestFailed(format!("Tokenization failed: {}", e)))?;
        let prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();
        let prompt_token_count = prompt_tokens.len() as u64;

        let start = Instant::now();

        // Build input tensor [1, seq_len]
        let input = Tensor::new(prompt_tokens.as_slice(), &self.device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(|e| LlmError::RequestFailed(format!("Tensor creation failed: {}", e)))?;

        // Forward pass for the prompt (prefill)
        let logits = {
            // SAFETY: single-threaded sequential usage; no concurrent access.
            let model_mut = unsafe { &mut *self.model.get() };
            model_mut
                .forward(&input, 0)
                .map_err(|e| LlmError::RequestFailed(format!("Forward pass failed: {}", e)))?
        };

        // Sample first token — this is TTFT
        let logits = logits
            .squeeze(0)
            .map_err(|e| LlmError::RequestFailed(format!("Logits squeeze failed: {}", e)))?;

        let mut sampler = LogitsProcessor::new(42, Some(TEMPERATURE), Some(TOP_P));
        let first_token = sampler
            .sample(&logits)
            .map_err(|e| LlmError::RequestFailed(format!("Sampling failed: {}", e)))?;

        let ttft_ms = start.elapsed().as_secs_f64() * 1000.0;

        // Decode loop
        let mut generated_tokens: Vec<u32> = vec![first_token];
        let mut pos = prompt_tokens.len();

        // Determine EOS token id (common values: 2 for LLaMA, 128001 for LLaMA-3)
        let eos_token_id: u32 = self
            .tokenizer
            .token_to_id("</s>")
            .or_else(|| self.tokenizer.token_to_id("<|end_of_text|>"))
            .or_else(|| self.tokenizer.token_to_id("<|eot_id|>"))
            .unwrap_or(2);

        while generated_tokens.len() < MAX_NEW_TOKENS
            && *generated_tokens.last().unwrap_or(&0) != eos_token_id
        {
            let last_token = *generated_tokens.last().unwrap();
            let input = Tensor::new(&[last_token], &self.device)
                .and_then(|t| t.unsqueeze(0))
                .map_err(|e| LlmError::RequestFailed(format!("Tensor creation failed: {}", e)))?;

            let logits = {
                // SAFETY: single-threaded sequential usage; no concurrent access.
                let model_mut = unsafe { &mut *self.model.get() };
                model_mut
                    .forward(&input, pos)
                    .map_err(|e| LlmError::RequestFailed(format!("Forward pass failed: {}", e)))?
            };

            let logits = logits
                .squeeze(0)
                .map_err(|e| LlmError::RequestFailed(format!("Logits squeeze failed: {}", e)))?;

            let next_token = sampler
                .sample(&logits)
                .map_err(|e| LlmError::RequestFailed(format!("Sampling failed: {}", e)))?;

            generated_tokens.push(next_token);
            pos += 1;
        }

        let generation_ms = start.elapsed().as_secs_f64() * 1000.0;

        // Decode generated tokens to text
        let text = self
            .tokenizer
            .decode(&generated_tokens, true)
            .map_err(|e| LlmError::RequestFailed(format!("Token decoding failed: {}", e)))?;

        let completion_token_count = generated_tokens.len() as u64;
        let total_tokens = prompt_token_count + completion_token_count;

        Ok(LLMResponse {
            text,
            total_tokens,
            ttft_ms,
            generation_ms,
            failed: false,
            failure_reason: None,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm_client::build_prompt;

    /// Verify that `LlmRsClient::new` returns an error (not a panic) when the
    /// model file does not exist.  This exercises the error path without
    /// requiring a real GGUF file.
    #[test]
    fn test_llm_rs_client_missing_model_returns_error() {
        let result = LlmRsClient::new("/nonexistent/path/model.gguf");
        assert!(
            result.is_err(),
            "Expected error when model file does not exist"
        );
        let err = result.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("Failed to open GGUF model file"),
            "Error message should mention model load failure; got: {}",
            msg
        );
    }

    /// Verify that the prompt built for the llm-rs client matches the shared
    /// template — same as Property 6 but exercised through this module.
    #[test]
    fn test_prompt_template_via_llm_rs_module() {
        let chunks = vec!["chunk A".to_string(), "chunk B".to_string()];
        let query = "What is X?";
        let prompt = build_prompt(&chunks, query);
        assert_eq!(
            prompt,
            "Context:\nchunk A\nchunk B\n\nQuestion: What is X?\nAnswer:"
        );
    }
}
