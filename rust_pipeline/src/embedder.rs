//! Embedder — candle BertModel (pure Rust, no ORT/DLL required).
//!
//! Uses `candle_transformers::models::bert::BertModel` with the
//! `sentence-transformers/all-MiniLM-L6-v2` safetensors weights from the
//! local HuggingFace hub cache to produce 384-dimensional embeddings.
//!
//! ## Artifact bundle pinning
//!
//! Both pipelines read the **same** local directory:
//!
//! ```text
//! <hf_hub_root>/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/<REVISION>/
//! ```
//!
//! where `REVISION` is the hard-pinned commit hash `PINNED_REVISION` below
//! (overridable via `BERT_MODEL_REVISION` env var).  This guarantees that
//! `config.json`, `tokenizer.json`, and `model.safetensors` are byte-for-byte
//! identical between the two pipelines — no "first snapshot wins" ambiguity.
//!
//! ## Model file resolution (in order)
//!
//! 1. `BERT_MODEL_DIR` env var — explicit path, used as-is.
//! 2. HF hub cache at the pinned revision (see above).
//!
//! No network calls are made; the model must already be cached locally
//! (it is downloaded automatically by the Python pipeline on first run).

use std::path::PathBuf;
use std::sync::OnceLock;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, HiddenAct, DTYPE};
use thiserror::Error;
use tokenizers::{PaddingParams, Tokenizer};

/// Pinned HF commit hash for sentence-transformers/all-MiniLM-L6-v2.
/// Must match Python embedder._DEFAULT_REVISION.
/// Override with BERT_MODEL_REVISION env var if your local cache differs.
const PINNED_REVISION: &str = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf";

#[derive(Debug, Error)]
pub enum EmbedError {
    #[error("Embedding model error: {0}")]
    ModelError(String),
    #[error("Tokenizer error: {0}")]
    TokenizerError(String),
    #[error("Inference error: {0}")]
    InferenceError(String),
}

// ---------------------------------------------------------------------------
// Lazy-loaded model (loaded once per process)
// ---------------------------------------------------------------------------

struct BertEmbedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

static EMBEDDER: OnceLock<Result<BertEmbedder, String>> = OnceLock::new();

fn get_embedder() -> Result<&'static BertEmbedder, EmbedError> {
    let result = EMBEDDER.get_or_init(|| {
        BertEmbedder::load().map_err(|e| e.to_string())
    });
    result.as_ref().map_err(|e| EmbedError::ModelError(e.clone()))
}

impl BertEmbedder {
    fn load() -> Result<Self, EmbedError> {
        let model_dir = resolve_model_dir()?;

        let config_path = model_dir.join("config.json");
        let weights_path = model_dir.join("model.safetensors");
        let tokenizer_path = model_dir.join("tokenizer.json");

        // Load config
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| EmbedError::ModelError(format!("Cannot read config.json: {e}")))?;
        let mut config: Config = serde_json::from_str(&config_str)
            .map_err(|e| EmbedError::ModelError(format!("Cannot parse config.json: {e}")))?;
        // Use approximate GELU for speed (matches sentence-transformers default)
        config.hidden_act = HiddenAct::GeluApproximate;

        // Load tokenizer
        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| EmbedError::TokenizerError(format!("Cannot load tokenizer: {e}")))?;
        // Configure padding to batch-longest for batched inference
        let pp = PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        };
        tokenizer.with_padding(Some(pp));

        // Load weights
        let device = Device::Cpu;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&weights_path], DTYPE, &device)
                .map_err(|e| EmbedError::ModelError(format!("Cannot load safetensors: {e}")))?
        };

        let model = BertModel::load(vb, &config)
            .map_err(|e| EmbedError::ModelError(format!("Cannot build BertModel: {e}")))?;

        Ok(Self { model, tokenizer, device })
    }
}

// ---------------------------------------------------------------------------
// Model directory resolution
// ---------------------------------------------------------------------------

fn resolve_model_dir() -> Result<PathBuf, EmbedError> {
    // 1. Explicit env var — used as-is, no revision check
    if let Ok(p) = std::env::var("BERT_MODEL_DIR") {
        let path = PathBuf::from(&p);
        if path.join("model.safetensors").exists() {
            return Ok(path);
        }
        return Err(EmbedError::ModelError(format!(
            "BERT_MODEL_DIR='{}' does not contain model.safetensors", p
        )));
    }

    // 2. HF hub cache at the pinned revision
    let revision = std::env::var("BERT_MODEL_REVISION")
        .unwrap_or_else(|_| PINNED_REVISION.to_string());

    let snap = hf_hub_root()
        .join("models--sentence-transformers--all-MiniLM-L6-v2")
        .join("snapshots")
        .join(&revision);

    let required = ["config.json", "tokenizer.json", "model.safetensors"];
    let missing: Vec<&str> = required
        .iter()
        .filter(|f| !snap.join(f).exists())
        .copied()
        .collect();

    if missing.is_empty() {
        return Ok(snap);
    }

    Err(EmbedError::ModelError(format!(
        "Pinned revision '{}' not found or incomplete at '{}'. Missing: {:?}. \
         Set BERT_MODEL_DIR to an explicit path, or BERT_MODEL_REVISION to the \
         correct commit hash for your local HF cache.",
        revision,
        snap.display(),
        missing
    )))
}

fn hf_hub_root() -> PathBuf {
    if let Ok(hf_home) = std::env::var("HF_HOME") {
        return PathBuf::from(hf_home).join("hub");
    }
    let base = std::env::var("USERPROFILE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| dirs_next::cache_dir().unwrap_or_else(|| PathBuf::from(".")));
    base.join(".cache").join("huggingface").join("hub")
}

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

/// Embed text chunks using all-MiniLM-L6-v2 via candle BertModel (CPU).
///
/// Processes chunks in batches of BATCH_SIZE (mirrors Python embedder batch_size=64)
/// to avoid OOM on large corpora and keep memory usage bounded.
///
/// Returns one 384-dimensional `[f32; 384]` per input chunk.
const BATCH_SIZE: usize = 64;

pub fn embed_chunks(chunks: &[String]) -> Result<Vec<[f32; 384]>, EmbedError> {
    if chunks.is_empty() {
        return Ok(Vec::new());
    }

    let mut result: Vec<[f32; 384]> = Vec::with_capacity(chunks.len());

    for batch_start in (0..chunks.len()).step_by(BATCH_SIZE) {
        let batch = &chunks[batch_start..(batch_start + BATCH_SIZE).min(chunks.len())];
        let batch_result = embed_batch(batch)?;
        result.extend_from_slice(&batch_result);
    }

    Ok(result)
}

fn embed_batch(chunks: &[String]) -> Result<Vec<[f32; 384]>, EmbedError> {
    let embedder = get_embedder()?;
    let device = &embedder.device;

    // Tokenize the batch
    let encodings = embedder
        .tokenizer
        .encode_batch(chunks.to_vec(), true)
        .map_err(|e| EmbedError::TokenizerError(format!("Tokenization failed: {e}")))?;

    // Build token_ids and attention_mask tensors
    let token_ids: Vec<Tensor> = encodings
        .iter()
        .map(|enc| {
            let ids: Vec<u32> = enc.get_ids().to_vec();
            Tensor::new(ids.as_slice(), device)
                .map_err(|e| EmbedError::InferenceError(format!("Tensor creation: {e}")))
        })
        .collect::<Result<_, _>>()?;

    let attention_mask: Vec<Tensor> = encodings
        .iter()
        .map(|enc| {
            let mask: Vec<u32> = enc.get_attention_mask().to_vec();
            Tensor::new(mask.as_slice(), device)
                .map_err(|e| EmbedError::InferenceError(format!("Tensor creation: {e}")))
        })
        .collect::<Result<_, _>>()?;

    let token_ids = Tensor::stack(&token_ids, 0)
        .map_err(|e| EmbedError::InferenceError(format!("Stack token_ids: {e}")))?;
    let attention_mask = Tensor::stack(&attention_mask, 0)
        .map_err(|e| EmbedError::InferenceError(format!("Stack attention_mask: {e}")))?;
    let token_type_ids = token_ids
        .zeros_like()
        .map_err(|e| EmbedError::InferenceError(format!("zeros_like: {e}")))?;

    // Forward pass
    let embeddings = embedder
        .model
        .forward(&token_ids, &token_type_ids, Some(&attention_mask))
        .map_err(|e| EmbedError::InferenceError(format!("BERT forward: {e}")))?;

    // Mean pooling with attention mask
    let attention_mask_f = attention_mask
        .to_dtype(DType::F32)
        .map_err(|e| EmbedError::InferenceError(format!("dtype cast: {e}")))?
        .unsqueeze(2)
        .map_err(|e| EmbedError::InferenceError(format!("unsqueeze: {e}")))?;

    let sum_mask = attention_mask_f
        .sum(1)
        .map_err(|e| EmbedError::InferenceError(format!("sum_mask: {e}")))?;

    let pooled = embeddings
        .broadcast_mul(&attention_mask_f)
        .map_err(|e| EmbedError::InferenceError(format!("broadcast_mul: {e}")))?
        .sum(1)
        .map_err(|e| EmbedError::InferenceError(format!("sum: {e}")))?
        .broadcast_div(&sum_mask)
        .map_err(|e| EmbedError::InferenceError(format!("broadcast_div: {e}")))?;

    // Extract rows into [f32; 384] arrays and L2-normalise (mirrors Python embedder)
    let n = chunks.len();
    let mut result: Vec<[f32; 384]> = Vec::with_capacity(n);
    for i in 0..n {
        let row = pooled
            .get(i)
            .map_err(|e| EmbedError::InferenceError(format!("get row {i}: {e}")))?
            .to_vec1::<f32>()
            .map_err(|e| EmbedError::InferenceError(format!("to_vec1 row {i}: {e}")))?;

        if row.len() != 384 {
            return Err(EmbedError::InferenceError(format!(
                "Expected 384 dims, got {} for chunk {i}",
                row.len()
            )));
        }

        // L2 normalisation — identical to Python: v / max(||v||, 1e-9)
        let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        let scale = if norm > 1e-9 { 1.0 / norm } else { 1.0 };

        let mut arr = [0f32; 384];
        for (j, &v) in row.iter().enumerate() {
            arr[j] = v * scale;
        }
        result.push(arr);
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    fn stub_embed(chunks: &[String]) -> Vec<[f32; 384]> {
        chunks.iter().map(|_| [0f32; 384]).collect()
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_every_embedding_has_384_dimensions(
            chunks in prop::collection::vec("[a-zA-Z0-9 .,!?]{1,200}", 1..=20)
        ) {
            let string_chunks: Vec<String> = chunks.iter().map(|s: &String| s.clone()).collect();
            let embeddings = stub_embed(&string_chunks);
            prop_assert_eq!(embeddings.len(), string_chunks.len());
            for vec in &embeddings {
                prop_assert_eq!(vec.len(), 384);
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_ort_every_embedding_has_384_dimensions(
            chunks in prop::collection::vec("[a-zA-Z0-9 .,!?]{1,200}", 1..=20)
        ) {
            let string_chunks: Vec<String> = chunks.iter().map(|s: &String| s.clone()).collect();
            let embeddings = stub_embed(&string_chunks);
            prop_assert_eq!(embeddings.len(), string_chunks.len());
            for vec in &embeddings {
                prop_assert_eq!(vec.len(), 384);
            }
        }
    }
}
