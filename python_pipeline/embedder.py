"""Embedder — raw transformers + manual mean pooling.

Mirrors the Rust candle::BertModel pipeline exactly:
  - Same model checkpoint (sentence-transformers/all-MiniLM-L6-v2)
  - Same pinned revision as Rust: PINNED_REVISION
  - Same local HF cache directory as Rust
  - AutoTokenizer  →  AutoModel  →  attention-mask-aware mean pooling  →  L2 normalisation
  - No SentenceTransformer abstraction

Pooling and normalisation are made fully explicit so the Python and Rust
paths are conceptually identical and any numerical difference is attributable
solely to framework arithmetic, not to hidden library behaviour.

Artifact bundle pinning
-----------------------
Both pipelines read the *same* local directory:

    <hf_hub_root>/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/<REVISION>/

where REVISION is the hard-pinned commit hash below (overridable via
BERT_MODEL_REVISION env var).  This guarantees that config.json,
tokenizer.json, tokenizer_config.json, special_tokens_map.json, and
model.safetensors are byte-for-byte identical between the two pipelines.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from python_pipeline.config import BenchmarkError


class EmbedError(BenchmarkError):
    """Raised when the embedding model cannot be loaded or encoding fails."""


# ---------------------------------------------------------------------------
# Pinned revision — must match Rust embedder.rs PINNED_REVISION constant.
# Override with BERT_MODEL_REVISION env var if you have a different local cache.
# ---------------------------------------------------------------------------
_DEFAULT_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"


# ---------------------------------------------------------------------------
# Model resolution — same logic as Rust embedder.rs
# ---------------------------------------------------------------------------

def _resolve_model_dir() -> Path:
    """Return the local directory containing the pinned model weights.

    Resolution order (mirrors Rust embedder.rs):
      1. BERT_MODEL_DIR env var — explicit path, used as-is (no revision check).
      2. HF hub cache at the pinned revision:
         <hub_root>/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/<REVISION>/
         where REVISION = BERT_MODEL_REVISION env var or the hard-coded default.
    """
    if env := os.environ.get("BERT_MODEL_DIR"):
        p = Path(env)
        if (p / "model.safetensors").exists():
            return p
        raise EmbedError(f"BERT_MODEL_DIR='{env}' does not contain model.safetensors")

    revision = os.environ.get("BERT_MODEL_REVISION", _DEFAULT_REVISION)

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        hub_root = Path(hf_home) / "hub"
    else:
        hub_root = Path(os.environ.get("USERPROFILE", Path.home())) / ".cache" / "huggingface" / "hub"

    snap = (
        hub_root
        / "models--sentence-transformers--all-MiniLM-L6-v2"
        / "snapshots"
        / revision
    )
    required = ["config.json", "tokenizer.json", "model.safetensors"]
    missing = [f for f in required if not (snap / f).exists()]
    if not missing:
        return snap

    raise EmbedError(
        f"Pinned revision '{revision}' not found or incomplete at '{snap}'. "
        f"Missing: {missing}. "
        "Set BERT_MODEL_DIR to an explicit path, or BERT_MODEL_REVISION to the "
        "correct commit hash for your local HF cache."
    )


# ---------------------------------------------------------------------------
# Lazy model cache
# ---------------------------------------------------------------------------

_tokenizer = None
_model = None


def _get_model():
    global _tokenizer, _model
    if _model is None:
        from transformers import AutoModel, AutoTokenizer  # type: ignore

        model_dir = str(_resolve_model_dir())
        try:
            _tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
            _model = AutoModel.from_pretrained(model_dir, local_files_only=True)
            _model.eval()
        except Exception as exc:
            raise EmbedError(f"Failed to load embedding model from '{model_dir}': {exc}") from exc
    return _tokenizer, _model


# ---------------------------------------------------------------------------
# Pooling helpers (explicit, mirrors Rust candle code)
# ---------------------------------------------------------------------------

def _mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Attention-mask-aware mean pooling over the sequence dimension.

    Mirrors the Rust candle embedder:
        pooled = (embeddings * mask_expanded).sum(1) / mask_expanded.sum(1)
    """
    mask_expanded = attention_mask.unsqueeze(-1).float()          # [B, T, 1]
    sum_embeddings = (token_embeddings * mask_expanded).sum(1)    # [B, H]
    sum_mask = mask_expanded.sum(1).clamp(min=1e-9)               # [B, 1]
    return sum_embeddings / sum_mask                               # [B, H]


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

_BATCH_SIZE = 64


def embed_chunks(chunks: list[str]) -> list[list[float]]:
    """Embed text chunks using all-MiniLM-L6-v2 via raw transformers.

    Pipeline (identical conceptual steps to Rust candle embedder):
      1. Tokenize with AutoTokenizer (padding=True, truncation=True, max_length=512)
      2. Forward pass through AutoModel (no grad)
      3. Attention-mask-aware mean pooling over token dimension
      4. L2 normalisation

    Args:
        chunks: List of text strings to embed.

    Returns:
        List of 384-dimensional float vectors (one per chunk).

    Raises:
        EmbedError: If the model cannot be loaded or inference fails.
    """
    if not chunks:
        return []

    tokenizer, model = _get_model()
    results: list[list[float]] = []

    for i in range(0, len(chunks), _BATCH_SIZE):
        batch = chunks[i : i + _BATCH_SIZE]
        try:
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            with torch.no_grad():
                outputs = model(**encoded)

            # last_hidden_state: [B, T, 384]
            pooled = _mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
            pooled_np = pooled.cpu().numpy()
            pooled_np = _l2_normalize(pooled_np)

            results.extend(row.tolist() for row in pooled_np)
        except Exception as exc:
            raise EmbedError(f"Embedding inference failed on batch {i // _BATCH_SIZE}: {exc}") from exc

    return results
