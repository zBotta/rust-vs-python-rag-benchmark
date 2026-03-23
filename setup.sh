#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# 1. Check for uv
# ---------------------------------------------------------------------------
echo "==> Checking for uv..."
if ! command -v uv &>/dev/null; then
    echo "ERROR: uv not found on PATH."
    echo "Install it from https://docs.astral.sh/uv/getting-started/installation/"
    echo "  e.g.  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
echo "  uv found: $(uv --version)"

# ---------------------------------------------------------------------------
# 2. Initialise project and install Python dependencies
# ---------------------------------------------------------------------------
echo "==> Initialising uv project..."
uv init --python ">=3.11" --no-workspace 2>/dev/null || true   # idempotent

echo "==> Installing Python dependencies..."
uv add langchain-text-splitters datasets hnswlib httpx matplotlib numpy pytest hypothesis respx tomli \
    onnxruntime tokenizers psutil llama-cpp-python

echo "==> Syncing environment..."
uv sync

# ---------------------------------------------------------------------------
# 3. Build Rust binary (release)
# ---------------------------------------------------------------------------
echo "==> Building Rust binary (release)..."
if ! command -v cargo &>/dev/null; then
    echo "ERROR: cargo not found."
    echo "Install Rust stable >= 1.78 from https://rustup.rs/ and restart your shell."
    exit 1
fi

cargo build --release --manifest-path rust_pipeline/Cargo.toml

echo ""
echo "Setup complete."
echo "Run the benchmark with: ./run_benchmark.sh"
