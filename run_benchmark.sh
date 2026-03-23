#!/usr/bin/env bash
set -euo pipefail
# Set DISABLE_SSL_VERIFY=1 if behind a corporate proxy with SSL inspection:
#   DISABLE_SSL_VERIFY=1 ./run_benchmark.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# 0. Read llm_backend from benchmark_config.toml
# ---------------------------------------------------------------------------
LLM_BACKEND=$(grep -E '^llm_backend\s*=' benchmark_config.toml \
    | head -1 | sed 's/.*=\s*"\(.*\)".*/\1/')
echo "==> LLM backend: ${LLM_BACKEND}"

OUTPUT_DIR=$(grep -E '^output_dir\s*=' benchmark_config.toml \
    | head -1 | sed 's/.*=\s*"\(.*\)".*/\1/')
OUTPUT_DIR="${OUTPUT_DIR:-output/}"

# ---------------------------------------------------------------------------
# 1. Run Python pipeline (skipped automatically when llm_backend = "llm_rs")
# ---------------------------------------------------------------------------
echo "==> Running Python pipeline..."
uv run python -m python_pipeline.pipeline

# ---------------------------------------------------------------------------
# 2. Cool-down between pipeline runs (Requirement 7.4)
# ---------------------------------------------------------------------------
echo "Waiting 10 seconds for LLM server to drain..."
sleep 10

# ---------------------------------------------------------------------------
# 3. Ensure parquet dataset is available for the Rust pipeline
# ---------------------------------------------------------------------------
PARQUET_FILE="./data/20231101.simple/train-00000-of-00001.parquet"
if [ ! -f "$PARQUET_FILE" ]; then
    echo "==> Parquet dataset not found. Downloading..."
    uv run python download_parquet.py
else
    echo "==> Parquet dataset already present, skipping download."
fi

# ---------------------------------------------------------------------------
# 4. Run Rust pipeline
# ---------------------------------------------------------------------------
echo "==> Running Rust pipeline..."
RUST_BIN="./target/release/rust_pipeline"
if [ ! -f "$RUST_BIN" ]; then
    echo "Rust binary not found. Building now..."
    cargo build --release --manifest-path rust_pipeline/Cargo.toml
fi
"$RUST_BIN"

# ---------------------------------------------------------------------------
# 5. Run report generator
# ---------------------------------------------------------------------------
echo "==> Generating report..."
uv run python report/generate_report.py \
    --python-jsonl "${OUTPUT_DIR}metrics_python_${LLM_BACKEND}.jsonl" \
    --rust-jsonl   "${OUTPUT_DIR}metrics_rust_${LLM_BACKEND}.jsonl" \
    --output       "${OUTPUT_DIR}benchmark_report_${LLM_BACKEND}.md" \
    --llm-backend  "${LLM_BACKEND}"

echo ""
echo "Benchmark complete. See ${OUTPUT_DIR}benchmark_report_${LLM_BACKEND}.md for results."
