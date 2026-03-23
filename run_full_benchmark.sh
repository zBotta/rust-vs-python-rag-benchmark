#!/usr/bin/env bash
set -euo pipefail
#
# run_full_benchmark.sh — Run all LLM backend combinations + stress test.
#
# Runs the benchmark for each enabled backend in sequence:
#   1. ollama_http  (standard + stress)
#   2. llama_cpp    (standard + stress)  — skipped if gguf_model_path is unset
#   3. llm_rs       (standard + stress)  — skipped if gguf_model_path is unset
#
# Each backend writes to its own output files so nothing is overwritten.
#
# Usage:
#   bash run_full_benchmark.sh
#
# Options (env vars):
#   SKIP_OLLAMA=1        Skip the ollama_http backend
#   SKIP_LLAMA_CPP=1     Skip the llama_cpp backend
#   SKIP_LLM_RS=1        Skip the llm_rs backend
#   STRESS=1             Enable stress test for all backends
#   DISABLE_SSL_VERIFY=1 Bypass SSL verification (corporate proxy)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
RUST_BIN="./target/release/rust_pipeline"
PARQUET_FILE="./data/20231101.simple/train-00000-of-00001.parquet"

# Read base config values
OUTPUT_DIR=$(grep -E '^output_dir\s*=' benchmark_config.toml \
    | head -1 | sed 's/.*=\s*"\(.*\)".*/\1/')
OUTPUT_DIR="${OUTPUT_DIR:-output/}"

GGUF_PATH=$(grep -E '^gguf_model_path\s*=' benchmark_config.toml \
    | head -1 | sed 's/.*=\s*"\(.*\)".*/\1/')

STRESS_ENABLED="${STRESS:-0}"

# Ensure output dir exists
mkdir -p "${OUTPUT_DIR}"

# Build Rust binary once up front
if [ ! -f "$RUST_BIN" ]; then
    echo "==> Building Rust binary (release)..."
    cargo build --release --manifest-path rust_pipeline/Cargo.toml
fi

# Ensure parquet dataset is present for Rust pipeline
if [ ! -f "$PARQUET_FILE" ]; then
    echo "==> Parquet dataset not found. Downloading..."
    uv run python download_parquet.py
fi

# ---------------------------------------------------------------------------
# run_one_backend BACKEND
#   Patches benchmark_config.toml in-place, runs both pipelines, generates report.
# ---------------------------------------------------------------------------
run_one_backend() {
    local BACKEND="$1"
    echo ""
    echo "========================================================"
    echo "  Backend: ${BACKEND}"
    echo "========================================================"

    # Patch llm_backend in config (sed in-place, portable)
    sed -i.bak "s|^llm_backend\s*=.*|llm_backend     = \"${BACKEND}\"|" benchmark_config.toml

    # Enable or disable stress test
    if [ "${STRESS_ENABLED}" = "1" ]; then
        sed -i.bak "s|^enabled\s*=.*|enabled           = true|" benchmark_config.toml
    else
        sed -i.bak "s|^enabled\s*=.*|enabled           = false|" benchmark_config.toml
    fi

    # Run Python pipeline (exits cleanly with skip message for llm_rs)
    echo "==> [${BACKEND}] Running Python pipeline..."
    uv run python -m python_pipeline.pipeline || true

    echo "==> [${BACKEND}] Waiting 10 seconds (cool-down)..."
    sleep 10

    # Run Rust pipeline
    echo "==> [${BACKEND}] Running Rust pipeline..."
    "${RUST_BIN}"

    # Generate report
    echo "==> [${BACKEND}] Generating report..."
    uv run python report/generate_report.py \
        --python-jsonl "${OUTPUT_DIR}metrics_python_${BACKEND}.jsonl" \
        --rust-jsonl   "${OUTPUT_DIR}metrics_rust_${BACKEND}.jsonl" \
        --output       "${OUTPUT_DIR}benchmark_report_${BACKEND}.md" \
        --llm-backend  "${BACKEND}"

    echo "==> [${BACKEND}] Report: ${OUTPUT_DIR}benchmark_report_${BACKEND}.md"
}

# ---------------------------------------------------------------------------
# Backend: ollama_http
# ---------------------------------------------------------------------------
if [ "${SKIP_OLLAMA:-0}" != "1" ]; then
    run_one_backend "ollama_http"
else
    echo "==> Skipping ollama_http (SKIP_OLLAMA=1)"
fi

# ---------------------------------------------------------------------------
# Backend: llama_cpp  (requires gguf_model_path)
# ---------------------------------------------------------------------------
if [ "${SKIP_LLAMA_CPP:-0}" != "1" ]; then
    if [ -z "${GGUF_PATH}" ]; then
        echo ""
        echo "==> Skipping llama_cpp: gguf_model_path is not set in benchmark_config.toml"
        echo "    Set gguf_model_path = \"/path/to/model.gguf\" to enable this backend."
    else
        run_one_backend "llama_cpp"
    fi
else
    echo "==> Skipping llama_cpp (SKIP_LLAMA_CPP=1)"
fi

# ---------------------------------------------------------------------------
# Backend: llm_rs  (requires gguf_model_path; Python pipeline is skipped)
# ---------------------------------------------------------------------------
if [ "${SKIP_LLM_RS:-0}" != "1" ]; then
    if [ -z "${GGUF_PATH}" ]; then
        echo ""
        echo "==> Skipping llm_rs: gguf_model_path is not set in benchmark_config.toml"
        echo "    Set gguf_model_path = \"/path/to/model.gguf\" to enable this backend."
    else
        run_one_backend "llm_rs"
    fi
else
    echo "==> Skipping llm_rs (SKIP_LLM_RS=1)"
fi

# ---------------------------------------------------------------------------
# Restore original llm_backend value in config
# ---------------------------------------------------------------------------
sed -i.bak "s|^llm_backend\s*=.*|llm_backend     = \"ollama_http\"|" benchmark_config.toml
sed -i.bak "s|^enabled\s*=.*|enabled           = false|" benchmark_config.toml
rm -f benchmark_config.toml.bak

# Generate consolidated cross-scenario report from all JSONL outputs
echo "==> Generating cross-scenario report..."
uv run python report/generate_all_scenarios_report.py \
    --output-dir "${OUTPUT_DIR}" \
    --output "${OUTPUT_DIR}benchmark_report_all_scenarios.md"

echo ""
echo "========================================================"
echo "  Full benchmark complete."
echo "  Reports written to: ${OUTPUT_DIR}"
ls -1 "${OUTPUT_DIR}"benchmark_report_*.md 2>/dev/null || true
echo "========================================================"
