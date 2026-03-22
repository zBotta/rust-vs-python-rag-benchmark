#!/usr/bin/env bash
set -euo pipefail
# Set DISABLE_SSL_VERIFY=1 if behind a corporate proxy with SSL inspection:
#   DISABLE_SSL_VERIFY=1 ./run_benchmark.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 0. Start embedding server (used by Rust pipeline)
echo "==> Starting embedding server..."
uv run python embedding_server.py &
EMBED_PID=$!
trap "echo '==> Stopping embedding server...'; kill $EMBED_PID 2>/dev/null" EXIT

# Wait until server is ready
for i in $(seq 1 30); do
    sleep 2
    if curl -sf http://127.0.0.1:8765/health > /dev/null 2>&1; then
        echo "Embedding server ready."
        break
    fi
    if [ $i -eq 30 ]; then
        echo "ERROR: Embedding server did not start in time." >&2
        exit 1
    fi
done

# 1. Run Python pipeline
echo "==> Running Python pipeline..."
uv run python -m python_pipeline.pipeline

# 2. Wait 10 seconds cool-down
echo "Waiting 10 seconds for LLM server to drain..."
sleep 10

# 3. Ensure parquet dataset is available for the Rust pipeline
PARQUET_FILE="./data/20231101.simple/train-00000-of-00001.parquet"
if [ ! -f "$PARQUET_FILE" ]; then
    echo "==> Parquet dataset not found. Downloading..."
    uv run python download_parquet.py
else
    echo "==> Parquet dataset already present, skipping download."
fi

# 4. Run Rust pipeline
echo "==> Running Rust pipeline..."
RUST_BIN="./target/release/rust_pipeline"
if [ ! -f "$RUST_BIN" ]; then
    echo "Rust binary not found. Building now..."
    cargo build --release --manifest-path rust_pipeline/Cargo.toml
fi
"$RUST_BIN"

# 5. Run report generator
echo "==> Generating report..."
uv run python report/generate_report.py

echo ""
echo "Benchmark complete. See benchmark_report.md for results."
