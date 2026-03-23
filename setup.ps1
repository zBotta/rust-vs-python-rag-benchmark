# setup.ps1 — Bootstrap the benchmark environment on Windows using uv.
# Usage: .\setup.ps1
# Run from the rust-vs-python-rag-benchmark\ directory.

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# ---------------------------------------------------------------------------
# 1. Check for uv
# ---------------------------------------------------------------------------
Write-Host "==> Checking for uv..." -ForegroundColor Cyan

if (-not (Get-Command "uv" -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: uv not found on PATH." -ForegroundColor Red
    Write-Host "Install it from https://docs.astral.sh/uv/getting-started/installation/" -ForegroundColor Red
    Write-Host "  e.g.  winget install --id=astral-sh.uv  -e" -ForegroundColor Yellow
    exit 1
}

Write-Host "  uv found: $(uv --version)" -ForegroundColor Green

# ---------------------------------------------------------------------------
# 2. Initialise project and install Python dependencies
# ---------------------------------------------------------------------------
Write-Host "==> Initialising uv project..." -ForegroundColor Cyan
$ErrorActionPreference = "Continue"
uv init --python ">=3.11" --no-workspace 2>&1 | Out-Null
$ErrorActionPreference = "Stop"

Write-Host "==> Installing Python dependencies..." -ForegroundColor Cyan
uv add langchain-text-splitters datasets hnswlib httpx matplotlib numpy pytest hypothesis respx tomli `
    onnxruntime tokenizers psutil llama-cpp-python pyarrow sentence-transformers waitress

Write-Host "==> Syncing environment..." -ForegroundColor Cyan
uv sync

# ---------------------------------------------------------------------------
# 3. Build Rust binary (release)
# ---------------------------------------------------------------------------
Write-Host "==> Building Rust binary (release)..." -ForegroundColor Cyan

if (-not (Get-Command "cargo" -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: cargo not found." -ForegroundColor Red
    Write-Host "Install Rust stable >= 1.78 from https://rustup.rs/ and restart your shell." -ForegroundColor Red
    exit 1
}

cargo build --release --manifest-path rust_pipeline\Cargo.toml

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Run the benchmark : .\run_benchmark.ps1"
