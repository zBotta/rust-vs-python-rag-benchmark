# run_benchmark.ps1 — Orchestrate the full benchmark on Windows.
#
# Usage: .\run_benchmark.ps1
# Assumes setup.ps1 has already been run.
#
# Set DISABLE_SSL_VERIFY=1 if behind a corporate proxy with SSL inspection:
#   $env:DISABLE_SSL_VERIFY=1; .\run_benchmark.ps1

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# ---------------------------------------------------------------------------
# 0. Read llm_backend and output_dir from benchmark_config.toml
# ---------------------------------------------------------------------------
$configContent = Get-Content "benchmark_config.toml" -Raw
$llmBackendMatch = [regex]::Match($configContent, 'llm_backend\s*=\s*"([^"]+)"')
$LlmBackend = if ($llmBackendMatch.Success) { $llmBackendMatch.Groups[1].Value } else { "ollama_http" }

$outputDirMatch = [regex]::Match($configContent, 'output_dir\s*=\s*"([^"]+)"')
$OutputDir = if ($outputDirMatch.Success) { $outputDirMatch.Groups[1].Value } else { "output/" }
$OutputDir = $OutputDir.TrimEnd('/')

Write-Host "==> LLM backend: $LlmBackend" -ForegroundColor Cyan

# ---------------------------------------------------------------------------
# 1. Run Python pipeline (skipped automatically when llm_backend = "llm_rs")
# ---------------------------------------------------------------------------
Write-Host "==> Running Python pipeline..." -ForegroundColor Cyan
uv run python -m python_pipeline.pipeline

# ---------------------------------------------------------------------------
# 2. Cool-down between pipeline runs (Requirement 7.4)
# ---------------------------------------------------------------------------
Write-Host "Waiting 10 seconds for LLM server to drain..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# ---------------------------------------------------------------------------
# 3. Ensure parquet dataset is available for the Rust pipeline
# ---------------------------------------------------------------------------
$parquetFile = ".\data\20231101.simple\train-00000-of-00001.parquet"
if (-not (Test-Path $parquetFile)) {
    Write-Host "==> Parquet dataset not found. Downloading..." -ForegroundColor Yellow
    uv run python download_parquet.py
} else {
    Write-Host "==> Parquet dataset already present, skipping download." -ForegroundColor DarkGray
}

# ---------------------------------------------------------------------------
# 4. Run Rust pipeline
# ---------------------------------------------------------------------------
Write-Host "==> Running Rust pipeline..." -ForegroundColor Cyan

$rustBinary = ".\target\release\rust_pipeline.exe"
if (-not (Test-Path $rustBinary)) {
    Write-Host "Rust binary not found. Building now..." -ForegroundColor Yellow
    cargo build --release --manifest-path rust_pipeline\Cargo.toml
}
& $rustBinary

# ---------------------------------------------------------------------------
# 5. Generate report
# ---------------------------------------------------------------------------
Write-Host "==> Generating report..." -ForegroundColor Cyan
uv run python report\generate_report.py `
    --python-jsonl "$OutputDir\metrics_python_$LlmBackend.jsonl" `
    --rust-jsonl   "$OutputDir\metrics_rust_$LlmBackend.jsonl" `
    --output       "$OutputDir\benchmark_report_$LlmBackend.md" `
    --llm-backend  "$LlmBackend"

Write-Host ""
Write-Host "Benchmark complete. See $OutputDir\benchmark_report_$LlmBackend.md for results." -ForegroundColor Green
