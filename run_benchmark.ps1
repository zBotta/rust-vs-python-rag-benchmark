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
# 0. Start embedding server (used by Rust pipeline)
# ---------------------------------------------------------------------------
Write-Host "==> Starting embedding server..." -ForegroundColor Cyan
$embedJob = Start-Job -ScriptBlock {
    param($dir)
    Set-Location $dir
    $env:DISABLE_SSL_VERIFY = $using:env:DISABLE_SSL_VERIFY
    uv run python embedding_server.py 2>&1
} -ArgumentList $ScriptDir

# Wait until the server is ready
$ready = $false
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 2
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:8765/health" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
        if ($r.StatusCode -eq 200) { $ready = $true; break }
    } catch {}
}
if (-not $ready) {
    Write-Host "ERROR: Embedding server did not start in time." -ForegroundColor Red
    Stop-Job $embedJob; Remove-Job $embedJob
    exit 1
}
Write-Host "Embedding server ready." -ForegroundColor Green

# ---------------------------------------------------------------------------
# 1. Run Python pipeline
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
uv run python report\generate_report.py

Write-Host ""
Write-Host "Benchmark complete. See benchmark_report.md for results." -ForegroundColor Green

# ---------------------------------------------------------------------------
# Cleanup: stop embedding server
# ---------------------------------------------------------------------------
Write-Host "==> Stopping embedding server..." -ForegroundColor DarkGray
Stop-Job $embedJob -ErrorAction SilentlyContinue
Remove-Job $embedJob -ErrorAction SilentlyContinue
