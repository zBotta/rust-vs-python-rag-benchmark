# run_full_benchmark.ps1 — Run all LLM backend combinations + stress test.
#
# Runs the benchmark for each enabled backend in sequence:
#   1. ollama_http  (standard + stress)
#   2. llama_cpp    (standard + stress)  — skipped if gguf_model_path is unset
#   3. llm_rs       (standard + stress)  — skipped if gguf_model_path is unset
#
# Each backend writes to its own output files so nothing is overwritten.
#
# Usage:
#   .\run_full_benchmark.ps1
#
# Options (env vars):
#   $env:SKIP_OLLAMA    = "1"   Skip the ollama_http backend
#   $env:SKIP_LLAMA_CPP = "1"   Skip the llama_cpp backend
#   $env:SKIP_LLM_RS    = "1"   Skip the llm_rs backend
#   $env:STRESS         = "1"   Enable stress test for all backends
#   $env:DISABLE_SSL_VERIFY = "1"  Bypass SSL verification

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# ---------------------------------------------------------------------------
# Read base config values
# ---------------------------------------------------------------------------
$configContent = Get-Content "benchmark_config.toml" -Raw

$outputDirMatch = [regex]::Match($configContent, 'output_dir\s*=\s*"([^"]+)"')
$OutputDir = if ($outputDirMatch.Success) { $outputDirMatch.Groups[1].Value.TrimEnd('/') } else { "output" }

$ggufMatch = [regex]::Match($configContent, 'gguf_model_path\s*=\s*"([^"]*)"')
$GgufPath = if ($ggufMatch.Success) { $ggufMatch.Groups[1].Value } else { "" }

$StressEnabled = $env:STRESS -eq "1"

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

# ---------------------------------------------------------------------------
# Build Rust binary once
# ---------------------------------------------------------------------------
$rustBinary = ".\target\release\rust_pipeline.exe"
if (-not (Test-Path $rustBinary)) {
    Write-Host "==> Building Rust binary (release)..." -ForegroundColor Cyan
    cargo build --release --manifest-path rust_pipeline\Cargo.toml
}

# Ensure parquet dataset is present
$parquetFile = ".\data\20231101.simple\train-00000-of-00001.parquet"
if (-not (Test-Path $parquetFile)) {
    Write-Host "==> Parquet dataset not found. Downloading..." -ForegroundColor Yellow
    uv run python download_parquet.py
}

# ---------------------------------------------------------------------------
# Helper: patch a key in benchmark_config.toml
# ---------------------------------------------------------------------------
function Set-ConfigValue([string]$Key, [string]$Value) {
    $content = Get-Content "benchmark_config.toml" -Raw
    # Preserve original spacing: capture the whitespace between key and '='
    $content = [regex]::Replace($content, "(?m)^($Key)(\s*=\s*).*", "`${1}`${2}`"$Value`"")
    Set-Content "benchmark_config.toml" $content -NoNewline
}

function Set-StressEnabled([bool]$Enabled) {
    $val = if ($Enabled) { "true" } else { "false" }
    $content = Get-Content "benchmark_config.toml" -Raw
    $content = [regex]::Replace($content, "(?m)^(enabled)(\s*=\s*).*", "`${1}`${2}$val")
    Set-Content "benchmark_config.toml" $content -NoNewline
}

# ---------------------------------------------------------------------------
# Helper: run one backend end-to-end
# ---------------------------------------------------------------------------
function Invoke-Backend([string]$Backend) {
    Write-Host "" 
    Write-Host "========================================================" -ForegroundColor Magenta
    Write-Host "  Backend: $Backend" -ForegroundColor Magenta
    Write-Host "========================================================" -ForegroundColor Magenta

    Set-ConfigValue "llm_backend" $Backend
    Set-StressEnabled $StressEnabled

    # Python pipeline (exits cleanly with skip message for llm_rs)
    Write-Host "==> [$Backend] Running Python pipeline..." -ForegroundColor Cyan
    try { uv run python -m python_pipeline.pipeline } catch { Write-Host "  Python pipeline exited (may be expected for llm_rs)" -ForegroundColor DarkGray }

    Write-Host "==> [$Backend] Waiting 10 seconds (cool-down)..." -ForegroundColor Yellow
    Start-Sleep -Seconds 10

    # Rust pipeline
    Write-Host "==> [$Backend] Running Rust pipeline..." -ForegroundColor Cyan
    & $rustBinary

    # Report
    Write-Host "==> [$Backend] Generating report..." -ForegroundColor Cyan
    uv run python report\generate_report.py `
        --python-jsonl "$OutputDir\metrics_python_$Backend.jsonl" `
        --rust-jsonl   "$OutputDir\metrics_rust_$Backend.jsonl" `
        --output       "$OutputDir\benchmark_report_$Backend.md" `
        --llm-backend  "$Backend"

    Write-Host "==> [$Backend] Report: $OutputDir\benchmark_report_$Backend.md" -ForegroundColor Green
}

# ---------------------------------------------------------------------------
# Backend: ollama_http
# ---------------------------------------------------------------------------
if ($env:SKIP_OLLAMA -ne "1") {
    Invoke-Backend "ollama_http"
} else {
    Write-Host "==> Skipping ollama_http (SKIP_OLLAMA=1)" -ForegroundColor DarkGray
}

# ---------------------------------------------------------------------------
# Backend: llama_cpp
# ---------------------------------------------------------------------------
if ($env:SKIP_LLAMA_CPP -ne "1") {
    if ([string]::IsNullOrWhiteSpace($GgufPath)) {
        Write-Host ""
        Write-Host "==> Skipping llama_cpp: gguf_model_path is not set in benchmark_config.toml" -ForegroundColor Yellow
        Write-Host "    Set gguf_model_path = `"C:\path\to\model.gguf`" to enable this backend." -ForegroundColor Yellow
    } else {
        Invoke-Backend "llama_cpp"
    }
} else {
    Write-Host "==> Skipping llama_cpp (SKIP_LLAMA_CPP=1)" -ForegroundColor DarkGray
}

# ---------------------------------------------------------------------------
# Backend: llm_rs
# ---------------------------------------------------------------------------
if ($env:SKIP_LLM_RS -ne "1") {
    if ([string]::IsNullOrWhiteSpace($GgufPath)) {
        Write-Host ""
        Write-Host "==> Skipping llm_rs: gguf_model_path is not set in benchmark_config.toml" -ForegroundColor Yellow
        Write-Host "    Set gguf_model_path = `"C:\path\to\model.gguf`" to enable this backend." -ForegroundColor Yellow
    } else {
        Invoke-Backend "llm_rs"
    }
} else {
    Write-Host "==> Skipping llm_rs (SKIP_LLM_RS=1)" -ForegroundColor DarkGray
}

# ---------------------------------------------------------------------------
# Restore config to defaults
# ---------------------------------------------------------------------------
Set-ConfigValue "llm_backend" "ollama_http"
Set-StressEnabled $false
Set-StressEnabled $false

# Generate consolidated cross-scenario report from all JSONL outputs
Write-Host "==> Generating cross-scenario report..." -ForegroundColor Cyan
uv run python report\generate_all_scenarios_report.py `
    --output-dir "$OutputDir" `
    --output "$OutputDir\benchmark_report_all_scenarios.md"

Write-Host ""
Write-Host "========================================================" -ForegroundColor Magenta
Write-Host "  Full benchmark complete." -ForegroundColor Green
Write-Host "  Reports written to: $OutputDir\" -ForegroundColor Green
Get-ChildItem "$OutputDir\benchmark_report_*.md" -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "  $_" }
Write-Host "========================================================" -ForegroundColor Magenta
