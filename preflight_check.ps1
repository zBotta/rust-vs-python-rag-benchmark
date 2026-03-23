# preflight_check.ps1 — Smoke-test all LLM backends before running the full benchmark.
#
# Runs each enabled backend with:
#   - num_documents = 10   (fast embedding)
#   - query_set_debug.json (5 queries)
#   - stress_test disabled
#
# A backend is considered healthy when:
#   - Preflight OK line is printed
#   - failures=0/5 in the final summary line
#
# Usage:
#   .\preflight_check.ps1
#
# Options (env vars):
#   $env:SKIP_OLLAMA    = "1"   Skip the ollama_http backend
#   $env:SKIP_LLAMA_CPP = "1"   Skip the llama_cpp backend
#   $env:SKIP_LLM_RS    = "1"   Skip the llm_rs backend
#   $env:DISABLE_SSL_VERIFY = "1"  Bypass SSL verification

$ErrorActionPreference = "Continue"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# ---------------------------------------------------------------------------
# Read base config values
# ---------------------------------------------------------------------------
$configContent = Get-Content "benchmark_config.toml" -Raw

$ggufMatch = [regex]::Match($configContent, 'gguf_model_path\s*=\s*"([^"]*)"')
$GgufPath = if ($ggufMatch.Success) { $ggufMatch.Groups[1].Value } else { "" }

$outputDirMatch = [regex]::Match($configContent, 'output_dir\s*=\s*"([^"]+)"')
$OutputDir = if ($outputDirMatch.Success) { $outputDirMatch.Groups[1].Value.TrimEnd('/') } else { "output" }

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

# ---------------------------------------------------------------------------
# Build Rust binary if needed
# ---------------------------------------------------------------------------
$rustBinary = ".\target\release\rust_pipeline.exe"
if (-not (Test-Path $rustBinary)) {
    Write-Host "==> Building Rust binary (release)..." -ForegroundColor Cyan
    cargo build --release --manifest-path rust_pipeline\Cargo.toml
}

# ---------------------------------------------------------------------------
# Helper: patch a key in benchmark_config.toml
# ---------------------------------------------------------------------------
function Set-ConfigValue([string]$Key, [string]$Value) {
    $content = Get-Content "benchmark_config.toml" -Raw
    $content = [regex]::Replace($content, "(?m)^($Key)(\s*=\s*).*", "`${1}`${2}`"$Value`"")
    Set-Content "benchmark_config.toml" $content -NoNewline
}

function Set-ConfigInt([string]$Key, [int]$Value) {
    $content = Get-Content "benchmark_config.toml" -Raw
    $content = [regex]::Replace($content, "(?m)^($Key)(\s*=\s*).*", "`${1}`${2}$Value")
    Set-Content "benchmark_config.toml" $content -NoNewline
}

function Set-StressEnabled([bool]$Enabled) {
    $val = if ($Enabled) { "true" } else { "false" }
    $content = Get-Content "benchmark_config.toml" -Raw
    $content = [regex]::Replace($content, "(?m)^(enabled)(\s*=\s*).*", "`${1}`${2}$val")
    Set-Content "benchmark_config.toml" $content -NoNewline
}

# ---------------------------------------------------------------------------
# Save original config values to restore at the end
# ---------------------------------------------------------------------------
$origNumDocsMatch  = [regex]::Match($configContent, '(?m)^num_documents\s*=\s*(\d+)')
$origNumDocs       = if ($origNumDocsMatch.Success) { $origNumDocsMatch.Groups[1].Value } else { "1000" }

$origQuerySetMatch = [regex]::Match($configContent, 'query_set_path\s*=\s*"([^"]+)"')
$origQuerySet      = if ($origQuerySetMatch.Success) { $origQuerySetMatch.Groups[1].Value } else { "query_set.json" }

$origBackendMatch  = [regex]::Match($configContent, 'llm_backend\s*=\s*"([^"]+)"')
$origBackend       = if ($origBackendMatch.Success) { $origBackendMatch.Groups[1].Value } else { "ollama_http" }

# ---------------------------------------------------------------------------
# Apply smoke-test overrides
# ---------------------------------------------------------------------------
Set-ConfigInt  "num_documents"  10
Set-ConfigValue "query_set_path" "query_set_debug.json"
Set-StressEnabled $false

# ---------------------------------------------------------------------------
# Track results
# ---------------------------------------------------------------------------
$results = @{}

# ---------------------------------------------------------------------------
# Helper: run a command, stream output live, capture to variable for analysis.
# Sets $script:lastCmdOutput. Returns $true if exit code was 0.
# ---------------------------------------------------------------------------
function Invoke-Streaming([string[]]$CmdArgs) {
    $lines = [System.Collections.Generic.List[string]]::new()
    & $CmdArgs[0] $CmdArgs[1..($CmdArgs.Length-1)] 2>&1 | ForEach-Object {
        Write-Host $_
        $lines.Add($_.ToString())
    }
    $script:lastCmdOutput = $lines -join "`n"
    return ($LASTEXITCODE -eq 0)
}

# ---------------------------------------------------------------------------
# Helper: analyse captured output — returns $true if the run looks healthy.
# "Healthy" means: no "^ERROR:" line AND failures=0/N (or no failures line at all).
# uv warnings like "warning: Failed to uninstall..." are intentionally ignored.
# ---------------------------------------------------------------------------
function Test-OutputHealthy([string]$Output) {
    # Any line that starts with ERROR: (our preflight / pipeline errors)
    if ($Output -match '(?m)^ERROR:') { return $false }
    # failures=N/M where N > 0
    if ($Output -match 'failures=(\d+)/\d+') {
        if ([int]$Matches[1] -gt 0) { return $false }
    }
    return $true
}

# ---------------------------------------------------------------------------
# Helper: run one backend smoke test
# ---------------------------------------------------------------------------
function Invoke-PreflightBackend([string]$Backend) {
    Write-Host ""
    Write-Host "--------------------------------------------------------" -ForegroundColor Cyan
    Write-Host "  Preflight: $Backend" -ForegroundColor Cyan
    Write-Host "--------------------------------------------------------" -ForegroundColor Cyan

    Set-ConfigValue "llm_backend" $Backend

    $pythonOk = $true
    $rustOk   = $true

    # --- Python pipeline ---
    if ($Backend -ne "llm_rs") {
        Write-Host "  [Python] Running..." -ForegroundColor DarkCyan
        $null = Invoke-Streaming @("uv", "run", "python", "-m", "python_pipeline.pipeline")
        $pythonOk = Test-OutputHealthy $script:lastCmdOutput
    } else {
        Write-Host "  [Python] Skipped (llm_rs is Rust-only)" -ForegroundColor DarkGray
    }

    # --- Rust pipeline ---
    Write-Host "  [Rust] Running..." -ForegroundColor DarkCyan
    $null = Invoke-Streaming @($rustBinary)
    $rustOk = Test-OutputHealthy $script:lastCmdOutput

    $overallOk = if ($Backend -eq "llm_rs") { $rustOk } else { $pythonOk -and $rustOk }
    $results[$Backend] = $overallOk

    if ($overallOk) {
        Write-Host "  RESULT: $Backend — PASS" -ForegroundColor Green
    } else {
        Write-Host "  RESULT: $Backend — FAIL" -ForegroundColor Red
        if (-not $pythonOk -and $Backend -ne "llm_rs") {
            Write-Host "    Python pipeline failed. Check output above." -ForegroundColor Yellow
        }
        if (-not $rustOk) {
            Write-Host "    Rust pipeline failed. Check output above." -ForegroundColor Yellow
        }
    }
}

# ---------------------------------------------------------------------------
# Run backends
# ---------------------------------------------------------------------------
if ($env:SKIP_OLLAMA -ne "1") {
    Invoke-PreflightBackend "ollama_http"
} else {
    Write-Host "==> Skipping ollama_http (SKIP_OLLAMA=1)" -ForegroundColor DarkGray
}

if ($env:SKIP_LLAMA_CPP -ne "1") {
    if ([string]::IsNullOrWhiteSpace($GgufPath)) {
        Write-Host "==> Skipping llama_cpp: gguf_model_path not set in benchmark_config.toml" -ForegroundColor Yellow
    } else {
        Invoke-PreflightBackend "llama_cpp"
    }
} else {
    Write-Host "==> Skipping llama_cpp (SKIP_LLAMA_CPP=1)" -ForegroundColor DarkGray
}

if ($env:SKIP_LLM_RS -ne "1") {
    if ([string]::IsNullOrWhiteSpace($GgufPath)) {
        Write-Host "==> Skipping llm_rs: gguf_model_path not set in benchmark_config.toml" -ForegroundColor Yellow
    } else {
        Invoke-PreflightBackend "llm_rs"
    }
} else {
    Write-Host "==> Skipping llm_rs (SKIP_LLM_RS=1)" -ForegroundColor DarkGray
}

# ---------------------------------------------------------------------------
# Restore original config
# ---------------------------------------------------------------------------
Set-ConfigInt  "num_documents"  ([int]$origNumDocs)
Set-ConfigValue "query_set_path" $origQuerySet
Set-ConfigValue "llm_backend"    $origBackend
Set-StressEnabled $false

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "========================================================"
Write-Host "  Preflight Summary" -ForegroundColor Cyan
Write-Host "========================================================"

$allPassed = $true
foreach ($backend in $results.Keys) {
    $ok = $results[$backend]
    if ($ok) {
        Write-Host "  $backend : PASS" -ForegroundColor Green
    } else {
        Write-Host "  $backend : FAIL" -ForegroundColor Red
        $allPassed = $false
    }
}

Write-Host "========================================================"
if ($allPassed) {
    Write-Host "  All backends passed. Safe to run .\run_full_benchmark.ps1" -ForegroundColor Green
} else {
    Write-Host "  One or more backends failed. Fix issues before running the full benchmark." -ForegroundColor Red
    exit 1
}
Write-Host "========================================================"
