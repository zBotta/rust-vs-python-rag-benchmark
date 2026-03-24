# Rust vs Python RAG Benchmark

A proof-of-concept benchmark comparing end-to-end latency and resource usage of a
Retrieval-Augmented Generation (RAG) pipeline implemented in Python and Rust.

---

## Prerequisites

| Requirement | Version |
|---|---|
| [uv](https://docs.astral.sh/uv/) | latest |
| Python (managed by uv) | ≥ 3.11 (3.14 supported) |
| Rust toolchain (stable) | ≥ 1.78 |
| Ollama | latest (for `ollama_http` backend only) |
| System libraries | `libssl`, `libstdc++` (standard on most Linux/macOS) |

### Installing uv

**Windows:**
```powershell
winget install --id=astral-sh.uv -e
```

**Linux / macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

See the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) for other options.

### Installing Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup update stable
```

### Installing Ollama (required for `ollama_http` backend only)

Follow the instructions at <https://ollama.com/download>, then pull a model:

```bash
ollama pull llama3.2:3b
```

If you are behind a corporate proxy, see [Using a Local HuggingFace Model](#using-a-local-huggingface-model) below.

---

## Installation / Setup

**Linux / macOS:**
```bash
git clone <repo-url>
cd rust-vs-python-rag
bash setup.sh
```

**Windows (PowerShell):**
```powershell
git clone <repo-url>
cd rust-vs-python-rag
.\setup.ps1
```

`setup.sh` / `setup.ps1` does the following automatically:

1. Checks for `uv` and `cargo` on `PATH`
2. Initialises the uv project (`uv init`)
3. Installs all Python dependencies via `uv add` (see [Python Dependencies](#python-dependencies))
4. Syncs the virtual environment (`uv sync`)
5. Builds the Rust binary in release mode (`cargo build --release`)

No manual Python installation or `venv` activation is needed — uv handles it all.

---

## First-Run Downloads

On the first run, two assets are fetched automatically and cached locally. Subsequent runs use the cache and require no network access.

### ONNX embedding model (Rust pipeline)

The Rust pipeline embeds text using `sentence-transformers/all-MiniLM-L6-v2` in ONNX format via the `ort` crate. On first run, `hf-hub` downloads two files into the HuggingFace hub cache (`~/.cache/huggingface/hub/`):

- `onnx/model.onnx` (~86 MB)
- `tokenizer.json` (~226 KB)

If you are behind a corporate proxy with SSL inspection, pre-download them manually:

```powershell
# Windows — run once before the benchmark
$env:HTTPX_VERIFY="0"
$env:HF_HUB_DISABLE_SSL_VERIFICATION="1"
uv run python -c "
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
import httpx
_orig = httpx.Client.__init__
def _p(self, *a, **kw): kw['verify'] = False; _orig(self, *a, **kw)
httpx.Client.__init__ = _p
from huggingface_hub import hf_hub_download
hf_hub_download('sentence-transformers/all-MiniLM-L6-v2', 'onnx/model.onnx')
hf_hub_download('sentence-transformers/all-MiniLM-L6-v2', 'tokenizer.json')
print('Done.')
"
```

```bash
# Linux / macOS
DISABLE_SSL_VERIFY=1 uv run python -c "
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
from huggingface_hub import hf_hub_download
hf_hub_download('sentence-transformers/all-MiniLM-L6-v2', 'onnx/model.onnx')
hf_hub_download('sentence-transformers/all-MiniLM-L6-v2', 'tokenizer.json')
print('Done.')
"
```

You can also point the Rust pipeline at a pre-downloaded copy by setting environment variables before running:

```powershell
$env:ONNX_MODEL_PATH  = "C:\path\to\model.onnx"
$env:TOKENIZER_PATH   = "C:\path\to\tokenizer.json"
```

### Wikipedia parquet dataset

The Python pipeline reads documents from a local parquet file (`data/20231101.simple/train-00000-of-00001.parquet`) when it exists, avoiding any network call. The Rust pipeline always uses this local file.

If the file is missing, the benchmark scripts download it automatically via `download_parquet.py`. To download it manually:

```bash
uv run python download_parquet.py
# or with SSL bypass:
DISABLE_SSL_VERIFY=1 uv run python download_parquet.py
```

---

## Running the Benchmark

There are two orchestrator scripts depending on what you want to run.

### Single backend

Runs whichever `llm_backend` is set in `benchmark_config.toml`:

```bash
ollama serve &               # only needed for ollama_http backend
bash run_benchmark.sh        # Linux/macOS
.\run_benchmark.ps1          # Windows
```

### All backends (full suite)

Runs `ollama_http`, `llama_cpp`, and `llm_rs` in sequence, with an optional stress test pass for each:

```bash
# Standard benchmark across all backends
bash run_full_benchmark.sh

# With stress test enabled for every backend
STRESS=1 bash run_full_benchmark.sh

# Skip backends you don't want
SKIP_OLLAMA=1 bash run_full_benchmark.sh          # llama_cpp + llm_rs only
SKIP_LLM_RS=1 bash run_full_benchmark.sh          # ollama_http + llama_cpp only

# Windows equivalents
.\run_full_benchmark.ps1
$env:STRESS="1"; .\run_full_benchmark.ps1
$env:SKIP_OLLAMA="1"; .\run_full_benchmark.ps1
$env:SKIP_OLLAMA="1"; $env:STRESS="1"; .\run_full_benchmark.ps1
```

> **Note:** `llama_cpp` and `llm_rs` backends require `gguf_model_path` to be set in `benchmark_config.toml` (see [LLM Backend Modes](#llm-backend-modes)). They are automatically skipped if the path is empty.

Each backend writes to its own output files so nothing is overwritten:
- `output/metrics_python_ollama_http.jsonl` / `output/metrics_rust_ollama_http.jsonl`
- `output/metrics_python_llama_cpp.jsonl` / `output/metrics_rust_llama_cpp.jsonl`
- `output/metrics_rust_llm_rs.jsonl` (Python pipeline skipped for this backend)
- `output/benchmark_report_ollama_http.md`
- `output/benchmark_report_llama_cpp.md`
- `output/benchmark_report_llm_rs.md`

In addition, each pipeline run writes a structured log file alongside the metrics:
- `output/ollama_http-python.log` / `output/ollama_http-rust.log`
- `output/llama_cpp-python.log` / `output/llama_cpp-rust.log`
- `output/llm_rs-rust.log`

See [Benchmark Logs](#benchmark-logs) for the log format and configuration.

---

## Configuration

All tunable parameters live in `benchmark_config.toml`:

```toml
dataset_name    = "wikimedia/wikipedia"
dataset_subset  = "20231101.simple"
num_documents   = 1000
chunk_size      = 512
chunk_overlap   = 64
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
top_k           = 5
llm_model       = "lfm2-8b"
llm_host        = "http://localhost:11434"
llm_backend     = "ollama_http"   # "ollama_http" | "llama_cpp" | "llm_rs"
gguf_model_path = ""              # required when llm_backend != "ollama_http"
                                  # e.g. "C:\\Users\\you\\.models\\LFM2-8B-A1B-Q4_K_M.gguf"
query_set_path  = "query_set.json"
output_dir      = "output/"

[stress_test]
enabled           = false
concurrency       = 8
num_documents     = 10000
query_repetitions = 10
```

Override the LLM model at runtime:

```bash
BENCHMARK_MODEL=mistral:7b bash run_benchmark.sh
```

### LLM Backend Modes

| `llm_backend` | Python pipeline | Rust pipeline | Requires |
|---|---|---|---|
| `ollama_http` | HTTP → Ollama API | HTTP → Ollama API | Ollama running locally |
| `llama_cpp` | In-process `llama-cpp-python` | In-process `llama_cpp` crate | `gguf_model_path` set |
| `llm_rs` | Skipped (prints skip message) | In-process `llm` crate | `gguf_model_path` set |

When `llm_backend = "llama_cpp"` or `"llm_rs"`, you must set `gguf_model_path` to the path of a GGUF model file. The config loader will return a descriptive error and halt if it is absent.

When `llm_backend = "llm_rs"`, the Python pipeline exits cleanly with the message `"Python pipeline skipped: llm_rs backend is Rust-only"` and the report generator produces a single-pipeline (Rust-only) report.

### Rust llama.cpp Setup (Windows)

The Rust `llama_cpp` backend requires LLVM tooling at build time.

1. Install LLVM:

```powershell
winget install --id LLVM.LLVM -e
```

2. Open a new terminal, then confirm LLVM tools exist:

```powershell
Test-Path "C:\Program Files\LLVM\bin\libclang.dll"
Test-Path "C:\Program Files\LLVM\bin\llvm-nm.exe"
Test-Path "C:\Program Files\LLVM\bin\llvm-objcopy.exe"
```

3. Set required environment variables for the current shell:

```powershell
$llvmBin = "C:\Program Files\LLVM\bin"
$env:PATH = "$llvmBin;$env:PATH"
$env:LIBCLANG_PATH = $llvmBin
$env:NM_PATH = "$llvmBin\llvm-nm.exe"
$env:OBJCOPY_PATH = "$llvmBin\llvm-objcopy.exe"
```

4. Build Rust pipeline with the llama.cpp feature enabled:

```powershell
cargo build --release --manifest-path rust_pipeline/Cargo.toml --features llama_cpp_backend
```

5. Configure `benchmark_config.toml` for in-process llama.cpp:

```toml
llm_backend     = "llama_cpp"
gguf_model_path = "C:\\Users\\you\\.models\\Llama-3.2-1B-Instruct-Q4_K_S.gguf"
```

If the build fails with missing `nm`/`objcopy`, verify `NM_PATH` and `OBJCOPY_PATH` point to the LLVM executables above.

### Rust llama.cpp Loader Probe

For model-load diagnostics (before running the full benchmark), run the standalone probe binary:

```powershell
cargo run --release --manifest-path rust_pipeline/Cargo.toml --features llama_cpp_backend --bin llama_cpp_probe -- "C:\Users\you\.models\Llama-3.2-1B-Instruct-Q4_K_S.gguf"
```

The probe prints file metadata and attempts model loading with multiple parameter sets to provide more detailed failure context.

### llama.cpp Runtime Parity Microbenchmark

To isolate llama.cpp generation runtime (without dataset/chunking/embedding/index overhead), run the generation-only microbench in both languages with the same prompt and token settings.

Python:

```powershell
uv run python scripts/llama_cpp_microbench.py --model "C:\Users\you\.models\qwen2.5-0.5b-instruct-q4_k_m.gguf" --n-ctx 2048 --max-tokens 128 --warmup 1 --repeats 5 --output output_debug/llama_cpp_microbench_python.json
```

Rust (conservative mode):

```powershell
$llvmBin = "C:\Program Files\LLVM\bin"
$env:PATH = "$llvmBin;$env:PATH"
$env:LIBCLANG_PATH = $llvmBin
$env:NM_PATH = "$llvmBin\llvm-nm.exe"
$env:OBJCOPY_PATH = "$llvmBin\llvm-objcopy.exe"

cargo run --release --manifest-path rust_pipeline/Cargo.toml --features llama_cpp_backend --bin llama_cpp_microbench -- --model "C:\Users\you\.models\qwen2.5-0.5b-instruct-q4_k_m.gguf" --n-ctx 2048 --max-tokens 128 --warmup 1 --repeats 5 --conservative 1 --output output_debug/llama_cpp_microbench_rust_conservative.json
```

Rust (non-conservative mode, mmap enabled):

```powershell
cargo run --release --manifest-path rust_pipeline/Cargo.toml --features llama_cpp_backend --bin llama_cpp_microbench -- --model "C:\Users\you\.models\qwen2.5-0.5b-instruct-q4_k_m.gguf" --n-ctx 2048 --max-tokens 128 --warmup 1 --repeats 5 --conservative 0 --use-mmap 1 --use-mlock 0 --output output_debug/llama_cpp_microbench_rust_nonconservative.json
```

Compare `mean_total_ms`, `p50_total_ms`, and `p95_total_ms` between JSON outputs.

### Stress Test Mode

Set `stress_test.enabled = true` in `benchmark_config.toml` to run a concurrent load test after the standard sequential benchmark:

```toml
[stress_test]
enabled           = true
concurrency       = 8        # number of concurrent workers
num_documents     = 10000    # document corpus size for stress phase
query_repetitions = 10       # query set repeated N times
```

Stress test results are appended to the JSONL files as a `stress_summary` record and included in the report with throughput (QPS), peak RSS (MB), and p50/p95/p99 latency columns.

`concurrency` must be ≥ 1; the config loader returns an error otherwise.

---

## Benchmark Logs

Both pipelines write a structured log file for every run. The file is created (or truncated) at startup and written to `{output_dir}/{backend}-{language}.log`.

### Log file naming

| Pipeline | Backend | Log file |
|---|---|---|
| Python | `ollama_http` | `output/ollama_http-python.log` |
| Python | `llama_cpp` | `output/llama_cpp-python.log` |
| Rust | `ollama_http` | `output/ollama_http-rust.log` |
| Rust | `llama_cpp` | `output/llama_cpp-rust.log` |
| Rust | `llm_rs` | `output/llm_rs-rust.log` |

### Log record format

Each record is a single line:

```
2024-01-15T10:23:45.123Z [INFO] [loading] Starting dataset load: wikimedia/wikipedia / 20231101.simple, requested=1000 docs
```

Fields in order:

| Field | Format | Example |
|---|---|---|
| Timestamp | ISO-8601 UTC with millisecond precision | `2024-01-15T10:23:45.123Z` |
| Level | `[DEBUG]` / `[INFO]` / `[WARNING]` / `[ERROR]` | `[INFO]` |
| Stage | `[loading]` / `[chunking]` / `[embedding]` / `[index_build]` / `[retrieval]` / `[generation]` / `[summary]` | `[loading]` |
| Message | Free-form text | `Starting dataset load: ...` |

### Log levels

| Level | Numeric | What is logged |
|---|---|---|
| `DEBUG` | 0 | Per-query retrieval/generation start+complete, embedding batch progress every 100 chunks |
| `INFO` | 1 | Stage start/complete events, run summary (default) |
| `WARNING` | 2 | Zero-chunk output, failed LLM responses, unrecognised `log_level` value |
| `ERROR` | 3 | Exceptions in loading, embedding, index build, retrieval, generation |

### Configuring the log level

Set `log_level` in `benchmark_config.toml` (optional, defaults to `INFO`):

```toml
# log_level = "INFO"  # DEBUG | INFO | WARNING | ERROR  (default: INFO)
```

If an unrecognised value is supplied, both pipelines emit a `WARNING` to stderr and fall back to `INFO`.

### What each stage logs

| Stage | INFO records | DEBUG records |
|---|---|---|
| `loading` | start (dataset, subset, requested count), complete (actual count, elapsed ms) | — |
| `chunking` | start (chunk size, overlap), complete (chunk count, elapsed ms), zero-chunk warning | — |
| `embedding` | start (model, chunk count), complete (elapsed ms) | progress every 100 chunks embedded |
| `index_build` | start (embedding count), complete (elapsed ms) | — |
| `retrieval` | — | start (query id), complete (query id, chunks retrieved, elapsed ms) |
| `generation` | — | start (query id, context chunks), complete (query id, tokens, TTFT ms, generation ms) |
| `summary` | run complete (total queries, failures, p50 ms, p95 ms, output path), stress complete (QPS, peak RSS MB, p99 ms) | — |

---

## Python Dependencies

The environment is managed by uv. `requirements.txt` is a human-readable pinned snapshot; `uv.lock` is the authoritative lock file.

| Package | Version | Purpose |
|---|---|---|
| `datasets` | 4.8.3 | HuggingFace dataset loader (network fallback) |
| `pyarrow` | — | Local parquet reading (primary dataset path) |
| `langchain-text-splitters` | 1.1.1 | Document chunking |
| `hnswlib` | 0.8.0 | In-memory HNSW vector index |
| `httpx` | 0.28.1 | Ollama HTTP client |
| `onnxruntime` | 1.24.4 | ONNX embedding inference (same model as Rust) |
| `tokenizers` | 0.22.2 | Tokenizer for `bert-base-uncased` vocab |
| `llama-cpp-python` | 0.3.16 | In-process LLM inference (`llama_cpp` backend) |
| `psutil` | 7.2.2 | Peak RSS memory measurement (stress test) |
| `matplotlib` | 3.10.8 | Latency histogram/CDF plots in report |
| `numpy` | 2.4.3 | Percentile computation for plots |
| `pytest` | 9.0.2 | Test runner |
| `hypothesis` | 6.151.9 | Property-based testing |
| `respx` | 0.22.0 | HTTP mock for LLM client tests |
| `tomli` | 2.4.0 | TOML parsing (Python < 3.11 fallback) |

> `sentence-transformers` is no longer a direct dependency. Both pipelines load the `all-MiniLM-L6-v2` ONNX weights directly via `onnxruntime` / `ort`, ensuring identical embeddings.

---

## Running Tests

```bash
# Python property and unit tests
uv run pytest tests/

# Rust tests (including proptest property tests)
cargo test --manifest-path rust_pipeline/Cargo.toml
```

---

## Using a Local HuggingFace Model

If Ollama cannot reach the internet (corporate proxy), you can serve a locally downloaded GGUF model.

**Step 1 — Download a GGUF model:**

Download a `.gguf` file from HuggingFace, e.g. [LiquidAI/LFM2-8B-A1B-GGUF](https://huggingface.co/LiquidAI/LFM2-8B-A1B-GGUF/tree/main) (Q4_K_M recommended for 8 GB GPU).

**Step 2 — Create a `Modelfile`:**

```
FROM ./models/LFM2-8B-A1B-Q4_K_M.gguf

PARAMETER num_ctx 4096
PARAMETER num_gpu 99
```

**Step 3 — Register with Ollama:**

```bash
ollama create lfm2-8b -f Modelfile
ollama list   # verify it appears
```

**Step 4 — Update `benchmark_config.toml`:**

```toml
llm_model   = "lfm2-8b"
llm_backend = "ollama_http"
llm_host    = "http://localhost:11434"
```

**Step 5 — Verify:**

```bash
ollama serve
curl http://localhost:11434/api/generate -d '{"model":"lfm2-8b","prompt":"Hello","stream":false}'
```

Alternatively, use `llm_backend = "llama_cpp"` to run the model fully in-process (no Ollama required):

```toml
llm_backend     = "llama_cpp"
gguf_model_path = "./models/LFM2-8B-A1B-Q4_K_M.gguf"
```

---

## Project Structure

```
rust-vs-python-rag/
├── benchmark_config.toml   # Shared configuration
├── query_set.json           # 50 benchmark questions
├── requirements.in          # Unpinned direct Python dependencies (reference)
├── requirements.txt         # Pinned snapshot (human-readable)
├── uv.lock                  # Authoritative lock file (managed by uv)
├── pyproject.toml           # uv project manifest
├── Cargo.toml               # Rust workspace
├── Cargo.lock               # Pinned Rust dependencies
├── setup.sh                 # Environment bootstrap (Linux/macOS)
├── setup.ps1                # Environment bootstrap (Windows)
├── run_benchmark.sh         # Single-backend orchestrator (Linux/macOS)
├── run_benchmark.ps1        # Single-backend orchestrator (Windows)
├── run_full_benchmark.sh    # All-backends orchestrator (Linux/macOS)
├── run_full_benchmark.ps1   # All-backends orchestrator (Windows)
├── python_pipeline/         # Python RAG implementation
│   ├── pipeline.py          # Entry point (wires all components)
│   ├── stress_runner.py     # Concurrent stress test dispatcher
│   ├── logger.py            # Structured per-backend log writer
│   ├── dataset_loader.py
│   ├── chunker.py
│   ├── embedder.py          # ONNX Runtime inference
│   ├── vector_store.py      # hnswlib HNSW index
│   ├── retriever.py
│   ├── llm_client.py        # Ollama HTTP client
│   ├── llm_client_llama_cpp.py  # In-process llama-cpp-python client
│   ├── metrics_collector.py
│   └── config.py
├── rust_pipeline/           # Rust RAG implementation
│   └── src/
│       ├── main.rs          # Entry point
│       ├── logger.rs        # Structured per-backend log writer
│       ├── stress_runner.rs # Concurrent stress test dispatcher
│       ├── embedder.rs      # ort (ONNX Runtime) inference
│       ├── llm_client.rs    # Ollama HTTP client
│       ├── llm_client_llama_cpp.rs  # In-process llama_cpp crate
│       ├── llm_client_llm_rs.rs     # In-process llm crate
│       └── ...
├── report/
│   └── generate_report.py  # Markdown report generator
└── tests/                   # Python property and unit tests
```

---

## Troubleshooting

**SSL errors behind a corporate proxy:**

Set `DISABLE_SSL_VERIFY=1` when running any benchmark or setup script:

```bash
DISABLE_SSL_VERIFY=1 bash run_full_benchmark.sh
```

```powershell
$env:DISABLE_SSL_VERIFY="1"; .\run_full_benchmark.ps1
```

This sets `HTTPX_VERIFY=0`, `HF_HUB_DISABLE_SSL_VERIFICATION=1`, and disables Python's default SSL context. The Python pipeline uses `HTTPX_VERIFY=0` (read by httpx at client creation time) rather than monkey-patching, which is required for compatibility with Python 3.14.

**ONNX model not found (Rust pipeline fails on first run):**

The Rust pipeline downloads `onnx/model.onnx` (~86 MB) from HuggingFace on first run via `hf-hub`. If you are behind a proxy, pre-download it manually — see [ONNX embedding model](#onnx-embedding-model-rust-pipeline) above. You can also bypass the download entirely by pointing to a local copy:

```powershell
$env:ONNX_MODEL_PATH = "C:\path\to\model.onnx"
$env:TOKENIZER_PATH  = "C:\path\to\tokenizer.json"
.\run_full_benchmark.ps1
```

**Python pipeline fails with `DatasetLoadError` / `Cannot send a request, as the client has been closed`:**

This is a Python 3.14 + httpx compatibility issue with the old SSL monkey-patch. It is fixed in the current codebase — the pipeline now uses `HTTPX_VERIFY=0` instead. If you see this error, ensure you are running the latest version of `pipeline.py`.

The pipeline also now reads documents from the local parquet file (`data/20231101.simple/train-00000-of-00001.parquet`) first, so network access is not required once the file is present.

**Rust binary not found:**
```bash
cargo build --release --manifest-path rust_pipeline/Cargo.toml
```

**`gguf_model_path` error on startup:**

Set `gguf_model_path` in `benchmark_config.toml` when using `llama_cpp` or `llm_rs` backends:

```toml
gguf_model_path = "C:\\Users\\you\\.models\\LFM2-8B-A1B-Q4_K_M.gguf"
```

Or switch to `llm_backend = "ollama_http"` if you have Ollama running.

**Rust llama_cpp build fails on Windows (`libclang`, `nm`, or `objcopy` not found):**

Set LLVM paths in your PowerShell session, then rebuild with the backend feature:

```powershell
$llvmBin = "C:\Program Files\LLVM\bin"
$env:PATH = "$llvmBin;$env:PATH"
$env:LIBCLANG_PATH = $llvmBin
$env:NM_PATH = "$llvmBin\llvm-nm.exe"
$env:OBJCOPY_PATH = "$llvmBin\llvm-objcopy.exe"

cargo build --release --manifest-path rust_pipeline/Cargo.toml --features llama_cpp_backend
```

If model loading still fails for a specific GGUF (for example `Llama-3.2-1B-Instruct-Q4_K_S.gguf`), run the standalone probe to capture diagnostics:

```powershell
cargo run --release --manifest-path rust_pipeline/Cargo.toml --features llama_cpp_backend --bin llama_cpp_probe -- "C:\Users\you\.models\Llama-3.2-1B-Instruct-Q4_K_S.gguf"
```

**Parquet dataset missing (Rust pipeline):**
The benchmark scripts auto-download it via `download_parquet.py` if not present in `data/`. Run manually with:
```bash
uv run python download_parquet.py
```
