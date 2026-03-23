"""Configuration loader for the RAG benchmark.

Reads benchmark_config.toml, validates required keys, and applies defaults
for optional keys. Raises BenchmarkError with the missing key name if any
required key is absent.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


class BenchmarkError(Exception):
    """Base exception for all benchmark errors."""


REQUIRED_KEYS: list[str] = [
    "dataset_name",
    "dataset_subset",
    "num_documents",
    "chunk_size",
    "chunk_overlap",
    "embedding_model",
    "top_k",
    "llm_model",
    "llm_host",
    "query_set_path",
    "output_dir",
]

# Optional keys with their documented defaults.
OPTIONAL_DEFAULTS: dict[str, object] = {
    "llm_backend": "ollama_http",
    "gguf_model_path": "",
}


@dataclass
class StressTestConfig:
    enabled: bool
    concurrency: int
    num_documents: int
    query_repetitions: int


@dataclass
class BenchmarkConfig:
    dataset_name: str
    dataset_subset: str
    num_documents: int
    chunk_size: int
    chunk_overlap: int
    embedding_model: str
    top_k: int
    llm_model: str
    llm_host: str
    query_set_path: str
    output_dir: str
    llm_backend: str
    gguf_model_path: str
    stress_test: StressTestConfig


def load_config(config_path: str | Path = "benchmark_config.toml") -> BenchmarkConfig:
    """Load and validate benchmark configuration from a TOML file.

    Args:
        config_path: Path to the TOML configuration file.

    Returns:
        A validated BenchmarkConfig dataclass.

    Raises:
        BenchmarkError: If a required key is missing, naming the missing key.
        FileNotFoundError: If the config file does not exist.
    """
    path = Path(config_path)
    with path.open("rb") as fh:
        raw: dict = tomllib.load(fh)

    # Apply optional defaults first so required-key validation sees the full picture.
    for key, default in OPTIONAL_DEFAULTS.items():
        raw.setdefault(key, default)

    # Validate required keys.
    for key in REQUIRED_KEYS:
        if key not in raw:
            raise BenchmarkError(
                f"Missing required configuration key: '{key}' in {config_path}"
            )

    # Validate gguf_model_path is set when using an in-process backend.
    llm_backend = str(raw["llm_backend"])
    gguf_model_path = str(raw["gguf_model_path"])
    if llm_backend in ("llama_cpp", "llm_rs") and not gguf_model_path:
        raise BenchmarkError(
            f"Missing required configuration key: 'gguf_model_path' in {config_path}"
        )

    # Parse optional [stress_test] subsection
    stress_raw: dict = raw.get("stress_test", {})
    stress_concurrency = int(stress_raw.get("concurrency", 8))
    if stress_concurrency < 1:
        raise BenchmarkError(
            "Configuration validation error: stress_test.concurrency must be >= 1"
        )
    stress_test = StressTestConfig(
        enabled=bool(stress_raw.get("enabled", False)),
        concurrency=stress_concurrency,
        num_documents=int(stress_raw.get("num_documents", 10000)),
        query_repetitions=int(stress_raw.get("query_repetitions", 10)),
    )

    return BenchmarkConfig(
        dataset_name=raw["dataset_name"],
        dataset_subset=raw["dataset_subset"],
        num_documents=int(raw["num_documents"]),
        chunk_size=int(raw["chunk_size"]),
        chunk_overlap=int(raw["chunk_overlap"]),
        embedding_model=raw["embedding_model"],
        top_k=int(raw["top_k"]),
        llm_model=raw["llm_model"],
        llm_host=raw["llm_host"],
        query_set_path=raw["query_set_path"],
        output_dir=raw["output_dir"],
        llm_backend=llm_backend,
        gguf_model_path=gguf_model_path,
        stress_test=stress_test,
    )
