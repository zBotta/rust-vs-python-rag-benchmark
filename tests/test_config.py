"""Property-based tests for the benchmark configuration loader.

Sub-task 1.1 — Property 15: Missing required config key produces descriptive error
Sub-task 1.2 — Property 16: Absent optional config key uses documented default

Validates: Requirements 9.3, 9.4
"""

from __future__ import annotations

import tomllib
import tempfile
import os
from pathlib import Path

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------
import sys

# Allow running tests from the project root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from python_pipeline.config import (
    BenchmarkError,
    BenchmarkConfig,
    REQUIRED_KEYS,
    OPTIONAL_DEFAULTS,
    load_config,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FULL_CONFIG: dict = {
    "dataset_name": "wikipedia",
    "dataset_subset": "20220301.simple",
    "num_documents": 1000,
    "chunk_size": 512,
    "chunk_overlap": 64,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "top_k": 5,
    "llm_model": "llama3.2:3b",
    "llm_host": "http://localhost:11434",
    "query_set_path": "query_set.json",
    "output_dir": "output/",
}


def _write_toml(data: dict) -> str:
    """Write a dict as a TOML file to a temp path and return the path."""
    lines: list[str] = []
    for k, v in data.items():
        if isinstance(v, str):
            lines.append(f'{k} = "{v}"')
        else:
            lines.append(f"{k} = {v}")
    content = "\n".join(lines) + "\n"

    fd, path = tempfile.mkstemp(suffix=".toml")
    with os.fdopen(fd, "w") as fh:
        fh.write(content)
    return path


# ---------------------------------------------------------------------------
# Sub-task 1.1 — Property 15
# ---------------------------------------------------------------------------

# Feature: rust-vs-python-rag-benchmark, Property 15: Missing required config key produces descriptive error


@given(key=st.sampled_from(REQUIRED_KEYS))
@settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_missing_required_key_produces_descriptive_error(key: str) -> None:
    """For any required key that is absent, load_config must raise BenchmarkError
    whose message identifies the missing key by name.

    # Feature: rust-vs-python-rag-benchmark, Property 15: Missing required config key produces descriptive error
    """
    # Build a config that omits exactly one required key.
    partial = {k: v for k, v in FULL_CONFIG.items() if k != key}
    path = _write_toml(partial)
    try:
        with pytest.raises(BenchmarkError) as exc_info:
            load_config(path)
        error_message = str(exc_info.value)
        assert key in error_message, (
            f"Error message should mention missing key '{key}', got: {error_message!r}"
        )
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Sub-task 1.2 — Property 16
# ---------------------------------------------------------------------------

# Feature: rust-vs-python-rag-benchmark, Property 16: Absent optional config key uses documented default


# Strategy: pick any subset of optional keys to omit (currently empty set,
# but the test is structured to grow as optional keys are added).
@given(omitted=st.frozensets(st.sampled_from(list(OPTIONAL_DEFAULTS.keys()) or ["_sentinel"])))
@settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_absent_optional_key_uses_documented_default(omitted: frozenset) -> None:
    """For any subset of optional keys that are absent from the config, the
    loaded BenchmarkConfig must contain the documented default value for each
    omitted key.

    # Feature: rust-vs-python-rag-benchmark, Property 16: Absent optional config key uses documented default
    """
    # Build a config that includes all required keys but omits the chosen optional ones.
    config_data = dict(FULL_CONFIG)
    for key in omitted:
        config_data.pop(key, None)

    path = _write_toml(config_data)
    try:
        cfg = load_config(path)
        # Verify each omitted optional key has its documented default.
        for key in omitted:
            if key in OPTIONAL_DEFAULTS:
                actual = getattr(cfg, key, None)
                expected = OPTIONAL_DEFAULTS[key]
                assert actual == expected, (
                    f"Optional key '{key}' should default to {expected!r}, got {actual!r}"
                )
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Sub-task 17.2 — Property 21
# ---------------------------------------------------------------------------

# Feature: rust-vs-python-rag-benchmark, Property 21: Config with llm_backend=llama_cpp or llm_rs and absent gguf_model_path → error naming gguf_model_path

IN_PROCESS_BACKENDS = ["llama_cpp", "llm_rs"]


@given(
    backend=st.sampled_from(IN_PROCESS_BACKENDS),
    gguf_present=st.booleans(),
)
@settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_gguf_model_path_required_for_in_process_backends(
    backend: str, gguf_present: bool
) -> None:
    """For any config where llm_backend is 'llama_cpp' or 'llm_rs' and
    gguf_model_path is absent or empty string, load_config must raise
    BenchmarkError whose message identifies 'gguf_model_path'.

    # Feature: rust-vs-python-rag-benchmark, Property 21: Config with llm_backend=llama_cpp or llm_rs and absent gguf_model_path → error naming gguf_model_path
    """
    config_data = dict(FULL_CONFIG)
    config_data["llm_backend"] = backend

    if gguf_present:
        # Empty string is treated as absent — must still error.
        config_data["gguf_model_path"] = ""
    else:
        # Key entirely absent.
        config_data.pop("gguf_model_path", None)

    path = _write_toml(config_data)
    try:
        with pytest.raises(BenchmarkError) as exc_info:
            load_config(path)
        error_message = str(exc_info.value)
        assert "gguf_model_path" in error_message, (
            f"Error message should mention 'gguf_model_path', got: {error_message!r}"
        )
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Baseline unit tests
# ---------------------------------------------------------------------------


def test_full_config_loads_successfully() -> None:
    """A complete config with all required keys must load without error."""
    path = _write_toml(FULL_CONFIG)
    try:
        cfg = load_config(path)
        assert isinstance(cfg, BenchmarkConfig)
        assert cfg.dataset_name == "wikipedia"
        assert cfg.num_documents == 1000
        assert cfg.chunk_size == 512
        assert cfg.top_k == 5
    finally:
        os.unlink(path)


def test_all_required_keys_individually() -> None:
    """Each required key, when missing, must produce an error naming that key."""
    for key in REQUIRED_KEYS:
        partial = {k: v for k, v in FULL_CONFIG.items() if k != key}
        path = _write_toml(partial)
        try:
            with pytest.raises(BenchmarkError) as exc_info:
                load_config(path)
            assert key in str(exc_info.value), (
                f"Error for missing '{key}' should mention the key"
            )
        finally:
            os.unlink(path)

