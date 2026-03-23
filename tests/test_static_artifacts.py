"""
Unit tests for static project artifacts.
Validates: Requirements 7.1, 9.2, 10.1
"""

import json
import sys
from pathlib import Path

import pytest

# Project root is one level up from the tests/ directory
PROJECT_ROOT = Path(__file__).parent.parent


def test_query_set_has_50_entries():
    """Assert query_set.json contains exactly 50 entries."""
    query_set_path = PROJECT_ROOT / "query_set.json"
    assert query_set_path.exists(), "query_set.json not found in project root"

    with query_set_path.open() as f:
        entries = json.load(f)

    assert isinstance(entries, list), "query_set.json must be a JSON array"
    assert len(entries) == 50, f"Expected 50 entries, got {len(entries)}"


def test_benchmark_config_loads_with_all_required_keys():
    """Assert benchmark_config.toml loads successfully with all required keys present."""
    config_path = PROJECT_ROOT / "benchmark_config.toml"
    assert config_path.exists(), "benchmark_config.toml not found in project root"

    if sys.version_info >= (3, 11):
        import tomllib
        with config_path.open("rb") as f:
            config = tomllib.load(f)
    else:
        import tomli
        with config_path.open("rb") as f:
            config = tomli.load(f)

    required_keys = [
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

    missing = [key for key in required_keys if key not in config]
    assert not missing, f"benchmark_config.toml is missing required keys: {missing}"


def test_readme_exists_and_contains_prerequisite_sections():
    """Assert README.md exists and contains prerequisite sections."""
    readme_path = PROJECT_ROOT / "README.md"
    assert readme_path.exists(), "README.md not found in project root"

    content = readme_path.read_text(encoding="utf-8")

    # Must document Python ≥ 3.11
    assert "3.11" in content, "README.md must mention Python 3.11 requirement"

    # Must document Rust stable ≥ 1.78
    assert "1.78" in content, "README.md must mention Rust 1.78 requirement"

    # Must mention Ollama
    assert "Ollama" in content or "ollama" in content, \
        "README.md must mention Ollama as a prerequisite"

    # Must have a Prerequisites section
    assert "Prerequisites" in content or "prerequisites" in content, \
        "README.md must contain a Prerequisites section"

