"""Dataset loader — loads documents from HuggingFace datasets library."""
from __future__ import annotations

import os
import pathlib
import shutil

from python_pipeline.config import BenchmarkError


class DatasetLoadError(BenchmarkError):
    """Raised when the dataset cannot be fetched."""


def _import_load_dataset():
    """Import and return datasets.load_dataset, raising DatasetLoadError if unavailable."""
    try:
        from datasets import load_dataset  # type: ignore[import]
        return load_dataset
    except ImportError as exc:
        raise DatasetLoadError(
            f"The 'datasets' library is not installed. Run: pip install datasets. Error: {exc}"
        ) from exc


def load_documents(dataset_name: str, subset: str, num_docs: int) -> list[str]:
    """Load documents from HuggingFace datasets library.

    Args:
        dataset_name: HuggingFace dataset name (e.g. "wikipedia").
        subset: Dataset subset/config (e.g. "20220301.simple").
        num_docs: Maximum number of documents to return.

    Returns:
        List of document text strings (first num_docs documents).

    Raises:
        DatasetLoadError: On network failure or dataset not found.
    """
    load_dataset = _import_load_dataset()

    try:
        ds = load_dataset(dataset_name, subset, split="train", streaming=True, trust_remote_code=False)
    except DatasetLoadError:
        raise
    except Exception as exc:
        raise DatasetLoadError(
            f"Failed to load dataset '{dataset_name}/{subset}' from network: {exc}"
        ) from exc

    documents: list[str] = []
    try:
        for record in ds:
            if len(documents) >= num_docs:
                break
            text = record.get("text", "")
            if text:
                documents.append(text)
    except Exception as exc:
        raise DatasetLoadError(
            f"Network error while streaming dataset '{dataset_name}/{subset}': {exc}"
        ) from exc

    return documents
