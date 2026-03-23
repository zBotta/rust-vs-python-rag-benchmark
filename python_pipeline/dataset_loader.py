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


def _load_from_local_parquet(subset: str, num_docs: int) -> list[str] | None:
    """Try to load documents from the local parquet file.

    Returns a list of document strings if the local file exists, else None.
    """
    parquet_path = pathlib.Path("data") / subset / "train-00000-of-00001.parquet"
    if not parquet_path.exists():
        return None

    try:
        import pyarrow.parquet as pq  # type: ignore[import]
    except ImportError as exc:
        raise DatasetLoadError(
            "The 'pyarrow' library is not installed. Run: uv add pyarrow  (or pip install pyarrow)"
        ) from exc

    try:
        table = pq.read_table(str(parquet_path), columns=["text"])
        documents: list[str] = []
        for batch in table.to_batches():
            for text in batch.column("text").to_pylist():
                if text and len(documents) < num_docs:
                    documents.append(text)
                if len(documents) >= num_docs:
                    break
            if len(documents) >= num_docs:
                break
        return documents
    except Exception:
        return None


def load_documents(dataset_name: str, subset: str, num_docs: int) -> list[str]:
    """Load documents from local parquet file or HuggingFace datasets library.

    Tries the local parquet file first (data/{subset}/train-00000-of-00001.parquet)
    to avoid network calls. Falls back to streaming from HuggingFace if not found.

    Args:
        dataset_name: HuggingFace dataset name (e.g. "wikipedia").
        subset: Dataset subset/config (e.g. "20231101.simple").
        num_docs: Maximum number of documents to return.

    Returns:
        List of document text strings (first num_docs documents).

    Raises:
        DatasetLoadError: On network failure or dataset not found.
    """
    # Fast path: use local parquet if available (avoids network + SSL issues)
    local_docs = _load_from_local_parquet(subset, num_docs)
    if local_docs is not None:
        print(f"Loaded {len(local_docs)} documents from local parquet.")
        return local_docs

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
