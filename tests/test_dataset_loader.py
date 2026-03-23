"""Unit tests for dataset_loader.py.

Tests use unittest.mock to avoid real network calls.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from python_pipeline.dataset_loader import DatasetLoadError, load_documents

# We patch _import_load_dataset so it returns a mock callable.
# This avoids the need for the 'datasets' package to be installed.
_IMPORT_PATCH = "python_pipeline.dataset_loader._import_load_dataset"


def _make_mock_dataset(texts: list[str]):
    """Return an iterable that yields records with a 'text' field."""
    return iter({"text": t} for t in texts)


def _make_load_dataset_mock(texts: list[str]) -> MagicMock:
    """Return a mock that behaves like datasets.load_dataset."""
    mock_fn = MagicMock(return_value=_make_mock_dataset(texts))
    return mock_fn


class TestLoadDocumentsCount(unittest.TestCase):
    """Test that load_documents returns exactly num_docs documents."""

    @patch(_IMPORT_PATCH)
    def test_returns_exact_num_docs(self, mock_import):
        texts = [f"Document number {i}" for i in range(20)]
        mock_import.return_value = _make_load_dataset_mock(texts)

        result = load_documents("wikipedia", "20220301.simple", 10)

        self.assertEqual(len(result), 10)
        self.assertEqual(result[0], "Document number 0")
        self.assertEqual(result[9], "Document number 9")

    @patch(_IMPORT_PATCH)
    def test_returns_all_when_dataset_smaller_than_num_docs(self, mock_import):
        texts = [f"Doc {i}" for i in range(5)]
        mock_import.return_value = _make_load_dataset_mock(texts)

        result = load_documents("wikipedia", "20220301.simple", 1000)

        self.assertEqual(len(result), 5)

    @patch(_IMPORT_PATCH)
    def test_passes_correct_args_to_load_dataset(self, mock_import):
        mock_fn = _make_load_dataset_mock(["text1"])
        mock_import.return_value = mock_fn

        load_documents("wikipedia", "20220301.simple", 1)

        mock_fn.assert_called_once_with(
            "wikipedia", "20220301.simple", split="train", streaming=True, trust_remote_code=False
        )

    @patch(_IMPORT_PATCH)
    def test_returns_text_field_values(self, mock_import):
        texts = ["Hello world", "Foo bar", "Baz qux"]
        mock_import.return_value = _make_load_dataset_mock(texts)

        result = load_documents("wikipedia", "20220301.simple", 3)

        self.assertEqual(result, texts)


class TestLoadDocumentsNetworkFailure(unittest.TestCase):
    """Test that DatasetLoadError is raised on simulated network failure."""

    @patch(_IMPORT_PATCH)
    def test_raises_dataset_load_error_on_load_failure(self, mock_import):
        mock_fn = MagicMock(side_effect=ConnectionError("Network unreachable"))
        mock_import.return_value = mock_fn

        with self.assertRaises(DatasetLoadError) as ctx:
            load_documents("wikipedia", "20220301.simple", 10)

        self.assertIn("wikipedia", str(ctx.exception))
        self.assertIn("20220301.simple", str(ctx.exception))

    @patch(_IMPORT_PATCH)
    def test_raises_dataset_load_error_on_streaming_failure(self, mock_import):
        def _failing_iter():
            yield {"text": "first doc"}
            raise ConnectionError("Connection dropped mid-stream")

        mock_fn = MagicMock(return_value=_failing_iter())
        mock_import.return_value = mock_fn

        with self.assertRaises(DatasetLoadError) as ctx:
            load_documents("wikipedia", "20220301.simple", 100)

        self.assertIn("Network error", str(ctx.exception))

    @patch(_IMPORT_PATCH)
    def test_error_message_is_descriptive(self, mock_import):
        mock_fn = MagicMock(side_effect=OSError("DNS resolution failed"))
        mock_import.return_value = mock_fn

        with self.assertRaises(DatasetLoadError) as ctx:
            load_documents("wikipedia", "20220301.simple", 10)

        msg = str(ctx.exception)
        self.assertTrue(len(msg) > 20, f"Error message too short: {msg!r}")

    def test_dataset_load_error_is_benchmark_error(self):
        from python_pipeline.config import BenchmarkError

        err = DatasetLoadError("test")
        self.assertIsInstance(err, BenchmarkError)


if __name__ == "__main__":
    unittest.main()

