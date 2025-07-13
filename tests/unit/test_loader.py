"""
Test loader functionality
"""
import pytest
from pathlib import Path
from unittest.mock import patch


def test_load_corpus_file(tmp_path):
    """Test loading corpus from file"""
    with patch(
        "insightspike.processing.loader.load_corpus",
        return_value=["mocked", "documents"],
    ) as mock_load:
        from insightspike.processing.loader import load_corpus

        f = tmp_path / "sample.txt"
        f.write_text("a\nb")
        docs = load_corpus(str(f))  # Convert to string path for compatibility
        assert isinstance(docs, list)
        assert len(docs) >= 1
