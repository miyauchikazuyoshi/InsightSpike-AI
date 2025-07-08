"""Tests for text utilities"""
import pytest
from insightspike.utils.text_utils import clean_text, iter_text


def test_clean_text():
    """Test text cleaning function."""
    # Basic cleaning
    assert clean_text("  Hello World  ") == "Hello World"
    assert clean_text("Hello\n\nWorld") == "Hello World"
    assert clean_text("Hello\tWorld") == "Hello World"
    
    # Empty and None
    assert clean_text("") == ""
    assert clean_text("   ") == ""
    
    # Multiple spaces
    assert clean_text("Hello   World") == "Hello World"
    
    # Mixed whitespace
    assert clean_text("  Hello\n\t World  \n") == "Hello World"


def test_iter_text():
    """Test text iteration function."""
    # Test with Path object
    from pathlib import Path
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        
        # Create test files
        (tmppath / "test1.txt").write_text("content1")
        (tmppath / "test2.txt").write_text("content2")
        (tmppath / "test.md").write_text("markdown")
        
        # Test finding .txt files
        result = list(iter_text(tmppath))
        assert len(result) == 2
        assert all(str(p).endswith('.txt') for p in result)
        
        # Test with different suffix
        result = list(iter_text(tmppath, suffix=".md"))
        assert len(result) == 1
        assert str(result[0]).endswith('.md')