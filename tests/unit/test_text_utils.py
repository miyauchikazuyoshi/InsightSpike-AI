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
    # Test file path
    result = list(iter_text("test.txt"))
    assert result == ["test.txt"]
    
    # Test list of texts
    texts = ["Hello", "World"]
    result = list(iter_text(texts))
    assert result == texts
    
    # Test single string
    result = list(iter_text("Hello"))
    assert result == ["Hello"]
    
    # Test empty list
    result = list(iter_text([]))
    assert result == []