"""Tests for mock factory utilities"""
import pytest
from insightspike.utils.mock_factory import create_mock_llm_response


def test_create_mock_llm_response():
    """Test mock LLM response creation."""
    # Test with default parameters
    response = create_mock_llm_response("Test query")
    assert isinstance(response, str)
    assert len(response) > 0
    assert "Test query" in response or "insight" in response.lower()
    
    # Test with custom response
    custom_response = "Custom test response"
    response = create_mock_llm_response("Query", response_text=custom_response)
    assert response == custom_response
    
    # Test with None query
    response = create_mock_llm_response(None)
    assert isinstance(response, str)
    assert len(response) > 0