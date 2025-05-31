"""
Test embedder module functionality
"""
import sys
import pytest
from unittest.mock import patch, MagicMock

# Mock sentence_transformers before any imports
mock_sentence_transformers = MagicMock()

class MockSentenceTransformer:
    """Mock class that mimics SentenceTransformer behavior"""
    def __init__(self, model_name, device=None):
        self.model_name = model_name
        self.device = device
    
    def encode(self, texts, normalize_embeddings=True):
        # Return mock embeddings
        if isinstance(texts, str):
            return [[0.1, 0.2, 0.3]]
        return [[0.1, 0.2, 0.3] for _ in texts]

mock_sentence_transformers.SentenceTransformer = MockSentenceTransformer

def test_get_model_singleton():
    """Test that get_model returns a singleton instance"""
    # Patch at the sys.modules level to ensure complete isolation
    with patch.dict('sys.modules', {'sentence_transformers': mock_sentence_transformers}):
        # Force reimport of embedder module
        if 'insightspike.embedder' in sys.modules:
            del sys.modules['insightspike.embedder']
        
        # Import embedder with mocked dependencies
        import importlib
        embedder = importlib.import_module('insightspike.embedder')
        
        # Reset the global model to ensure clean test
        embedder._model = None
        
        m1 = embedder.get_model()
        m2 = embedder.get_model()
        
        # Both should be the same instance (singleton behavior)
        assert m1 is m2
        
        # Should be our mock type
        assert isinstance(m1, MockSentenceTransformer)
        
        # Should have an encode method
        assert hasattr(m1, 'encode')
        
        # Test that it can encode text
        result = m1.encode("test text")
        assert result is not None
        assert len(result) == 1
        assert len(result[0]) == 3  # Our mock returns 3-dimensional vectors
