"""
Test embedder module functionality
"""
import os
import sys
import pytest
import numpy as np

def test_get_model_singleton():
    """Test that get_model returns a singleton instance"""
    # Clean import of embedder module  
    import importlib
    
    # Force reimport to ensure clean state
    if 'insightspike.utils.embedder' in sys.modules:
        del sys.modules['insightspike.utils.embedder']
    
    # Import embedder module after environment is set
    embedder = importlib.import_module('insightspike.utils.embedder')
    
    # Debug information for CI
    lite_mode = os.getenv('INSIGHTSPIKE_LITE_MODE') == '1'
    print(f"INSIGHTSPIKE_LITE_MODE: {os.getenv('INSIGHTSPIKE_LITE_MODE')}")
    print(f"Lite mode detected: {lite_mode}")
    print(f"Python version: {sys.version}")
    print(f"SentenceTransformer class: {embedder.SentenceTransformer}")
    print(f"SentenceTransformer module: {embedder.SentenceTransformer.__module__}")
    
    # Reset the global model to ensure clean test
    embedder._model = None
    
    m1 = embedder.get_model()
    m2 = embedder.get_model()
    
    print(f"Model 1 type: {type(m1)}")
    print(f"Model 1 class name: {m1.__class__.__name__}")
    print(f"Model 1 module: {m1.__class__.__module__}")
    
    # Both should be the same instance (singleton behavior)
    assert m1 is m2
    
    # Should have an encode method
    assert hasattr(m1, 'encode')
    
    # Test that it can encode text
    result = m1.encode("test text")
    assert result is not None
    
    # Handle both numpy arrays and lists
    if hasattr(result, 'shape'):
        assert result.shape[0] == 1  # One text input
        assert result.shape[1] > 0   # Some embedding dimension
        embedding_dim = result.shape[1]
        print(f"Embedding result shape: {result.shape}")
    else:
        assert len(result) == 1      # One text input
        assert len(result[0]) > 0    # Some embedding dimension  
        embedding_dim = len(result[0])
        print(f"Embedding result length: {len(result)} x {len(result[0])}")
    
    # In lite mode, expect 384-dimensional vectors (as per embedder.py)
    if lite_mode:
        assert embedding_dim == 384, f"Expected 384-dim embeddings in lite mode, got {embedding_dim}"
        print("✅ Confirmed: 384-dimensional embeddings in lite mode")
        # Verify it's numpy array with correct dtype
        assert isinstance(result, np.ndarray), "Expected numpy array in lite mode"
        assert result.dtype == np.float32, f"Expected float32 dtype, got {result.dtype}"
    else:
        # Be flexible about dimensions in non-lite mode
        assert embedding_dim > 0, "Embeddings should have positive dimensions"
        print(f"ℹ️ Non-lite mode embeddings: {embedding_dim} dimensions")
