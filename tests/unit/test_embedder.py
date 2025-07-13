"""
Test embedder module functionality
"""
import os
import sys

import numpy as np
import pytest


def test_get_model_singleton():
    """Test that get_model returns a singleton instance"""
    # Clean import of embedder module
    import importlib

    # Force reimport to ensure clean state
    if "insightspike.utils.embedder" in sys.modules:
        del sys.modules["insightspike.utils.embedder"]

    # Import embedder module after environment is set
    embedder = importlib.import_module("insightspike.utils.embedder")

    # Debug information for CI
    lite_mode = os.getenv("INSIGHTSPIKE_LITE_MODE") == "1"
    safe_mode = os.getenv("INSIGHTSPIKE_SAFE_MODE") == "1"
    print(f"INSIGHTSPIKE_LITE_MODE: {os.getenv('INSIGHTSPIKE_LITE_MODE')}")
    print(f"INSIGHTSPIKE_SAFE_MODE: {os.getenv('INSIGHTSPIKE_SAFE_MODE')}")
    print(f"Lite mode detected: {lite_mode}")
    print(f"Safe mode detected: {safe_mode}")
    print(f"Python version: {sys.version}")

    # Test get_model function
    m1 = embedder.get_model()
    m2 = embedder.get_model()

    print(f"Model 1 type: {type(m1)}")
    print(f"Model 1 class name: {m1.__class__.__name__}")

    # Both should be the same instance (singleton behavior)
    assert m1 is m2

    # Should have an encode method
    assert hasattr(m1, "encode")

    # Test that it can encode text
    result = m1.encode("test text")
    assert result is not None

    # Handle both numpy arrays and lists
    if hasattr(result, "shape"):
        assert result.shape[0] == 1  # One text input
        assert result.shape[1] > 0  # Some embedding dimension
        embedding_dim = result.shape[1]
        print(f"Embedding result shape: {result.shape}")
    else:
        assert len(result) == 1  # One text input
        assert len(result[0]) > 0  # Some embedding dimension
        embedding_dim = len(result[0])
        print(f"Embedding result length: {len(result)} x {len(result[0])}")

    # Test EmbeddingManager class directly
    manager = embedder.EmbeddingManager()
    model = manager.get_model()
    assert model is not None
    assert hasattr(model, "encode")

    # Test encoding with manager
    result2 = model.encode("another test text")
    assert result2 is not None

    print("✅ All embedder tests passed")
