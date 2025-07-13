"""Tests for config module coverage"""
import pytest
from insightspike.config import SimpleConfig, get_config


def test_config_keys():
    """Test Config class and get_config function."""
    # Test get_config returns a config (legacy by default)
    config = get_config()
    assert config is not None

    # Test simple config
    simple_config = get_config(legacy=False)
    assert isinstance(simple_config, SimpleConfig)

    # Test config has expected attributes
    assert hasattr(config, "reasoning")
    assert hasattr(config, "embedding")
    assert hasattr(config, "gnn")
    assert hasattr(config, "memory")
    assert hasattr(config, "llm")


def test_config_keys_usage():
    """Test that config can be used properly."""
    config = get_config()

    # Test reasoning config
    assert hasattr(config.reasoning, "similarity_threshold")
    assert isinstance(config.reasoning.similarity_threshold, (int, float))

    # Test embedding config
    assert hasattr(config.embedding, "dimension")
    assert isinstance(config.embedding.dimension, int)
