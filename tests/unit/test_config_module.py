"""Tests for config module coverage"""
import pytest
from insightspike.config import ConfigKeys


def test_config_keys():
    """Test ConfigKeys class."""
    # Test that ConfigKeys has expected attributes
    assert hasattr(ConfigKeys, 'REASONING')
    assert hasattr(ConfigKeys, 'EMBEDDING')
    assert hasattr(ConfigKeys, 'GNN')
    assert hasattr(ConfigKeys, 'MEMORY')
    assert hasattr(ConfigKeys, 'LLM')
    
    # Test specific key values
    assert ConfigKeys.REASONING == 'reasoning'
    assert ConfigKeys.EMBEDDING == 'embedding'
    assert ConfigKeys.GNN == 'gnn'
    assert ConfigKeys.MEMORY == 'memory'
    assert ConfigKeys.LLM == 'llm'


def test_config_keys_usage():
    """Test that ConfigKeys can be used properly."""
    # Should be able to access all keys
    keys = [
        ConfigKeys.REASONING,
        ConfigKeys.EMBEDDING,
        ConfigKeys.GNN,
        ConfigKeys.MEMORY,
        ConfigKeys.LLM
    ]
    
    # All keys should be strings
    for key in keys:
        assert isinstance(key, str)
        assert len(key) > 0