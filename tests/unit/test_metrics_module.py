"""Tests for metrics module to improve coverage"""
import pytest
from unittest.mock import Mock, MagicMock
import numpy as np

from insightspike.metrics import delta_ged, delta_ig


def test_delta_ged_function():
    """Test delta_ged function."""
    # Test function exists and is callable
    assert callable(delta_ged)
    
    # Create mock graphs
    g1 = Mock()
    g1.num_nodes = 5
    g1.edge_index = Mock()
    g1.edge_index.size = Mock(return_value=(2, 10))
    
    g2 = Mock()
    g2.num_nodes = 6
    g2.edge_index = Mock()
    g2.edge_index.size = Mock(return_value=(2, 12))
    
    # Call function
    result = delta_ged(g1, g2)
    assert isinstance(result, (int, float))


def test_delta_ig_function():
    """Test delta_ig function."""
    # Test function exists and is callable
    assert callable(delta_ig)
    
    # Create mock graphs with embeddings
    g1 = Mock()
    g1.x = np.random.randn(5, 10)
    
    g2 = Mock()
    g2.x = np.random.randn(6, 10)
    
    # Call function
    result = delta_ig(g1, g2)
    assert isinstance(result, (int, float))


def test_metrics_module_imports():
    """Test metrics module imports."""
    import insightspike.metrics as metrics
    
    # Should have delta calculation functions
    assert hasattr(metrics, 'delta_ged')
    assert hasattr(metrics, 'delta_ig')