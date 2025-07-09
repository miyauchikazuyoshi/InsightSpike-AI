"""Tests for metrics module to improve coverage"""
import pytest
from unittest.mock import Mock, MagicMock
import numpy as np

from insightspike.metrics import delta_ged, delta_ig


def test_delta_ged_function():
    """Test delta_ged function."""
    # Test function exists and is callable
    assert callable(delta_ged)
    
    # Create mock graphs that behave like NetworkX graphs
    g1 = Mock()
    g1.nodes = Mock(return_value=[1, 2, 3, 4, 5])
    g1.edges = Mock(return_value=[(1, 2), (2, 3)])
    g1.__iter__ = Mock(return_value=iter([1, 2, 3, 4, 5]))
    
    g2 = Mock()
    g2.nodes = Mock(return_value=[1, 2, 3, 4, 5, 6])
    g2.edges = Mock(return_value=[(1, 2), (2, 3), (3, 4)])
    g2.__iter__ = Mock(return_value=iter([1, 2, 3, 4, 5, 6]))
    
    # Call function
    result = delta_ged(g1, g2)
    assert isinstance(result, (int, float))


def test_delta_ig_function():
    """Test delta_ig function."""
    # Test function exists and is callable
    assert callable(delta_ig)
    
    # Create numpy arrays directly as delta_ig expects
    vecs_old = np.random.randn(5, 10)
    vecs_new = np.random.randn(6, 10)
    
    # Call function
    result = delta_ig(vecs_old, vecs_new)
    assert isinstance(result, (int, float))


def test_metrics_module_imports():
    """Test metrics module imports."""
    import insightspike.metrics as metrics
    
    # Should have delta calculation functions
    assert hasattr(metrics, 'delta_ged')
    assert hasattr(metrics, 'delta_ig')