"""Tests for metrics module to improve coverage"""
import pytest
from unittest.mock import Mock, MagicMock
import numpy as np
import networkx as nx

from insightspike.metrics import delta_ged, delta_ig


def test_delta_ged_function():
    """Test delta_ged function."""
    # Test function exists and is callable
    assert callable(delta_ged)
    
    # Create actual NetworkX graphs
    g1 = nx.Graph()
    g1.add_nodes_from([1, 2, 3, 4, 5])
    g1.add_edges_from([(1, 2), (2, 3)])
    
    g2 = nx.Graph()
    g2.add_nodes_from([1, 2, 3, 4, 5, 6])
    g2.add_edges_from([(1, 2), (2, 3), (3, 4)])
    
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