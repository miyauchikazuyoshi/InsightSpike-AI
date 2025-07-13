"""Tests for metrics module to improve coverage"""
import pytest
import numpy as np

from insightspike.metrics import delta_ged, delta_ig
from tests.helpers.test_helpers import assert_metric_in_range


def test_delta_ged_function(simple_graph, graph_pair_similar):
    """Test delta_ged function using fixtures."""
    # Test function exists and is callable
    assert callable(delta_ged)

    # Test with graph pair fixture
    g1, g2 = graph_pair_similar

    # Call function
    result = delta_ged(g1, g2)
    assert isinstance(result, (int, float))
    assert result >= 0  # GED should be non-negative

    # Test with same graph (should be 0)
    result_same = delta_ged(simple_graph, simple_graph)
    assert result_same == 0.0


def test_delta_ig_function():
    """Test delta_ig function using factory."""
    # Test function exists and is callable
    assert callable(delta_ig)

    # Use factory to create embeddings
    from tests.factories.mock_factory import embedding_factory

    vecs_old = embedding_factory.create_embedding_batch(num_samples=5, dim=10)
    vecs_new = embedding_factory.create_embedding_batch(num_samples=6, dim=10)

    # Call function
    result = delta_ig(vecs_old, vecs_new)
    assert isinstance(result, (int, float))

    # Test with same embeddings (should be close to 0)
    result_same = delta_ig(vecs_old, vecs_old)
    assert abs(result_same) < 0.1  # Should be close to 0


def test_metrics_module_imports():
    """Test metrics module imports."""
    import insightspike.metrics as metrics

    # Should have delta calculation functions
    assert hasattr(metrics, "delta_ged")
    assert hasattr(metrics, "delta_ig")


def test_delta_ged_with_different_graphs(graph_pair_different):
    """Test delta_ged with very different graphs."""
    g1, g2 = graph_pair_different

    result = delta_ged(g1, g2)
    assert isinstance(result, float)
    assert result > 0  # Should have high edit distance


def test_delta_ig_edge_cases():
    """Test delta_ig with edge cases."""
    from tests.factories.mock_factory import embedding_factory

    # Test with single sample (edge case for clustering)
    vec_single = embedding_factory.create_embedding_batch(num_samples=1, dim=10)
    vec_multi = embedding_factory.create_embedding_batch(num_samples=5, dim=10)

    result = delta_ig(vec_single, vec_multi)
    assert isinstance(result, float)

    # Test with high-dimensional embeddings
    vec_high_dim1 = embedding_factory.create_embedding_batch(num_samples=10, dim=768)
    vec_high_dim2 = embedding_factory.create_embedding_batch(num_samples=12, dim=768)

    result_high_dim = delta_ig(vec_high_dim1, vec_high_dim2)
    assert isinstance(result_high_dim, float)
