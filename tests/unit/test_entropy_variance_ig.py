"""
Unit tests for Entropy Variance IG (legacy path).

Skip in cloud-safe runs because the module was integrated into core.
"""

import pytest
pytest.skip(
    "Legacy EntropyVarianceIG module was integrated; use core.metrics/GeDIGCore instead.",
    allow_module_level=True,
)
import numpy as np
import networkx as nx
import torch
import pytest
pytest.importorskip("torch_geometric")
from torch_geometric.data import Data

from insightspike.algorithms.entropy_variance_ig import (
    EntropyVarianceIG,
    RobustEntropyVarianceIG,
    calculate_entropy_variance_ig
)


class TestEntropyVarianceIG:
    """Test suite for EntropyVarianceIG."""
    
    def test_initialization(self):
        """Test proper initialization."""
        calculator = EntropyVarianceIG(
            n_bins=30,
            adaptive_bins=False,
            include_self=False,
            normalize=False
        )
        
        assert calculator.n_bins == 30
        assert calculator.adaptive_bins is False
        assert calculator.include_self is False
        assert calculator.normalize is False
    
    def test_simple_variance_reduction(self):
        """Test that uniform distribution reduces variance."""
        # Create a simple graph
        graph = nx.path_graph(5)
        
        # Before: highly variable entropy (different values per node)
        features_before = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5],
            [0.2, 0.8]
        ])
        
        # After: more uniform (all nodes similar)
        features_after = np.array([
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5]
        ])
        
        calculator = EntropyVarianceIG()
        result = calculator.calculate(graph, features_before, features_after)
        
        # Variance should decrease (positive IG)
        assert result.ig_value > 0
        assert result.variance_before > result.variance_after
    
    def test_networkx_input(self):
        """Test with NetworkX graph input."""
        # Create a star graph
        graph = nx.star_graph(4)
        
        # Random features
        np.random.seed(42)
        features_before = np.random.rand(5, 10)
        features_after = np.random.rand(5, 10)
        
        calculator = EntropyVarianceIG()
        result = calculator.calculate(graph, features_before, features_after)
        
        assert isinstance(result.ig_value, float)
        assert len(result.local_entropies_before) == 5
        assert len(result.local_entropies_after) == 5
    
    def test_pyg_data_input(self):
        """Test with PyTorch Geometric Data input."""
        # Create PyG data
        edge_index = torch.tensor([[0, 1, 2, 3, 4, 4, 4],
                                  [1, 2, 3, 4, 0, 1, 2]], dtype=torch.long)
        x = torch.randn(5, 10)
        
        data = Data(x=x, edge_index=edge_index)
        
        # Features
        features_before = np.random.rand(5, 10)
        features_after = np.random.rand(5, 10)
        
        calculator = EntropyVarianceIG()
        result = calculator.calculate(data, features_before, features_after)
        
        assert isinstance(result.ig_value, float)
        assert result.computation_time > 0
    
    def test_isolated_nodes(self):
        """Test handling of isolated nodes."""
        # Graph with isolated node
        graph = nx.Graph()
        graph.add_nodes_from([0, 1, 2, 3])
        graph.add_edges_from([(0, 1), (1, 2)])  # Node 3 is isolated
        
        features = np.random.rand(4, 5)
        
        calculator = EntropyVarianceIG(include_self=False)
        result = calculator.calculate(graph, features, features)
        
        # Should not crash
        assert len(result.local_entropies_before) == 4
        # Isolated node should have zero entropy (no neighbors)
        assert result.local_entropies_before[3] == 0.0
    
    def test_adaptive_bins(self):
        """Test adaptive bin selection."""
        graph = nx.complete_graph(10)
        
        # Large feature set
        features = np.random.rand(10, 100)
        
        # With adaptive bins
        calc_adaptive = EntropyVarianceIG(adaptive_bins=True)
        result_adaptive = calc_adaptive.calculate(graph, features, features)
        
        # Without adaptive bins
        calc_fixed = EntropyVarianceIG(adaptive_bins=False, n_bins=5)
        result_fixed = calc_fixed.calculate(graph, features, features)
        
        # Results should be different due to different binning
        assert abs(result_adaptive.mean_entropy_before - result_fixed.mean_entropy_before) > 1e-6
    
    def test_normalization(self):
        """Test entropy normalization."""
        graph = nx.cycle_graph(4)
        features = np.random.rand(4, 10)
        
        # With normalization
        calc_norm = EntropyVarianceIG(normalize=True)
        result_norm = calc_norm.calculate(graph, features, features)
        
        # Without normalization  
        calc_no_norm = EntropyVarianceIG(normalize=False)
        result_no_norm = calc_no_norm.calculate(graph, features, features)
        
        # Normalized entropy should be in [0, 1]
        assert all(0 <= e <= 1 for e in result_norm.local_entropies_before)
        
        # Non-normalized can exceed 1
        # (This might not always be true, depends on data)
        assert result_norm.mean_entropy_before != result_no_norm.mean_entropy_before
    
    def test_feature_padding(self):
        """Test automatic feature padding."""
        graph = nx.path_graph(5)
        
        # Fewer features than nodes
        features_small = np.random.rand(3, 10)
        features_full = np.random.rand(5, 10)
        
        calculator = EntropyVarianceIG()
        result = calculator.calculate(graph, features_small, features_full)
        
        # Should handle size mismatch
        assert len(result.local_entropies_before) == 5
        assert len(result.local_entropies_after) == 5
    
    def test_convenience_function(self):
        """Test convenience function."""
        graph = nx.karate_club_graph()
        n_nodes = graph.number_of_nodes()
        
        features_before = np.random.rand(n_nodes, 20)
        features_after = np.random.rand(n_nodes, 20)
        
        ig_value = calculate_entropy_variance_ig(
            graph, features_before, features_after,
            n_bins=15, normalize=True
        )
        
        assert isinstance(ig_value, float)
        assert -1.0 <= ig_value <= 1.0  # Should be bounded due to normalization


class TestRobustEntropyVarianceIG:
    """Test suite for RobustEntropyVarianceIG."""
    
    def test_multi_resolution(self):
        """Test multi-resolution entropy calculation."""
        graph = nx.complete_graph(6)
        features = np.random.rand(6, 50)
        
        # Single resolution
        calc_single = EntropyVarianceIG(n_bins=10)
        result_single = calc_single.calculate(graph, features, features)
        
        # Multi resolution
        calc_multi = RobustEntropyVarianceIG(resolutions=[5, 10, 20])
        result_multi = calc_multi.calculate(graph, features, features)
        
        # Results should be similar but not identical
        assert abs(result_single.mean_entropy_before - result_multi.mean_entropy_before) < 0.5
        assert result_single.mean_entropy_before != result_multi.mean_entropy_before
    
    def test_automatic_resolutions(self):
        """Test automatic resolution selection."""
        graph = nx.cycle_graph(8)
        features = np.random.rand(8, 30)
        
        calculator = RobustEntropyVarianceIG(resolutions=None)
        result = calculator.calculate(graph, features, features)
        
        # Should work with automatic resolutions
        assert isinstance(result.ig_value, float)
        assert len(result.local_entropies_before) == 8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
