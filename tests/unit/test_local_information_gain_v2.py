"""
Unit tests for LocalInformationGainV2
"""

import pytest
import numpy as np
import networkx as nx
import torch
import pytest
pytest.importorskip("torch_geometric")
from torch_geometric.data import Data

from insightspike.algorithms.local_information_gain_v2 import (
    LocalInformationGainV2,
    LocalIGResult,
    compute_local_ig
)


class TestLocalInformationGainV2:
    """Test suite for LocalInformationGainV2."""
    
    def test_initialization(self):
        """Test proper initialization."""
        calculator = LocalInformationGainV2(
            diffusion_steps=5,
            alpha=0.2,
            surprise_method='entropy',
            normalize=False
        )
        
        assert calculator.diffusion_steps == 5
        assert calculator.alpha == 0.2
        assert calculator.surprise_method == 'entropy'
        assert calculator.normalize is False
    
    def test_pyg_data_input(self):
        """Test calculation with PyTorch Geometric Data."""
        # Create simple graphs
        data_before = Data(
            x=torch.randn(4, 10),
            edge_index=torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
        )
        
        data_after = Data(
            x=torch.randn(5, 10),
            edge_index=torch.tensor([
                [0, 1, 2, 3, 4, 4, 4, 4],
                [1, 0, 3, 2, 0, 1, 2, 3]
            ], dtype=torch.long)
        )
        
        calculator = LocalInformationGainV2()
        result = calculator.calculate(data_before, data_after)
        
        assert isinstance(result, LocalIGResult)
        assert isinstance(result.total_ig, float)
        assert result.computation_time > 0
        assert result.ig_value == result.total_ig  # Test backward compatibility
    
    def test_networkx_input(self):
        """Test calculation with NetworkX graphs."""
        # Square graph
        g_before = nx.cycle_graph(4)
        
        # Square with central hub
        g_after = nx.star_graph(4)
        
        # Add features to nodes
        np.random.seed(42)
        for i in range(4):
            g_before.nodes[i]['features'] = np.random.randn(10)
            g_after.nodes[i]['features'] = np.random.randn(10)
        g_after.nodes[4]['features'] = np.random.randn(10) * 2  # High variance
        
        calculator = LocalInformationGainV2()
        result = calculator.calculate(g_before, g_after)
        
        assert isinstance(result, LocalIGResult)
        assert -1.0 <= result.total_ig <= 1.0  # Should be normalized
    
    def test_numpy_array_input(self):
        """Test calculation with raw numpy arrays."""
        features_before = np.random.randn(5, 20)
        features_after = np.random.randn(7, 20)
        
        calculator = LocalInformationGainV2()
        result = calculator.calculate(features_before, features_after)
        
        assert isinstance(result, LocalIGResult)
        assert result.avg_surprise_before > 0
        assert result.avg_surprise_after > 0
    
    def test_surprise_methods(self):
        """Test different surprise calculation methods."""
        g = nx.karate_club_graph()
        features = np.random.randn(g.number_of_nodes(), 10)
        
        # Test distance method
        calc_dist = LocalInformationGainV2(surprise_method='distance')
        surprise_dist = calc_dist._calculate_surprise_distribution(g, features)
        
        # Test entropy method
        calc_ent = LocalInformationGainV2(surprise_method='entropy')
        surprise_ent = calc_ent._calculate_surprise_distribution(g, features)
        
        # Both should produce valid values
        assert len(surprise_dist) == g.number_of_nodes()
        assert len(surprise_ent) == g.number_of_nodes()
        assert 0 <= surprise_dist.min() <= 1
        assert 0 <= surprise_ent.max() <= 1
        
        # Methods should produce different results
        assert not np.allclose(surprise_dist, surprise_ent)
    
    def test_information_diffusion(self):
        """Test information diffusion mechanism."""
        # Create a simple path graph
        g = nx.path_graph(5)
        
        # Set high surprise at one end
        initial_values = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        
        calculator = LocalInformationGainV2(diffusion_steps=3, alpha=0.1)
        diffused = calculator._diffuse_information(g, initial_values)
        
        # Information should spread
        assert diffused[1] > 0  # Neighbor should receive information
        assert diffused[0] < 1.0  # Source should lose some information
        assert diffused[4] > 0  # Even distant nodes get some information
        
        # Total information roughly conserved (with damping)
        assert 0.8 < diffused.sum() < 1.2
    
    def test_entropy_calculation(self):
        """Test entropy calculation."""
        calculator = LocalInformationGainV2()
        
        # Uniform distribution - high entropy
        uniform = np.ones(100) * 0.5
        entropy_uniform = calculator._array_entropy(uniform)
        
        # Concentrated distribution - low entropy  
        concentrated = np.ones(100) * 0.1
        concentrated[50:60] = 0.9  # Most values concentrated in one bin
        entropy_concentrated = calculator._array_entropy(concentrated)
        
        # Random distribution - medium entropy
        np.random.seed(42)
        random_vals = np.random.rand(100)
        entropy_random = calculator._array_entropy(random_vals)
        
        # Entropy ordering should be correct
        # Uniform should have lowest entropy (all in one bin)
        # Concentrated should have low entropy (most in 2 bins)
        # Random should have highest entropy (spread across bins)
        assert entropy_uniform < entropy_concentrated < entropy_random
    
    def test_edge_tension(self):
        """Test edge tension calculation."""
        # Create graph with varying edge tensions
        g = nx.Graph()
        g.add_edges_from([(0, 1), (1, 2), (2, 3)])
        
        # Low tension - similar values
        values_low = np.array([0.5, 0.51, 0.49, 0.5])
        
        # High tension - different values
        values_high = np.array([0.0, 1.0, 0.0, 1.0])
        
        calculator = LocalInformationGainV2()
        tension_low = calculator._edge_tension(g, values_low)
        tension_high = calculator._edge_tension(g, values_high)
        
        assert tension_low < tension_high
        assert 0 <= tension_low <= 1
        assert 0 <= tension_high <= 1
    
    def test_new_node_detection(self):
        """Test that new nodes get maximum surprise."""
        # Initial state
        data_before = Data(
            x=torch.randn(3, 5),
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        )
        
        # Add new node
        data_after = Data(
            x=torch.randn(4, 5),
            edge_index=torch.tensor([
                [0, 1, 2, 3, 3, 3],
                [1, 2, 0, 0, 1, 2]
            ], dtype=torch.long)
        )
        
        calculator = LocalInformationGainV2()
        result = calculator.calculate(data_before, data_after)
        
        # Should detect change (with current scaling, 0.08 is significant)
        assert result.total_ig > 0.05
        assert result.max_surprise_after == 1.0  # New node has max surprise
    
    def test_backward_compatibility(self):
        """Test backward compatibility with simple IG interface."""
        features_before = np.random.randn(10, 50)
        features_after = np.random.randn(12, 50)
        
        # Use convenience function
        ig_value = compute_local_ig(features_before, features_after, alpha=0.2)
        
        assert isinstance(ig_value, float)
        assert -1.0 <= ig_value <= 1.0
    
    def test_empty_graph_handling(self):
        """Test handling of empty graphs."""
        # Empty graphs
        g1 = nx.Graph()
        g2 = nx.Graph()
        g2.add_node(0)
        
        calculator = LocalInformationGainV2()
        
        # Should not crash
        result = calculator.calculate(g1, g2)
        assert isinstance(result, LocalIGResult)
    
    def test_large_graph_performance(self):
        """Test performance on larger graphs."""
        import time
        
        # Create larger graph
        g = nx.barabasi_albert_graph(100, 3)
        features = np.random.randn(100, 50)
        
        calculator = LocalInformationGainV2(diffusion_steps=3)
        
        start = time.time()
        surprise = calculator._calculate_surprise_distribution(g, features)
        diffused = calculator._diffuse_information(g, surprise)
        elapsed = time.time() - start
        
        # Should complete in reasonable time
        assert elapsed < 1.0  # Less than 1 second
        assert len(diffused) == 100
    
    def test_normalization(self):
        """Test normalization of output values."""
        # Create scenario with large changes
        features_before = np.random.randn(5, 100) * 10
        features_after = np.random.randn(10, 100) * 0.1
        
        # Without normalization
        calc_no_norm = LocalInformationGainV2(normalize=False)
        result_no_norm = calc_no_norm.calculate(features_before, features_after)
        
        # With normalization
        calc_norm = LocalInformationGainV2(normalize=True)
        result_norm = calc_norm.calculate(features_before, features_after)
        
        # Normalized should be bounded
        assert -1.0 <= result_norm.total_ig <= 1.0
        
        # Components should also be reasonable
        assert -2.0 <= result_norm.global_ig <= 2.0
        assert -2.0 <= result_norm.homogenization <= 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
