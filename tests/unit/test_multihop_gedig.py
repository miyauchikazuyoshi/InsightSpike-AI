"""
Unit tests for Multi-hop geDIG (legacy path).

Cloud-safe guard: skip if torch_geometric or legacy module is unavailable.
"""

import pytest
pytest.importorskip("torch_geometric")
pytest.importorskip(
    "insightspike.algorithms.multihop_gedig",
    reason="Legacy module integrated into GeDIGCore; skip in cloud-safe runs.",
)
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data

from insightspike.algorithms.multihop_gedig import (
    MultiHopGeDIG,
    calculate_multihop_gedig
)


class TestMultiHopGeDIG:
    """Test suite for MultiHopGeDIG."""
    
    def test_initialization(self):
        """Test proper initialization."""
        calculator = MultiHopGeDIG(
            max_hops=5,
            decay_factor=0.8,
            ged_weight=0.6,
            ig_weight=0.4,
            adaptive_hops=False
        )
        
        assert calculator.max_hops == 5
        assert calculator.decay_factor == 0.8
        assert calculator.ged_weight == 0.6
        assert calculator.ig_weight == 0.4
        assert calculator.adaptive_hops is False
    
    def test_simple_hub_formation(self):
        """Test multi-hop detection of hub formation."""
        # Before: Two disconnected pairs
        g_before = nx.Graph()
        g_before.add_edges_from([(0, 1), (2, 3)])
        
        # After: Hub connects all
        g_after = nx.Graph()
        g_after.add_edges_from([(0, 1), (2, 3), (4, 0), (4, 1), (4, 2), (4, 3)])
        
        # Features
        np.random.seed(42)
        features_before = np.random.rand(4, 10)
        features_after = np.vstack([features_before, np.random.rand(1, 10)])
        
        calculator = MultiHopGeDIG(max_hops=2)
        result = calculator.calculate(
            g_before, g_after, 
            features_before, features_after,
            focal_nodes=[4]  # Focus on hub
        )
        
        # Should detect changes at different hops
        assert result.hop_results[0]['nodes_in_subgraph'] == 1  # Just hub
        assert result.hop_results[1]['nodes_in_subgraph'] == 5  # Hub + neighbors
        assert result.hop_results[2]['nodes_in_subgraph'] == 5  # All nodes
        
        # Later hops should have lower weight
        assert result.hop_results[1]['weight'] < result.hop_results[0]['weight']
    
    def test_adaptive_stopping(self):
        """Test adaptive hop termination."""
        # Simple path graph
        g = nx.path_graph(10)
        features = np.random.rand(10, 5)
        
        calculator = MultiHopGeDIG(
            max_hops=5,
            adaptive_hops=True,
            min_improvement=0.1
        )
        
        # Small change - should stop early
        g_after = g.copy()
        g_after.add_edge(0, 2)  # Small shortcut
        
        result = calculator.calculate(
            g, g_after, features, features,
            focal_nodes=[0]
        )
        
        # Should stop before max_hops
        assert len(result.hop_results) < 6
    
    def test_focal_node_identification(self):
        """Test automatic focal node identification."""
        # Before: simple graph
        g_before = nx.cycle_graph(4)
        
        # After: add high-degree node
        g_after = g_before.copy()
        g_after.add_node(4)
        for i in range(4):
            g_after.add_edge(4, i)
        
        features_before = np.random.rand(4, 10)
        features_after = np.vstack([features_before, np.random.rand(1, 10)])
        
        calculator = MultiHopGeDIG()
        result = calculator.calculate(
            g_before, g_after,
            features_before, features_after
            # No focal_nodes provided - should identify automatically
        )
        
        # Should identify node 4 as focal
        assert result.hop_results[0]['nodes_in_subgraph'] >= 1
    
    def test_pyg_data_input(self):
        """Test with PyTorch Geometric Data."""
        # Before
        edge_index_before = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        data_before = Data(x=torch.randn(3, 10), edge_index=edge_index_before)
        
        # After
        edge_index_after = torch.tensor([
            [0, 1, 3, 3, 3],
            [1, 2, 0, 1, 2]
        ], dtype=torch.long)
        data_after = Data(x=torch.randn(4, 10), edge_index=edge_index_after)
        
        features_before = np.random.rand(3, 10)
        features_after = np.random.rand(4, 10)
        
        calculator = MultiHopGeDIG(max_hops=2)
        result = calculator.calculate(
            data_before, data_after,
            features_before, features_after
        )
        
        assert isinstance(result.total_gedig, float)
        assert len(result.hop_results) > 0
    
    def test_decay_factor(self):
        """Test that decay factor properly weights hops."""
        g = nx.complete_graph(5)
        features = np.random.rand(5, 10)
        
        # High decay (distant hops matter less)
        calc_high_decay = MultiHopGeDIG(max_hops=3, decay_factor=0.5)
        result_high = calc_high_decay.calculate(g, g, features, features)
        
        # Low decay (distant hops matter more)
        calc_low_decay = MultiHopGeDIG(max_hops=3, decay_factor=0.9)
        result_low = calc_low_decay.calculate(g, g, features, features)
        
        # Verify decay is applied (check last available hop)
        max_hop_high = max(result_high.hop_results.keys())
        max_hop_low = max(result_low.hop_results.keys())
        
        # Both should have at least hop 0
        assert result_high.hop_results[0]['weight'] == 1.0
        assert result_low.hop_results[0]['weight'] == 1.0
        
        # If we have hop 1, check decay
        if max_hop_high >= 1 and max_hop_low >= 1:
            assert result_high.hop_results[1]['weight'] < result_low.hop_results[1]['weight']
    
    def test_optimal_hop_detection(self):
        """Test detection of optimal hop level."""
        # Create a graph where 2-hop reveals most structure
        g_before = nx.Graph()
        g_before.add_edges_from([(0, 1), (2, 3), (4, 5), (6, 7)])
        
        # Add bridges at 2-hop distance
        g_after = g_before.copy()
        g_after.add_edges_from([(1, 2), (3, 4), (5, 6)])
        
        features_before = np.random.rand(8, 10)
        features_after = features_before.copy()
        
        calculator = MultiHopGeDIG(max_hops=3)
        result = calculator.calculate(
            g_before, g_after,
            features_before, features_after,
            focal_nodes=[1, 3, 5]  # Bridge nodes
        )
        
        # Optimal hop should be > 0
        assert result.optimal_hop > 0
    
    def test_empty_graph_handling(self):
        """Test handling of empty graphs."""
        g_empty = nx.Graph()
        g_single = nx.Graph()
        g_single.add_node(0)
        
        features_empty = np.array([])
        features_single = np.array([[1.0]])
        
        calculator = MultiHopGeDIG()
        
        # Should not crash
        result = calculator.calculate(
            g_empty, g_single,
            features_empty, features_single
        )
        
        assert isinstance(result.total_gedig, float)
    
    def test_convenience_function(self):
        """Test convenience function."""
        g1 = nx.cycle_graph(6)
        g2 = nx.star_graph(5)
        
        features1 = np.random.rand(6, 15)
        features2 = np.random.rand(6, 15)
        
        total_gedig = calculate_multihop_gedig(
            g1, g2, features1, features2,
            max_hops=2, decay_factor=0.6
        )
        
        assert isinstance(total_gedig, float)
    
    def test_subgraph_extraction(self):
        """Test k-hop subgraph extraction."""
        # Create a larger graph
        g = nx.karate_club_graph()
        features = np.random.rand(g.number_of_nodes(), 20)
        
        calculator = MultiHopGeDIG(max_hops=3)
        
        # Extract subgraphs manually to verify
        focal = [0]  # Mr. Hi
        
        result = calculator.calculate(g, g, features, features, focal_nodes=focal)
        
        # Verify hop sizes make sense
        previous_nodes = 0
        for hop in sorted(result.hop_results.keys()):
            nodes_at_hop = result.hop_results[hop]['nodes_in_subgraph']
            # Each hop should include at least as many nodes as previous
            assert nodes_at_hop >= previous_nodes
            previous_nodes = nodes_at_hop


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
