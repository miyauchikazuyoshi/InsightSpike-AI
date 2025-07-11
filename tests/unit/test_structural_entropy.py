"""
Test Structural Entropy Measures
================================

Tests for graph structural entropy calculations.
"""

import pytest
import numpy as np
import networkx as nx

from insightspike.algorithms.structural_entropy import (
    degree_distribution_entropy,
    von_neumann_entropy,
    structural_entropy,
    clustering_coefficient_entropy,
    path_length_entropy
)


class TestDegreeDistributionEntropy:
    """Test degree distribution entropy calculations."""
    
    def test_regular_graph_low_entropy(self):
        """Test that regular graphs have low entropy."""
        # Create a 4-regular graph (all nodes have degree 4)
        G = nx.circulant_graph(10, [1, 2])
        
        entropy = degree_distribution_entropy(G)
        
        # Should be 0 since all degrees are the same
        assert entropy == 0.0
        
    def test_star_graph_medium_entropy(self):
        """Test that star graphs have medium entropy."""
        # Star graph: one hub with high degree, others with degree 1
        G = nx.star_graph(10)
        
        entropy = degree_distribution_entropy(G)
        
        # Should have some entropy due to two different degrees
        assert 0 < entropy < 1.0
        
    def test_random_graph_high_entropy(self):
        """Test that random graphs have higher entropy."""
        # Random graph with varied degree distribution
        G = nx.erdos_renyi_graph(20, 0.3, seed=42)
        
        entropy = degree_distribution_entropy(G)
        
        # Should have higher entropy due to varied degrees
        assert entropy > 0.5
        
    def test_empty_graph(self):
        """Test empty graph returns zero entropy."""
        G = nx.Graph()
        assert degree_distribution_entropy(G) == 0.0
        
    def test_single_node_graph(self):
        """Test single node graph returns zero entropy."""
        G = nx.Graph()
        G.add_node(1)
        assert degree_distribution_entropy(G) == 0.0


class TestVonNeumannEntropy:
    """Test Von Neumann entropy calculations."""
    
    def test_complete_graph(self):
        """Test Von Neumann entropy for complete graph."""
        G = nx.complete_graph(5)
        
        entropy = von_neumann_entropy(G)
        
        # Complete graphs should have specific entropy patterns
        assert entropy > 0
        assert np.isfinite(entropy)
        
    def test_path_graph(self):
        """Test Von Neumann entropy for path graph."""
        G = nx.path_graph(10)
        
        entropy = von_neumann_entropy(G)
        
        # Path graphs have different spectral properties
        assert entropy > 0
        assert np.isfinite(entropy)
        
    def test_normalized_vs_unnormalized(self):
        """Test difference between normalized and unnormalized Laplacian."""
        G = nx.karate_club_graph()
        
        entropy_norm = von_neumann_entropy(G, normalized=True)
        entropy_unnorm = von_neumann_entropy(G, normalized=False)
        
        # Both should be positive and finite
        assert entropy_norm > 0
        assert entropy_unnorm > 0
        
        # But they should be different
        assert abs(entropy_norm - entropy_unnorm) > 0.01


class TestClusteringCoefficientEntropy:
    """Test clustering coefficient entropy."""
    
    def test_triangle_graph(self):
        """Test graph with many triangles has specific entropy."""
        # Create a graph with many triangles
        G = nx.Graph()
        # Add triangles
        for i in range(0, 9, 3):
            G.add_edges_from([(i, i+1), (i+1, i+2), (i+2, i)])
        
        entropy = clustering_coefficient_entropy(G)
        
        # All nodes have same clustering coefficient (1.0 for triangles)
        # So entropy should be low
        assert entropy < 0.5
        
    def test_tree_graph(self):
        """Test tree has zero clustering coefficient entropy."""
        G = nx.balanced_tree(2, 3)
        
        entropy = clustering_coefficient_entropy(G)
        
        # Trees have no triangles, all clustering coefficients are 0
        assert entropy == 0.0
        
    def test_mixed_clustering(self):
        """Test graph with varied clustering has higher entropy."""
        # Create a graph with some clusters and some sparse parts
        G = nx.Graph()
        # Dense cluster
        G.add_edges_from([(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)])
        # Sparse chain
        G.add_edges_from([(3, 4), (4, 5), (5, 6), (6, 7)])
        
        entropy = clustering_coefficient_entropy(G)
        
        # Should have medium entropy due to varied clustering
        assert 0.5 < entropy < 2.0


class TestPathLengthEntropy:
    """Test path length entropy calculations."""
    
    def test_chain_graph(self):
        """Test chain graph has specific path length distribution."""
        G = nx.path_graph(5)
        
        entropy = path_length_entropy(G)
        
        # Path graph has varied distances
        assert entropy > 0
        
    def test_complete_graph_uniform(self):
        """Test complete graph has low path length entropy."""
        G = nx.complete_graph(6)
        
        entropy = path_length_entropy(G)
        
        # All non-identical pairs have distance 1
        assert entropy == 0.0
        
    def test_disconnected_graph(self):
        """Test disconnected graph uses largest component."""
        G = nx.Graph()
        # Component 1
        G.add_edges_from([(0, 1), (1, 2)])
        # Component 2 (larger)
        G.add_edges_from([(3, 4), (4, 5), (5, 6), (6, 3)])
        
        entropy = path_length_entropy(G)
        
        # Should calculate on larger component
        assert entropy > 0


class TestStructuralEntropy:
    """Test combined structural entropy measures."""
    
    def test_combined_measures(self):
        """Test that combined measure aggregates properly."""
        G = nx.karate_club_graph()
        
        measures = structural_entropy(G)
        
        # Should have all measures
        assert "degree_entropy" in measures
        assert "clustering_entropy" in measures
        assert "path_length_entropy" in measures
        assert "combined" in measures
        
        # All should be non-negative
        for key, value in measures.items():
            assert value >= 0
            
        # Combined should be weighted average
        assert measures["combined"] > 0
        
    def test_custom_weights(self):
        """Test custom weights for combining measures."""
        G = nx.karate_club_graph()
        
        # Give all weight to degree entropy
        weights = {"degree": 1.0, "clustering": 0.0, "path_length": 0.0}
        measures = structural_entropy(G, weights)
        
        # Combined should equal degree entropy
        assert abs(measures["combined"] - measures["degree_entropy"]) < 1e-10
        

def test_pytorch_geometric_compatibility():
    """Test that functions work with PyTorch Geometric Data objects."""
    try:
        from torch_geometric.data import Data
        import torch
        
        # Create a simple PyG graph
        edge_index = torch.tensor([[0, 1, 1, 2],
                                  [1, 0, 2, 1]], dtype=getattr(torch, 'long', int))
        x = torch.randn(3, 16)
        data = Data(x=x, edge_index=edge_index)
        
        # Test all functions
        assert degree_distribution_entropy(data) >= 0
        assert clustering_coefficient_entropy(data) >= 0
        
        measures = structural_entropy(data)
        assert all(v >= 0 for v in measures.values())
        
    except (ImportError, TypeError) as e:
        pytest.skip(f"PyTorch Geometric not available or mocked: {e}")