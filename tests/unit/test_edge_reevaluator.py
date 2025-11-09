"""
Unit tests for the EdgeReevaluator module.
"""

import pytest
import numpy as np
import torch
import pytest
pytest.importorskip("torch_geometric")
from torch_geometric.data import Data

from insightspike.graph.edge_reevaluator import EdgeReevaluator


class TestEdgeReevaluator:
    """Test the EdgeReevaluator module."""
    
    def test_initialization(self):
        """Test EdgeReevaluator initialization."""
        er = EdgeReevaluator(
            similarity_threshold=0.6,
            new_edge_threshold=0.75,
            max_new_edges_per_node=3,
            edge_decay_factor=0.85
        )
        
        assert er.similarity_threshold == 0.6
        assert er.new_edge_threshold == 0.75
        assert er.max_new_edges_per_node == 3
        assert er.edge_decay_factor == 0.85
    
    def test_edge_retention(self):
        """Test that high-similarity edges are retained."""
        # Create a simple graph
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        node_features = torch.eye(2) * 10  # Orthogonal features
        graph = Data(x=node_features, edge_index=edge_index)
        
        # Make nodes more similar after "message passing"
        updated_representations = {
            0: np.array([1.0, 0.5]),
            1: np.array([0.9, 0.6])
        }
        
        query_vector = np.array([1.0, 0.0])
        
        er = EdgeReevaluator(similarity_threshold=0.8)
        new_graph = er.reevaluate(graph, updated_representations, query_vector)
        
        # Edge should be retained due to high similarity
        assert new_graph.edge_index.shape[1] >= 2
    
    def test_edge_removal(self):
        """Test that low-similarity edges are removed."""
        # Create a graph with dissimilar nodes
        edge_index = torch.tensor([[0, 1, 1, 2],
                                   [1, 0, 2, 1]], dtype=torch.long)
        node_features = torch.randn(3, 64)
        graph = Data(x=node_features, edge_index=edge_index)
        
        # Make nodes very dissimilar after "message passing"
        updated_representations = {
            0: np.random.randn(64),
            1: np.random.randn(64) * 10,  # Very different scale
            2: np.random.randn(64) * 0.1   # Very different scale
        }
        
        query_vector = np.random.randn(64)
        
        # High threshold should remove most edges
        er = EdgeReevaluator(similarity_threshold=0.95)
        new_graph = er.reevaluate(graph, updated_representations, query_vector)
        
        # Should have fewer edges
        assert new_graph.edge_index.shape[1] < graph.edge_index.shape[1]
    
    def test_new_edge_discovery(self):
        """Test discovery of new edges."""
        # Start with disconnected nodes
        edge_index = torch.tensor([[], []], dtype=torch.long)
        node_features = torch.randn(3, 32)
        graph = Data(x=node_features, edge_index=edge_index, num_nodes=3)
        
        # Make nodes similar after "message passing"
        base_vec = np.random.randn(32)
        updated_representations = {
            0: base_vec + np.random.randn(32) * 0.01,
            1: base_vec + np.random.randn(32) * 0.01,
            2: base_vec + np.random.randn(32) * 0.01
        }
        
        query_vector = base_vec
        
        er = EdgeReevaluator(
            similarity_threshold=0.5,
            new_edge_threshold=0.7,
            max_new_edges_per_node=2
        )
        new_graph = er.reevaluate(graph, updated_representations, query_vector, 
                                  return_edge_scores=True)
        
        # Should discover new edges
        assert new_graph.edge_index.shape[1] > 0
        assert hasattr(new_graph, 'edge_info')
        
        # Check that some edges are marked as new
        new_edges = [e for e in new_graph.edge_info if e['type'] == 'new']
        assert len(new_edges) > 0
    
    def test_max_new_edges_limit(self):
        """Test that max_new_edges_per_node limit is respected."""
        # Start with no edges
        edge_index = torch.tensor([[], []], dtype=torch.long)
        graph = Data(edge_index=edge_index, num_nodes=5)
        
        # All nodes very similar
        similar_vec = np.ones(16)
        updated_representations = {
            i: similar_vec + np.random.randn(16) * 0.001
            for i in range(5)
        }
        
        query_vector = similar_vec
        
        # Limit to 2 new edges per node
        er = EdgeReevaluator(
            new_edge_threshold=0.5,
            max_new_edges_per_node=2
        )
        new_graph = er.reevaluate(graph, updated_representations, query_vector)
        
        # Count edges per node
        edge_counts = {i: 0 for i in range(5)}
        for src, dst in new_graph.edge_index.t().numpy():
            edge_counts[src] += 1
        
        # No node should have more than 2 edges
        assert all(count <= 2 for count in edge_counts.values())
    
    def test_query_relevance_boost(self):
        """Test that query relevance boosts edge weights."""
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        graph = Data(edge_index=edge_index)
        
        # Node 0 similar to query, node 1 not
        query_vector = np.array([1.0, 0.0, 0.0, 0.0])
        updated_representations = {
            0: np.array([0.9, 0.1, 0.0, 0.0]),  # Similar to query
            1: np.array([0.0, 0.0, 0.9, 0.1])   # Dissimilar to query
        }
        
        er = EdgeReevaluator()
        new_graph = er.reevaluate(graph, updated_representations, query_vector,
                                  return_edge_scores=True)
        
        # Edge should still exist due to query relevance boost
        assert new_graph.edge_index.shape[1] >= 2
        assert hasattr(new_graph, 'query_relevance')
        
        # Node 0 should have higher query relevance
        assert new_graph.query_relevance[0] > new_graph.query_relevance[1]
    
    def test_edge_weight_blending(self):
        """Test edge weight blending with decay factor."""
        # Graph with edge weights
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        edge_attr = torch.tensor([[0.8], [0.8]], dtype=torch.float)
        graph = Data(edge_index=edge_index, edge_attr=edge_attr)
        
        # Similar nodes
        updated_representations = {
            0: np.ones(8),
            1: np.ones(8) * 0.9
        }
        query_vector = np.ones(8)
        
        er = EdgeReevaluator(edge_decay_factor=0.7)
        new_graph = er.reevaluate(graph, updated_representations, query_vector)
        
        # Check that edge weights are blended
        assert hasattr(new_graph, 'edge_attr')
        new_weight = new_graph.edge_attr[0].item()
        
        # Weight should be blend of old (0.8) and new similarity
        assert 0.7 < new_weight < 0.95
    
    def test_edge_statistics(self):
        """Test edge statistics calculation."""
        # Original graph
        edge_index_orig = torch.tensor([[0, 1, 1, 2],
                                        [1, 0, 2, 1]], dtype=torch.long)
        graph_orig = Data(edge_index=edge_index_orig)
        
        # Re-evaluated graph with changes
        edge_index_new = torch.tensor([[0, 1, 2, 3],
                                       [1, 0, 3, 2]], dtype=torch.long)
        graph_new = Data(edge_index=edge_index_new)
        graph_new.edge_info = [
            {'type': 'existing'},
            {'type': 'existing'},
            {'type': 'new'},
            {'type': 'new'}
        ]
        
        er = EdgeReevaluator()
        stats = er.get_edge_statistics(graph_orig, graph_new)
        
        assert stats['original_edges'] == 4
        assert stats['reevaluated_edges'] == 4
        assert stats['edges_removed'] == 2  # Edge (1,2) and (2,1) removed
        assert stats['edges_added'] == 2    # Edge (2,3) and (3,2) added
        assert stats['discovered_edges'] == 2
        assert 0 <= stats['edge_change_ratio'] <= 1
