"""
Unit tests for the MessagePassing module.
"""

import pytest
import numpy as np
import torch
import pytest
pytest.importorskip("torch_geometric")
from torch_geometric.data import Data

from insightspike.graph.message_passing import MessagePassing


class TestMessagePassing:
    """Test the MessagePassing module."""
    
    def test_initialization(self):
        """Test MessagePassing initialization."""
        mp = MessagePassing(
            alpha=0.4,
            iterations=5,
            aggregation="max",
            self_loop_weight=0.6,
            decay_factor=0.7
        )
        
        assert mp.alpha == 0.4
        assert mp.iterations == 5
        assert mp.aggregation == "max"
        assert mp.self_loop_weight == 0.6
        assert mp.decay_factor == 0.7
    
    def test_single_node_graph(self):
        """Test message passing on a single node graph."""
        # Single node with self-loop
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        node_features = torch.randn(1, 128)
        graph = Data(x=node_features, edge_index=edge_index)
        
        query_vector = np.random.randn(128)
        
        mp = MessagePassing(alpha=0.3, iterations=1)
        result = mp.forward(graph, query_vector)
        
        assert len(result) == 1
        assert result[0].shape == (128,)
    
    def test_disconnected_graph(self):
        """Test message passing on a disconnected graph."""
        # Two disconnected components
        edge_index = torch.tensor([[0, 1, 2, 3],
                                   [1, 0, 3, 2]], dtype=torch.long)
        node_features = torch.randn(4, 256)
        graph = Data(x=node_features, edge_index=edge_index)
        
        query_vector = np.random.randn(256)
        
        mp = MessagePassing(alpha=0.2, iterations=2)
        result = mp.forward(graph, query_vector)
        
        assert len(result) == 4
        assert all(result[i].shape == (256,) for i in range(4))
    
    def test_query_influence(self):
        """Test that query vector influences node representations."""
        # Simple chain graph
        edge_index = torch.tensor([[0, 1, 1, 2],
                                   [1, 0, 2, 1]], dtype=torch.long)
        node_features = torch.zeros(3, 64)  # Zero features
        graph = Data(x=node_features, edge_index=edge_index)
        
        # Non-zero query
        query_vector = np.ones(64)
        
        # High alpha for strong query influence
        mp = MessagePassing(alpha=0.8, iterations=1)
        result = mp.forward(graph, query_vector)
        
        # Check that nodes are influenced by query
        for i in range(3):
            assert not np.allclose(result[i], np.zeros(64))
            # Should have some component of the query vector
            assert np.dot(result[i], query_vector) > 0
    
    def test_aggregation_methods(self):
        """Test different aggregation methods."""
        # Triangle graph
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 0],
                                   [1, 0, 2, 1, 0, 2]], dtype=torch.long)
        node_features = torch.randn(3, 32)
        graph = Data(x=node_features, edge_index=edge_index)
        
        query_vector = np.random.randn(32)
        
        # Test weighted_mean
        mp_mean = MessagePassing(aggregation="weighted_mean", iterations=1)
        result_mean = mp_mean.forward(graph, query_vector)
        
        # Test max
        mp_max = MessagePassing(aggregation="max", iterations=1)
        result_max = mp_max.forward(graph, query_vector)
        
        # Results should be different
        assert not np.array_equal(result_mean[0], result_max[0])
    
    def test_multiple_iterations(self):
        """Test that multiple iterations propagate information."""
        # Linear chain
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
                                   [1, 0, 2, 1, 3, 2]], dtype=torch.long)
        
        # Node 0 has distinct features
        node_features = torch.zeros(4, 16)
        node_features[0] = torch.ones(16)
        graph = Data(x=node_features, edge_index=edge_index)
        
        query_vector = np.zeros(16)
        
        # Single iteration - information should not reach node 3
        mp1 = MessagePassing(alpha=0.1, iterations=1, self_loop_weight=0.0)
        result1 = mp1.forward(graph, query_vector)
        
        # Multiple iterations - information should reach node 3
        mp3 = MessagePassing(alpha=0.1, iterations=3, self_loop_weight=0.0)
        result3 = mp3.forward(graph, query_vector)
        
        # Node 3 should have more information after 3 iterations
        assert np.linalg.norm(result3[3]) > np.linalg.norm(result1[3])
    
    def test_with_custom_node_features(self):
        """Test with custom node features dictionary."""
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        graph = Data(edge_index=edge_index)
        
        # Custom features as dictionary
        node_features = {
            0: np.array([1.0, 0.0, 0.0]),
            1: np.array([0.0, 1.0, 0.0])
        }
        
        query_vector = np.array([0.0, 0.0, 1.0])
        
        mp = MessagePassing(alpha=0.5, iterations=1)
        result = mp.forward(graph, query_vector, node_features=node_features)
        
        assert len(result) == 2
        assert result[0].shape == (3,)
        assert result[1].shape == (3,)
    
    def test_attention_weights_computation(self):
        """Test attention weight computation."""
        mp = MessagePassing()
        
        query = np.array([1.0, 0.0, 0.0])
        keys = np.array([[1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0],
                         [0.0, 0.0, 1.0]])
        
        weights = mp.compute_attention_weights(query, keys)
        
        assert weights.shape == (3,)
        assert np.allclose(weights.sum(), 1.0)  # Weights sum to 1
        assert weights[0] > weights[1]  # First key most similar to query
        assert weights[0] > weights[2]
