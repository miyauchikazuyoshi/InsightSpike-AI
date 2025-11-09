"""
Regression tests for core modules
"""

import pytest
import numpy as np
import torch
import pytest
pytest.importorskip("torch_geometric")
from torch_geometric.data import Data

from insightspike.core.vector_integrator import VectorIntegrator
from insightspike.graph.message_passing import MessagePassing
from insightspike.graph.edge_reevaluator import EdgeReevaluator


class TestVectorIntegratorRegression:
    """Regression tests for VectorIntegrator"""
    
    def test_backward_compatibility(self):
        """Test that VectorIntegrator maintains backward compatibility"""
        vi = VectorIntegrator()
        
        # Old style usage (should still work)
        vectors = [np.array([1, 0, 0]), np.array([0, 1, 0])]
        result = vi.integrate_vectors(vectors)
        
        assert result is not None
        assert result.shape == (3,)
        assert np.allclose(np.linalg.norm(result), 1.0)  # Should be normalized
    
    def test_insight_vector_with_without_query(self):
        """Test insight vector creation with and without query"""
        vi = VectorIntegrator()
        
        embeddings = [
            np.random.randn(128) for _ in range(5)
        ]
        
        # Without query (should work)
        result_no_query = vi.create_insight_vector(embeddings, None)
        assert result_no_query is not None
        
        # With query (should work differently)
        query = np.random.randn(128)
        result_with_query = vi.create_insight_vector(embeddings, query)
        assert result_with_query is not None
        
        # Results should be different
        assert not np.allclose(result_no_query, result_with_query)
    
    def test_all_integration_types(self):
        """Test all predefined integration types"""
        vi = VectorIntegrator()
        types = ["insight", "episode_branching", "message_passing", "context_merging"]
        
        vectors = [np.random.randn(64) for _ in range(3)]
        primary = np.random.randn(64)
        
        for int_type in types:
            result = vi.integrate_vectors(
                vectors, 
                primary_vector=primary,
                integration_type=int_type
            )
            assert result is not None
            assert result.shape == (64,)


class TestMessagePassingRegression:
    """Regression tests for MessagePassing"""
    
    def test_basic_functionality(self):
        """Test basic message passing functionality"""
        mp = MessagePassing(alpha=0.3, iterations=2)
        
        # Create simple graph
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        node_features = torch.randn(2, 128)
        graph = Data(x=node_features, edge_index=edge_index)
        
        query = np.random.randn(128)
        
        # Should not fail
        result = mp.forward(graph, query)
        
        assert isinstance(result, dict)
        assert len(result) == 2
        assert all(isinstance(result[i], np.ndarray) for i in range(2))
    
    def test_empty_graph_handling(self):
        """Test handling of graphs with no edges"""
        mp = MessagePassing()
        
        # Graph with no edges
        edge_index = torch.tensor([[], []], dtype=torch.long)
        node_features = torch.randn(3, 64)
        graph = Data(x=node_features, edge_index=edge_index)
        
        query = np.random.randn(64)
        
        # Should not fail
        result = mp.forward(graph, query)
        assert len(result) == 3
    
    def test_different_aggregations(self):
        """Test different aggregation methods"""
        for agg in ["weighted_mean", "max", "mean"]:
            mp = MessagePassing(aggregation=agg)
            
            edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
            node_features = torch.randn(3, 32)
            graph = Data(x=node_features, edge_index=edge_index)
            
            query = np.random.randn(32)
            result = mp.forward(graph, query)
            
            assert len(result) == 3


class TestEdgeReevaluatorRegression:
    """Regression tests for EdgeReevaluator"""
    
    def test_basic_reevaluation(self):
        """Test basic edge re-evaluation"""
        er = EdgeReevaluator()
        
        # Original graph
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        graph = Data(edge_index=edge_index, num_nodes=2)
        
        # Updated representations
        updated_reps = {
            0: np.random.randn(64),
            1: np.random.randn(64)
        }
        query = np.random.randn(64)
        
        # Should not fail
        new_graph = er.reevaluate(graph, updated_reps, query)
        
        assert hasattr(new_graph, 'edge_index')
        assert hasattr(new_graph, 'edge_attr')
        assert new_graph.num_nodes == 2
    
    def test_new_edge_discovery(self):
        """Test that new edges can be discovered"""
        er = EdgeReevaluator(
            new_edge_threshold=0.5,
            max_new_edges_per_node=5
        )
        
        # Start with no edges
        edge_index = torch.tensor([[], []], dtype=torch.long)
        graph = Data(edge_index=edge_index, num_nodes=3)
        
        # Make nodes similar
        base_vec = np.random.randn(32)
        updated_reps = {
            0: base_vec + 0.1 * np.random.randn(32),
            1: base_vec + 0.1 * np.random.randn(32),
            2: base_vec + 0.1 * np.random.randn(32)
        }
        query = base_vec
        
        new_graph = er.reevaluate(graph, updated_reps, query, return_edge_scores=True)
        
        # Should discover some edges
        assert new_graph.edge_index.shape[1] > 0
        assert hasattr(new_graph, 'edge_info')
    
    def test_edge_statistics(self):
        """Test edge statistics calculation"""
        er = EdgeReevaluator()
        
        # Graphs with different edge counts
        graph1 = Data(edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long))
        graph2 = Data(edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long))
        
        stats = er.get_edge_statistics(graph1, graph2)
        
        assert stats['original_edges'] == 2
        assert stats['reevaluated_edges'] == 4
        assert stats['edges_added'] == 2
        assert 'edge_change_ratio' in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
