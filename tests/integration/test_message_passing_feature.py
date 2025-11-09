"""
Integration test for question-aware message passing and query integration features.
"""

import pytest
import numpy as np
import torch
import pytest
pytest.importorskip("torch_geometric")
from torch_geometric.data import Data

from insightspike.graph.message_passing import MessagePassing
from insightspike.graph.edge_reevaluator import EdgeReevaluator
from insightspike.implementations.layers.layer3_graph_reasoner import L3GraphReasoner
from insightspike.implementations.layers.layer4_llm_interface import L4LLMInterface
from insightspike.config import load_config


class TestMessagePassingFeature:
    """Test the complete message passing feature integration."""
    
    def test_message_passing_module(self):
        """Test basic message passing functionality."""
        # Create a simple graph
        num_nodes = 5
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4],
                                   [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long)
        
        # Random node features
        node_features = torch.randn(num_nodes, 384)
        
        graph = Data(x=node_features, edge_index=edge_index)
        
        # Create query vector
        query_vector = np.random.randn(384)
        
        # Initialize message passing
        mp = MessagePassing(alpha=0.3, iterations=2)
        
        # Run message passing
        updated_representations = mp.forward(graph, query_vector)
        
        # Verify output
        assert len(updated_representations) == num_nodes
        assert all(isinstance(updated_representations[i], np.ndarray) for i in range(num_nodes))
        assert all(updated_representations[i].shape == (384,) for i in range(num_nodes))
    
    def test_edge_reevaluator(self):
        """Test edge re-evaluation after message passing."""
        # Create a simple graph
        num_nodes = 4
        edge_index = torch.tensor([[0, 1, 1, 2],
                                   [1, 0, 2, 1]], dtype=torch.long)
        
        node_features = torch.randn(num_nodes, 384)
        graph = Data(x=node_features, edge_index=edge_index)
        
        # Create updated representations (simulating after message passing)
        updated_representations = {i: np.random.randn(384) for i in range(num_nodes)}
        query_vector = np.random.randn(384)
        
        # Initialize edge re-evaluator
        er = EdgeReevaluator(
            similarity_threshold=0.3,
            new_edge_threshold=0.5,
            max_new_edges_per_node=2
        )
        
        # Re-evaluate edges
        new_graph = er.reevaluate(graph, updated_representations, query_vector, return_edge_scores=True)
        
        # Verify output
        assert hasattr(new_graph, 'edge_index')
        assert hasattr(new_graph, 'edge_attr')
        assert hasattr(new_graph, 'edge_info')
        assert new_graph.num_nodes == num_nodes
    
    def test_layer3_message_passing_integration(self):
        """Test Layer3 integration with message passing."""
        # Load config with message passing enabled
        config = {
            "graph": {
                "enable_message_passing": True,
                "message_passing": {
                    "alpha": 0.3,
                    "iterations": 2
                },
                "edge_reevaluation": {
                    "similarity_threshold": 0.5,
                    "new_edge_threshold": 0.7
                },
                "similarity_threshold": 0.6,
                "spike_ged_threshold": 0.3,
                "spike_ig_threshold": 0.7,
                "conflict_threshold": 0.5
            },
            "embedding": {
                "dimension": 384
            }
        }
        
        # Initialize Layer3
        l3 = L3GraphReasoner(config)
        l3.initialize()
        
        # Create test documents
        documents = [
            {"text": "Document 1", "embedding": np.random.randn(384)},
            {"text": "Document 2", "embedding": np.random.randn(384)},
            {"text": "Document 3", "embedding": np.random.randn(384)}
        ]
        
        # Create context with query vector
        context = {
            "query_vector": np.random.randn(384)
        }
        
        # Run analysis
        result = l3.analyze_documents(documents, context)
        
        # Verify result
        assert "graph" in result
        assert "metrics" in result
        assert "spike_detected" in result
        assert result["graph"] is not None
    
    def test_layer4_query_integration(self):
        """Test Layer4 insight vector creation with query integration."""
        # Initialize Layer4
        config = {"llm": {"provider": "mock"}}
        l4 = L4LLMInterface(config)
        
        # Create test documents with embeddings
        documents = [
            {"text": "Doc 1", "embedding": np.random.randn(384)},
            {"text": "Doc 2", "embedding": np.random.randn(384)},
            {"text": "Doc 3", "embedding": np.random.randn(384)}
        ]
        
        # Create query vector
        query_vector = np.random.randn(384)
        
        # Create insight vector
        insight_vector = l4._create_insight_vector(documents, query_vector)
        
        # Verify
        assert insight_vector is not None
        assert insight_vector.shape == (384,)
        
        # Test without query vector (fallback)
        insight_vector_no_query = l4._create_insight_vector(documents, None)
        assert insight_vector_no_query is not None
        assert insight_vector_no_query.shape == (384,)
        
        # Vectors should be different when query is included
        assert not np.array_equal(insight_vector, insight_vector_no_query)
    
    def test_edge_statistics(self):
        """Test edge statistics calculation."""
        # Create original graph
        edge_index_orig = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        graph_orig = Data(edge_index=edge_index_orig)
        
        # Create re-evaluated graph with more edges
        edge_index_new = torch.tensor([[0, 1, 1, 2, 0, 2],
                                       [1, 0, 2, 1, 2, 0]], dtype=torch.long)
        graph_new = Data(edge_index=edge_index_new)
        graph_new.edge_info = [
            {'type': 'existing'},
            {'type': 'existing'},
            {'type': 'new'},
            {'type': 'new'},
            {'type': 'new'},
            {'type': 'new'}
        ]
        
        # Calculate statistics
        er = EdgeReevaluator()
        stats = er.get_edge_statistics(graph_orig, graph_new)
        
        # Verify statistics
        assert stats['original_edges'] == 2
        assert stats['reevaluated_edges'] == 6
        assert stats['edges_added'] == 4
        assert stats['discovered_edges'] == 4
