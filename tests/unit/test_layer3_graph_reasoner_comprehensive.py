"""
Comprehensive tests for Layer 3 Graph Reasoner
"""

import pytest
import numpy as np
import torch
from torch_geometric.data import Data
from unittest.mock import Mock, patch, MagicMock
import networkx as nx

from insightspike.core.layers.layer3_graph_reasoner import (
    L3GraphReasoner, ConflictScore, ConflictType
)


class TestConflictScore:
    """Test ConflictScore dataclass."""
    
    def test_conflict_score_creation(self):
        """Test creating ConflictScore instances."""
        score = ConflictScore(
            score=0.8,
            type=ConflictType.TEMPORAL,
            description="Time conflict detected"
        )
        
        assert score.score == 0.8
        assert score.type == ConflictType.TEMPORAL
        assert score.description == "Time conflict detected"
    
    def test_conflict_types(self):
        """Test all conflict types."""
        types = [
            ConflictType.SEMANTIC,
            ConflictType.TEMPORAL,
            ConflictType.CAUSAL,
            ConflictType.SPATIAL
        ]
        
        for conflict_type in types:
            score = ConflictScore(
                score=0.5,
                type=conflict_type,
                description=f"{conflict_type.value} conflict"
            )
            assert score.type == conflict_type


class TestL3GraphReasoner:
    """Test L3GraphReasoner functionality."""
    
    @pytest.fixture
    def reasoner(self):
        """Create a graph reasoner instance."""
        return L3GraphReasoner(embedding_dim=8, use_gnn=False)
    
    @pytest.fixture
    def sample_graph(self):
        """Create a sample graph."""
        x = torch.randn(5, 8)
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
        return Data(x=x, edge_index=edge_index)
    
    def test_init_without_gnn(self):
        """Test initialization without GNN."""
        reasoner = L3GraphReasoner(embedding_dim=16, use_gnn=False)
        assert reasoner.embedding_dim == 16
        assert reasoner.use_gnn is False
        assert reasoner.gnn_model is None
    
    def test_init_with_gnn(self):
        """Test initialization with GNN."""
        reasoner = L3GraphReasoner(embedding_dim=16, use_gnn=True)
        assert reasoner.embedding_dim == 16
        assert reasoner.use_gnn is True
        assert reasoner.gnn_model is not None
    
    def test_retrieve_simple(self, reasoner, sample_graph):
        """Test simple retrieval without GNN."""
        reasoner.graph = sample_graph
        query = torch.randn(8)
        
        results = reasoner.retrieve(query, k=3)
        
        assert len(results) == 3
        assert all('node_idx' in r for r in results)
        assert all('similarity' in r for r in results)
        assert all(0 <= r['node_idx'] < 5 for r in results)
    
    @patch('torch_geometric.nn.GCNConv')
    def test_retrieve_with_gnn(self, mock_gcn):
        """Test retrieval with GNN processing."""
        # Create reasoner with GNN
        reasoner = L3GraphReasoner(embedding_dim=8, use_gnn=True)
        reasoner.graph = Data(
            x=torch.randn(5, 8),
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        )
        
        # Mock GNN forward pass
        reasoner.gnn_model = Mock()
        reasoner.gnn_model.return_value = torch.randn(5, 8)
        
        query = torch.randn(8)
        results = reasoner.retrieve(query, k=2)
        
        assert len(results) == 2
        reasoner.gnn_model.assert_called_once()
    
    def test_detect_conflicts_no_conflicts(self, reasoner):
        """Test conflict detection with no conflicts."""
        node_indices = [0, 2, 4]
        conflicts = reasoner.detect_conflicts(node_indices)
        
        assert len(conflicts) == 0
    
    def test_detect_conflicts_semantic(self, reasoner):
        """Test semantic conflict detection."""
        reasoner.episodes = [
            Mock(text="The cat is black", metadata={}),
            Mock(text="The cat is white", metadata={}),
            Mock(text="Dogs are friendly", metadata={})
        ]
        
        conflicts = reasoner.detect_conflicts([0, 1])
        
        assert len(conflicts) > 0
        assert any(c.type == ConflictType.SEMANTIC for c in conflicts.values())
    
    def test_detect_conflicts_temporal(self, reasoner):
        """Test temporal conflict detection."""
        reasoner.episodes = [
            Mock(text="Event A happened first", metadata={'timestamp': 100}),
            Mock(text="Event A happened last", metadata={'timestamp': 200}),
        ]
        
        conflicts = reasoner.detect_conflicts([0, 1])
        
        assert len(conflicts) > 0
        # Should detect temporal ordering conflict
    
    def test_get_graph_context(self, reasoner, sample_graph):
        """Test getting graph context."""
        reasoner.graph = sample_graph
        reasoner.networkx_graph = nx.Graph()
        reasoner.networkx_graph.add_edges_from([(0, 1), (1, 2), (2, 3)])
        
        context = reasoner.get_graph_context([0, 2])
        
        assert 'subgraph_size' in context
        assert 'connectivity' in context
        assert 'node_degrees' in context
        assert len(context['node_degrees']) == 2
    
    def test_reason_with_memory(self, reasoner):
        """Test reasoning with memory context."""
        memory_context = {
            'episodes': [
                {'text': 'Episode 1', 'importance': 0.8},
                {'text': 'Episode 2', 'importance': 0.6}
            ],
            'query': 'Test query'
        }
        
        result = reasoner.reason(memory_context)
        
        assert 'reasoning' in result
        assert 'confidence' in result
        assert 0 <= result['confidence'] <= 1
    
    def test_update_graph(self, reasoner):
        """Test graph update."""
        new_graph = Data(
            x=torch.randn(10, 8),
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        )
        
        reasoner.update_graph(new_graph)
        
        assert reasoner.graph.x.shape[0] == 10
        assert reasoner.networkx_graph is not None
        assert reasoner.networkx_graph.number_of_nodes() == 10
    
    def test_process_with_gnn(self, reasoner):
        """Test GNN processing."""
        reasoner.use_gnn = True
        reasoner.gnn_model = Mock()
        reasoner.gnn_model.return_value = torch.randn(5, 8)
        
        graph = Data(
            x=torch.randn(5, 8),
            edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        )
        
        processed = reasoner._process_with_gnn(graph)
        
        assert processed.shape == (5, 8)
        reasoner.gnn_model.assert_called_once()
    
    def test_calculate_similarity(self, reasoner):
        """Test similarity calculation."""
        node_features = torch.randn(5, 8)
        query = torch.randn(8)
        
        similarities = reasoner._calculate_similarity(node_features, query)
        
        assert similarities.shape == (5,)
        assert all(-1 <= s <= 1 for s in similarities)
    
    def test_check_semantic_conflict(self, reasoner):
        """Test semantic conflict checking."""
        # Similar content - no conflict
        text1 = "Machine learning is a subset of AI"
        text2 = "AI includes machine learning"
        score1 = reasoner._check_semantic_conflict(text1, text2)
        assert score1 < 0.5
        
        # Contradictory content - conflict
        text3 = "The model accuracy is 95%"
        text4 = "The model accuracy is 60%"
        score2 = reasoner._check_semantic_conflict(text3, text4)
        assert score2 > 0
    
    def test_check_temporal_conflict(self, reasoner):
        """Test temporal conflict checking."""
        meta1 = {'timestamp': 100, 'text': 'Event A happened'}
        meta2 = {'timestamp': 200, 'text': 'Event A happened'}
        
        score = reasoner._check_temporal_conflict(meta1, meta2, "happened", "happened")
        assert score > 0  # Same event at different times
    
    def test_empty_graph_handling(self, reasoner):
        """Test handling of empty graph."""
        reasoner.graph = None
        query = torch.randn(8)
        
        results = reasoner.retrieve(query, k=5)
        assert len(results) == 0
    
    def test_large_k_handling(self, reasoner, sample_graph):
        """Test retrieval with k larger than graph size."""
        reasoner.graph = sample_graph  # 5 nodes
        query = torch.randn(8)
        
        results = reasoner.retrieve(query, k=10)
        assert len(results) == 5  # Should return all nodes
    
    @patch('insightspike.core.layers.layer3_graph_reasoner.logger')
    def test_error_logging(self, mock_logger, reasoner):
        """Test error logging."""
        # Cause an error in conflict detection
        reasoner.episodes = None
        
        conflicts = reasoner.detect_conflicts([0, 1])
        
        # Should log error and return empty dict
        assert conflicts == {}
        mock_logger.error.assert_called()
    
    def test_integration_with_episodes(self, reasoner):
        """Test integration with episode data."""
        # Set up episodes
        reasoner.episodes = [
            Mock(text=f"Episode {i}", embedding=torch.randn(8), metadata={'id': i})
            for i in range(5)
        ]
        
        # Create graph from episodes
        embeddings = torch.stack([e.embedding for e in reasoner.episodes])
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        reasoner.graph = Data(x=embeddings, edge_index=edge_index)
        
        # Test retrieval
        query = reasoner.episodes[0].embedding
        results = reasoner.retrieve(query, k=3)
        
        assert results[0]['node_idx'] == 0  # Should find itself first
        assert results[0]['similarity'] > 0.99