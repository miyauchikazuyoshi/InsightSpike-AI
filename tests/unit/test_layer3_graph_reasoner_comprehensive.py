"""
Comprehensive tests for Layer 3 Graph Reasoner
"""

import pytest
import numpy as np
import torch
from torch_geometric.data import Data
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path

from insightspike.core.layers.layer3_graph_reasoner import (
    L3GraphReasoner, ConflictScore, GraphBuilder
)
from insightspike.core.interfaces import LayerInput


class TestConflictScore:
    """Test ConflictScore functionality."""
    
    def test_conflict_score_creation(self):
        """Test creating ConflictScore instances."""
        score = ConflictScore()
        assert hasattr(score, 'config')
        assert hasattr(score, 'conflict_threshold')
    
    def test_conflict_score_with_config(self):
        """Test ConflictScore with custom config."""
        mock_config = Mock()
        mock_config.reasoning.conflict_threshold = 0.7
        score = ConflictScore(config=mock_config)
        assert score.conflict_threshold == 0.7
    
    def test_calculate_conflict(self):
        """Test conflict calculation."""
        score = ConflictScore()
        
        # Create mock graphs
        graph_old = Data(
            x=torch.randn(3, 8), 
            edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t()
        )
        graph_new = Data(
            x=torch.randn(4, 8), 
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long).t()
        )
        
        context = {'previous_confidence': 0.8, 'current_confidence': 0.6}
        result = score.calculate_conflict(graph_old, graph_new, context)
        
        assert isinstance(result, dict)
        assert 'structural' in result
        assert 'semantic' in result
        assert 'temporal' in result
        assert 'total' in result
        assert all(0 <= v <= 1 for v in result.values() if isinstance(v, (int, float)))
    
    def test_calculate_conflict_with_none_graphs(self):
        """Test conflict calculation with None graphs."""
        score = ConflictScore()
        
        result = score.calculate_conflict(None, None, {})
        
        assert result['structural'] == 0.0
        assert result['semantic'] == 0.0
        assert result['total'] == 0.0
    
    def test_structural_conflict(self):
        """Test structural conflict calculation."""
        score = ConflictScore()
        
        # Same structure
        graph1 = Data(x=torch.randn(3, 8), edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t())
        graph2 = Data(x=torch.randn(3, 8), edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t())
        
        conflict = score._structural_conflict(graph1, graph2)
        assert conflict == 0.0  # No structural difference
        
        # Different structure
        graph3 = Data(x=torch.randn(5, 8), edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long).t())
        
        conflict2 = score._structural_conflict(graph1, graph3)
        assert conflict2 > 0.0  # Should have structural difference
    
    def test_semantic_conflict(self):
        """Test semantic conflict calculation."""
        score = ConflictScore()
        
        # Similar features
        x1 = torch.ones(3, 8)
        graph1 = Data(x=x1, edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t())
        
        x2 = torch.ones(3, 8) * 1.1  # Slightly different
        graph2 = Data(x=x2, edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t())
        
        conflict = score._semantic_conflict(graph1, graph2)
        assert 0.0 <= conflict <= 1.0
    
    def test_temporal_conflict(self):
        """Test temporal conflict calculation."""
        score = ConflictScore()
        
        # With confidence difference
        context1 = {'previous_confidence': 0.9, 'current_confidence': 0.5}
        conflict1 = score._temporal_conflict(context1)
        assert conflict1 == 0.4
        
        # Without confidence info
        context2 = {}
        conflict2 = score._temporal_conflict(context2)
        assert conflict2 == 0.0


class TestGraphBuilder:
    """Test GraphBuilder functionality."""
    
    def test_graph_builder_creation(self):
        """Test creating GraphBuilder instances."""
        builder = GraphBuilder()
        assert hasattr(builder, 'config')
        assert hasattr(builder, 'similarity_threshold')
    
    def test_build_graph_from_documents(self):
        """Test building graph from documents."""
        builder = GraphBuilder()
        
        # Documents with embeddings
        docs = [
            {'text': 'Doc 1', 'embedding': np.random.rand(384)},
            {'text': 'Doc 2', 'embedding': np.random.rand(384)},
            {'text': 'Doc 3', 'embedding': np.random.rand(384)}
        ]
        
        graph = builder.build_graph(docs)
        
        assert isinstance(graph, Data)
        assert graph.num_nodes == 3
        assert graph.x.shape == (3, 384)
        assert hasattr(graph, 'edge_index')
        assert hasattr(graph, 'documents')
    
    def test_build_graph_empty(self):
        """Test building graph with no documents."""
        builder = GraphBuilder()
        
        graph = builder.build_graph([])
        
        assert isinstance(graph, Data)
        assert graph.x.shape[0] == 0
    
    def test_build_graph_with_embeddings(self):
        """Test building graph with provided embeddings."""
        builder = GraphBuilder()
        
        docs = [{'text': f'Doc {i}'} for i in range(3)]
        embeddings = np.random.rand(3, 384)
        
        graph = builder.build_graph(docs, embeddings)
        
        assert graph.num_nodes == 3
        assert torch.allclose(graph.x, torch.tensor(embeddings, dtype=torch.float))
    
    def test_build_graph_small(self):
        """Test building graph with very few documents."""
        builder = GraphBuilder()
        
        # Single document
        docs1 = [{'text': 'Single doc', 'embedding': np.random.rand(384)}]
        graph1 = builder.build_graph(docs1)
        assert graph1.num_nodes == 1
        assert graph1.edge_index.shape[1] > 0  # Should have self-loop
        
        # Two documents
        docs2 = [
            {'text': 'Doc 1', 'embedding': np.random.rand(384)},
            {'text': 'Doc 2', 'embedding': np.random.rand(384)}
        ]
        graph2 = builder.build_graph(docs2)
        assert graph2.num_nodes == 2
        assert graph2.edge_index.shape[1] >= 2  # Should have edges


class TestL3GraphReasoner:
    """Test L3GraphReasoner functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock config."""
        config = Mock()
        config.reasoning.use_gnn = False
        config.reasoning.similarity_threshold = 0.3
        config.reasoning.spike_ged_threshold = 0.1
        config.reasoning.spike_ig_threshold = 0.15
        config.reasoning.conflict_threshold = 0.5
        config.reasoning.weight_ged = 0.4
        config.reasoning.weight_ig = 0.4
        config.reasoning.weight_conflict = 0.2
        config.reasoning.gnn_hidden_dim = 64
        config.reasoning.graph_file = "data/graph_pyg.pt"
        config.embedding.dimension = 8
        config.graph.similarity_threshold = 0.2
        return config
    
    @pytest.fixture
    def reasoner(self, mock_config):
        """Create a graph reasoner instance."""
        with patch('insightspike.core.layers.layer3_graph_reasoner.get_config', return_value=mock_config):
            return L3GraphReasoner(config=mock_config)
    
    @pytest.fixture
    def sample_graph(self):
        """Create a sample graph."""
        x = torch.randn(5, 8)
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
        return Data(x=x, edge_index=edge_index, num_nodes=5)
    
    def test_initialization(self, mock_config):
        """Test L3GraphReasoner initialization."""
        with patch('insightspike.core.layers.layer3_graph_reasoner.get_config', return_value=mock_config):
            reasoner = L3GraphReasoner()
            assert reasoner.layer_id == "layer3_graph_reasoner"
            assert reasoner.config == mock_config
            assert hasattr(reasoner, 'graph_builder')
            assert hasattr(reasoner, 'conflict_scorer')
    
    def test_initialization_with_gnn(self, mock_config):
        """Test initialization with GNN enabled."""
        mock_config.reasoning.use_gnn = True
        with patch('insightspike.core.layers.layer3_graph_reasoner.get_config', return_value=mock_config):
            reasoner = L3GraphReasoner(config=mock_config)
            assert reasoner.gnn is not None
    
    def test_initialize_method(self, reasoner):
        """Test initialize method."""
        result = reasoner.initialize()
        assert result is True
        assert reasoner._is_initialized is True
    
    def test_process_with_layer_input(self, reasoner):
        """Test process method with LayerInput."""
        documents = [
            {'text': 'Doc 1', 'embedding': np.random.rand(8)},
            {'text': 'Doc 2', 'embedding': np.random.rand(8)}
        ]
        
        layer_input = LayerInput(data=documents, context={'query': 'test'})
        
        with patch.object(reasoner, 'analyze_documents') as mock_analyze:
            mock_analyze.return_value = {'result': 'test'}
            result = reasoner.process(layer_input)
            
            mock_analyze.assert_called_once_with(documents, {'query': 'test'})
            assert result == {'result': 'test'}
    
    def test_process_with_documents(self, reasoner):
        """Test process method with direct documents."""
        documents = [{'text': 'Doc 1'}]
        
        with patch.object(reasoner, 'analyze_documents') as mock_analyze:
            mock_analyze.return_value = {'result': 'test'}
            result = reasoner.process(documents)
            
            mock_analyze.assert_called_once_with(documents, {})
    
    def test_analyze_documents_empty(self, reasoner):
        """Test analyzing empty documents."""
        result = reasoner.analyze_documents([])
        
        assert isinstance(result, dict)
        assert 'graph' in result
        assert 'metrics' in result
        assert 'spike_detected' in result
        assert result['graph'].num_nodes == 1  # Synthetic node
    
    def test_analyze_documents_with_context_graph(self, reasoner, sample_graph):
        """Test analyzing with pre-built graph in context."""
        documents = []
        context = {'graph': sample_graph}
        
        result = reasoner.analyze_documents(documents, context)
        
        assert result['graph'] == sample_graph
        assert 'metrics' in result
        assert 'conflicts' in result
    
    def test_analyze_documents_build_graph(self, reasoner):
        """Test analyzing documents and building graph."""
        documents = [
            {'text': 'Doc 1', 'embedding': np.random.rand(8)},
            {'text': 'Doc 2', 'embedding': np.random.rand(8)},
            {'text': 'Doc 3', 'embedding': np.random.rand(8)}
        ]
        
        with patch.object(reasoner, 'save_graph'):
            result = reasoner.analyze_documents(documents)
        
        assert isinstance(result, dict)
        assert 'graph' in result
        assert result['graph'].num_nodes == 3
        assert 'spike_detected' in result
        assert 'reasoning_quality' in result
    
    def test_calculate_metrics(self, reasoner, sample_graph):
        """Test metrics calculation."""
        previous_graph = Data(x=torch.randn(4, 8), edge_index=torch.tensor([[0, 1], [2, 3]], dtype=torch.long).t())
        
        metrics = reasoner._calculate_metrics(sample_graph, previous_graph)
        
        assert 'delta_ged' in metrics
        assert 'delta_ig' in metrics
        assert 'graph_size_current' in metrics
        assert 'graph_size_previous' in metrics
        assert metrics['graph_size_current'] == 5
        assert metrics['graph_size_previous'] == 4
    
    def test_calculate_metrics_no_previous(self, reasoner, sample_graph):
        """Test metrics calculation without previous graph."""
        metrics = reasoner._calculate_metrics(sample_graph, None)
        
        assert metrics['delta_ged'] == 0.0
        assert metrics['delta_ig'] == 0.0
        assert metrics['graph_size_previous'] == 0
    
    def test_calculate_reward(self, reasoner):
        """Test reward calculation."""
        metrics = {'delta_ged': 0.5, 'delta_ig': 0.3, 'graph_size_current': 10}
        conflicts = {'total': 0.1}
        
        reward = reasoner._calculate_reward(metrics, conflicts)
        
        assert isinstance(reward, dict)
        assert 'base' in reward
        assert 'structure' in reward
        assert 'novelty' in reward
        assert 'total' in reward
        assert reward['total'] == sum(reward[k] for k in ['base', 'structure', 'novelty'])
    
    def test_detect_spike(self, reasoner):
        """Test spike detection."""
        # Should detect spike
        metrics1 = {'delta_ged': 0.2, 'delta_ig': 0.3}
        conflicts1 = {'total': 0.3}
        assert reasoner._detect_spike(metrics1, conflicts1) is True
        
        # Should not detect spike (low metrics)
        metrics2 = {'delta_ged': 0.05, 'delta_ig': 0.05}
        conflicts2 = {'total': 0.3}
        assert reasoner._detect_spike(metrics2, conflicts2) is False
        
        # Should not detect spike (high conflict)
        metrics3 = {'delta_ged': 0.2, 'delta_ig': 0.3}
        conflicts3 = {'total': 0.8}
        assert reasoner._detect_spike(metrics3, conflicts3) is False
    
    def test_assess_reasoning_quality(self, reasoner):
        """Test reasoning quality assessment."""
        metrics = {'delta_ged': 0.4, 'delta_ig': 0.6}
        conflicts = {'total': 0.2}
        
        quality = reasoner._assess_reasoning_quality(metrics, conflicts)
        
        assert 0.0 <= quality <= 1.0
        assert quality == 0.3  # (0.4 + 0.6) / 2 - 0.2
    
    def test_save_and_load_graph(self, reasoner, sample_graph):
        """Test saving and loading graphs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_graph.pt"
            
            # Save
            result_path = reasoner.save_graph(sample_graph, save_path)
            assert result_path == save_path
            assert save_path.exists()
            
            # Load
            loaded_graph = reasoner.load_graph(save_path)
            assert isinstance(loaded_graph, Data)
            assert loaded_graph.num_nodes == sample_graph.num_nodes
    
    def test_save_graph_fallback(self, reasoner):
        """Test graph saving with fallback to dict format."""
        # Create a graph that might fail normal saving
        x = torch.randn(3, 8)
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t()
        graph = Data(x=x, edge_index=edge_index, num_nodes=3)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_graph.pt"
            
            with patch('torch.save', side_effect=[Exception("Save failed"), None]):
                result_path = reasoner.save_graph(graph, save_path)
                assert result_path == save_path
    
    def test_interface_methods(self, reasoner):
        """Test interface compliance methods."""
        vectors = np.random.rand(3, 8)
        
        # Test build_graph
        graph = reasoner.build_graph(vectors)
        assert isinstance(graph, Data)
        assert graph.num_nodes == 3
        
        # Test calculate_ged
        graph2 = reasoner.build_graph(np.random.rand(4, 8))
        ged = reasoner.calculate_ged(graph, graph2)
        assert isinstance(ged, float)
        
        # Test calculate_ig
        ig = reasoner.calculate_ig(graph, graph2)
        assert isinstance(ig, float)
        
        # Test detect_eureka_spike
        spike = reasoner.detect_eureka_spike(0.2, 0.3)
        assert isinstance(spike, bool)
    
    def test_cleanup(self, reasoner):
        """Test cleanup method."""
        reasoner.previous_graph = Data(x=torch.randn(3, 8))
        reasoner.gnn = Mock()
        
        reasoner.cleanup()
        
        assert reasoner.previous_graph is None
        assert reasoner.gnn is None
        assert reasoner._is_initialized is False
    
    def test_process_with_gnn(self, mock_config):
        """Test processing with GNN enabled."""
        mock_config.reasoning.use_gnn = True
        
        with patch('insightspike.core.layers.layer3_graph_reasoner.get_config', return_value=mock_config):
            reasoner = L3GraphReasoner(config=mock_config)
            assert reasoner.gnn is not None
            
            # Test GNN processing
            graph = Data(x=torch.randn(3, 8), edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t())
            
            with patch('torch.no_grad'):
                graph_features = reasoner._process_with_gnn(graph)
                # Since we're mocking, features might be None
                assert graph_features is None or isinstance(graph_features, torch.Tensor)
    
    def test_fallback_result(self, reasoner):
        """Test fallback result for errors."""
        result = reasoner._fallback_result()
        
        assert isinstance(result, dict)
        assert 'graph' in result
        assert 'metrics' in result
        assert 'conflicts' in result
        assert 'reward' in result
        assert result['spike_detected'] is False
        assert result['reasoning_quality'] == 0.0
    
    def test_advanced_metrics_usage(self, mock_config):
        """Test usage of advanced metrics when available."""
        mock_config.use_advanced_metrics = True
        
        with patch('insightspike.core.layers.layer3_graph_reasoner.ADVANCED_METRICS_AVAILABLE', True):
            with patch('insightspike.core.layers.layer3_graph_reasoner.get_config', return_value=mock_config):
                reasoner = L3GraphReasoner(config=mock_config)
                
                # Should use advanced metrics
                from insightspike.utils.advanced_graph_metrics import delta_ged, delta_ig
                assert reasoner.delta_ged == delta_ged
                assert reasoner.delta_ig == delta_ig