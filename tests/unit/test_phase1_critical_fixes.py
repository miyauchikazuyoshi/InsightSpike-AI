"""
Unit tests for Phase 1 critical fixes
Tests that the immediate failures have been resolved
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from insightspike.config.presets import ConfigPresets
from insightspike.implementations.agents import MainAgent
from insightspike.implementations.layers.layer2_compatibility import CompatibleL2MemoryManager
from insightspike.features.graph_reasoning.graph_analyzer import GraphAnalyzer
from insightspike.implementations.datastore.memory_store import InMemoryDataStore


class TestMainAgentFixes:
    """Test MainAgent fixes"""
    
    def test_l1_embedder_exists(self):
        """Test that l1_embedder attribute exists and is initialized"""
        config = ConfigPresets.development()
        agent = MainAgent(config=config, datastore=InMemoryDataStore())
        
        # Check that l1_embedder exists
        assert hasattr(agent, 'l1_embedder'), "MainAgent should have l1_embedder attribute"
        assert agent.l1_embedder is not None, "l1_embedder should be initialized"
        
    def test_add_knowledge_with_l1_embedder(self):
        """Test that add_knowledge works with l1_embedder"""
        config = ConfigPresets.development()
        agent = MainAgent(config=config, datastore=InMemoryDataStore())
        agent.initialize()
        
        # Test add_knowledge
        result = agent.add_knowledge("Test knowledge")
        assert result.get("success") is not False, f"add_knowledge should succeed, got: {result}"
        
    def test_l1_embedder_with_different_configs(self):
        """Test l1_embedder initialization with different config types"""
        # Test with Pydantic config
        pydantic_config = ConfigPresets.development()
        agent1 = MainAgent(config=pydantic_config, datastore=InMemoryDataStore())
        assert hasattr(agent1, 'l1_embedder')
        
        # Test with dict-like config (legacy)
        class MockConfig:
            def __init__(self):
                self.embedding = type('obj', (object,), {
                    'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
                    'dimension': 384
                })()
        
        legacy_config = MockConfig()
        agent2 = MainAgent(config=legacy_config, datastore=InMemoryDataStore())
        assert hasattr(agent2, 'l1_embedder')


class TestL2MemoryManagerFixes:
    """Test L2MemoryManager fixes"""
    
    def test_encode_text_method_exists(self):
        """Test that _encode_text method exists"""
        memory = CompatibleL2MemoryManager(dim=384)
        
        # Check that _encode_text exists
        assert hasattr(memory, '_encode_text'), "CompatibleL2MemoryManager should have _encode_text method"
        
    def test_encode_text_returns_correct_shape(self):
        """Test that _encode_text returns correct embedding shape"""
        memory = CompatibleL2MemoryManager(dim=384)
        
        # Test encoding
        embedding = memory._encode_text("Test text")
        
        assert isinstance(embedding, np.ndarray), "Should return numpy array"
        assert embedding.shape == (384,), f"Should return shape (384,), got {embedding.shape}"
        assert embedding.dtype == np.float32 or embedding.dtype == np.float64, "Should be float type"
        
    def test_encode_text_handles_shape_normalization(self):
        """Test that _encode_text normalizes (1, 384) to (384,)"""
        memory = CompatibleL2MemoryManager(dim=384)
        
        # Mock embedder to return (1, 384) shape
        mock_embedder = Mock()
        mock_embedder.encode.return_value = np.ones((1, 384))
        memory.embedder = mock_embedder
        
        # Test encoding
        embedding = memory._encode_text("Test text")
        
        assert embedding.shape == (384,), f"Should normalize to (384,), got {embedding.shape}"


class TestGraphAnalyzerFixes:
    """Test GraphAnalyzer fixes"""
    
    def test_handles_networkx_graphs(self):
        """Test that GraphAnalyzer handles NetworkX graphs"""
        import networkx as nx
        
        analyzer = GraphAnalyzer()
        
        # Create NetworkX graphs
        g1 = nx.Graph()
        g1.add_nodes_from([0, 1, 2])
        g1.add_edges_from([(0, 1), (1, 2)])
        
        g2 = nx.Graph()
        g2.add_nodes_from([0, 1, 2, 3])
        g2.add_edges_from([(0, 1), (1, 2), (2, 3)])
        
        # Mock metric functions
        def mock_ged(g1, g2):
            return abs(g1.number_of_nodes() - g2.number_of_nodes())
        
        def mock_ig(g1, g2):
            return 0.1
        
        # Test calculation
        metrics = analyzer.calculate_metrics(g2, g1, mock_ged, mock_ig)
        
        assert metrics is not None, "Should return metrics"
        assert "delta_ged" in metrics, "Should have delta_ged"
        assert "delta_ig" in metrics, "Should have delta_ig"
        assert metrics["graph_size_current"] == 4, "Should have correct current size"
        assert metrics["graph_size_previous"] == 3, "Should have correct previous size"
        
    def test_handles_none_previous_graph(self):
        """Test that GraphAnalyzer handles None previous graph"""
        import networkx as nx
        
        analyzer = GraphAnalyzer()
        
        # Create NetworkX graph
        g = nx.Graph()
        g.add_nodes_from([0, 1, 2])
        
        # Test with None previous
        metrics = analyzer.calculate_metrics(g, None, None, None)
        
        assert metrics["delta_ged"] == 0.0, "Should return 0 for delta_ged"
        assert metrics["delta_ig"] == 0.0, "Should return 0 for delta_ig"
        assert metrics["graph_size_current"] == 3, "Should have correct current size"
        assert metrics["graph_size_previous"] == 0, "Should have 0 for previous size"
        
    def test_handles_pyg_graphs(self):
        """Test that GraphAnalyzer still handles PyTorch Geometric graphs"""
        try:
            from torch_geometric.data import Data
            import torch
        except ImportError:
            pytest.skip("PyTorch Geometric not available")
        
        analyzer = GraphAnalyzer()
        
        # Create PyG graphs
        g1 = Data(
            x=torch.randn(3, 384),
            edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t()
        )
        g1.num_nodes = 3
        
        g2 = Data(
            x=torch.randn(4, 384),
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long).t()
        )
        g2.num_nodes = 4
        
        # Mock metric functions
        def mock_ged(g1, g2):
            return 1.0
        
        def mock_ig(vecs1, vecs2):
            return 0.2
        
        # Test calculation
        metrics = analyzer.calculate_metrics(g2, g1, mock_ged, mock_ig)
        
        assert metrics is not None, "Should return metrics"
        assert metrics["graph_size_current"] == 4, "Should have correct current size"
        assert metrics["graph_size_previous"] == 3, "Should have correct previous size"


class TestIntegration:
    """Integration tests for all fixes together"""
    
    def test_full_pipeline_without_patches(self):
        """Test that the full pipeline works without applying patches"""
        # Create agent without patches
        config = ConfigPresets.development()
        agent = MainAgent(config=config, datastore=InMemoryDataStore())
        
        # Initialize
        assert agent.initialize(), "Agent should initialize successfully"
        
        # Add knowledge
        result = agent.add_knowledge("Neural networks are computational models.")
        assert result.get("success") is not False, "Should add knowledge successfully"
        
        # Process question
        answer = agent.process_question("What are neural networks?", max_cycles=1)
        assert hasattr(answer, 'response'), "Should return CycleResult with response"
        assert answer.success, "Processing should succeed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])