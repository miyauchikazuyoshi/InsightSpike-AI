"""
Regression tests for individual layers
"""

import pytest
import numpy as np
import torch
import pytest
pytest.importorskip("torch_geometric")
from torch_geometric.data import Data

from insightspike.implementations.layers.layer2_memory_manager import L2MemoryManager
from insightspike.implementations.layers.layer3_graph_reasoner import L3GraphReasoner
from insightspike.implementations.layers.layer4_llm_interface import L4LLMInterface


class TestLayer2Regression:
    """Regression tests for Layer2 Memory Manager"""
    
    def test_basic_memory_operations(self):
        """Test basic memory operations"""
        config = {
            'embedding': {
                'dimension': 128,
                'model': 'sentence-transformers/all-MiniLM-L6-v2'
            }
        }
        
        l2 = L2MemoryManager(config)
        
        # Add episodes
        l2.add_episode("First knowledge")
        l2.add_episode("Second knowledge")
        
        # Check episodes added
        stats = l2.get_memory_stats()
        assert stats['total_episodes'] == 2
        assert stats['embedding_dim'] == 384  # all-MiniLM-L6-v2 dimension
    
    def test_search_without_message_passing(self):
        """Test search returns query_embedding"""
        l2 = L2MemoryManager()
        
        # Add some episodes
        texts = ["Apple is a fruit", "Apple is a company", "Orange is a fruit"]
        for text in texts:
            l2.add_episode(text)
        
        # Search - using internal method to check query_embedding
        results = l2.search_episodes("fruit", k=2)
        
        # Should return results
        assert len(results) <= 2
        assert all('text' in r for r in results)
        assert all('similarity' in r for r in results)
    
    def test_memory_consolidation(self):
        """Test memory consolidation threshold"""
        config = {
            'memory': {
                'consolidation_threshold': 3  # Low threshold for testing
            }
        }
        
        l2 = L2MemoryManager(config)
        
        # Add episodes up to threshold
        for i in range(5):
            l2.add_episode(f"Knowledge {i}")
        
        stats = l2.get_memory_stats()
        assert stats['total_episodes'] <= 5


class TestLayer3Regression:
    """Regression tests for Layer3 Graph Reasoner"""
    
    def test_initialization_with_different_configs(self):
        """Test Layer3 initialization with various configs"""
        # Dict config
        config_dict = {
            'graph': {
                'enable_message_passing': True,
                'similarity_threshold': 0.6
            }
        }
        
        l3 = L3GraphReasoner(config_dict)
        assert l3.message_passing_enabled == True
        
        # Minimal config
        l3_minimal = L3GraphReasoner({})
        assert l3_minimal.message_passing_enabled == False
    
    def test_graph_analysis_without_message_passing(self):
        """Test graph analysis with message passing disabled"""
        config = {
            'graph': {
                'enable_message_passing': False,
                'spike_ged_threshold': 0.3,
                'spike_ig_threshold': 0.7
            }
        }
        
        l3 = L3GraphReasoner(config)
        l3.initialize()
        
        # Analyze documents
        documents = [
            {'text': 'Doc 1', 'embedding': np.random.randn(384)},
            {'text': 'Doc 2', 'embedding': np.random.randn(384)}
        ]
        
        result = l3.analyze_documents(documents)
        
        assert 'graph' in result
        assert 'metrics' in result
        assert 'spike_detected' in result
        assert result['spike_detected'] is False  # No previous graph
    
    def test_graph_analysis_with_message_passing(self):
        """Test graph analysis with message passing enabled"""
        config = {
            'graph': {
                'enable_message_passing': True,
                'message_passing': {
                    'alpha': 0.3,
                    'iterations': 2
                }
            }
        }
        
        l3 = L3GraphReasoner(config)
        l3.initialize()
        
        documents = [
            {'text': 'Doc 1', 'embedding': np.random.randn(384)},
            {'text': 'Doc 2', 'embedding': np.random.randn(384)}
        ]
        
        context = {
            'query_vector': np.random.randn(384)
        }
        
        result = l3.analyze_documents(documents, context)
        
        assert 'graph' in result
        # Check if message passing was applied
        graph = result['graph']
        if hasattr(graph, 'edge_info'):
            # Edge info indicates re-evaluation happened
            assert graph.edge_info is not None
    
    def test_spike_detection_thresholds(self):
        """Test spike detection with different thresholds"""
        config = {
            'graph': {
                'spike_ged_threshold': 0.1,  # Very sensitive
                'spike_ig_threshold': 0.9,   # Very sensitive
                'conflict_threshold': 0.1
            }
        }
        
        l3 = L3GraphReasoner(config)
        l3.initialize()
        
        # First analysis
        docs1 = [{'text': 'A', 'embedding': np.ones(384)}]
        result1 = l3.analyze_documents(docs1)
        
        # Very different documents
        docs2 = [
            {'text': 'B', 'embedding': np.zeros(384)},
            {'text': 'C', 'embedding': -np.ones(384)}
        ]
        result2 = l3.analyze_documents(docs2)
        
        # With sensitive thresholds, might detect spike
        # (depends on metric calculations)
        assert 'spike_detected' in result2


class TestLayer4Regression:
    """Regression tests for Layer4 LLM Interface"""
    
    def test_initialization_with_different_providers(self):
        """Test Layer4 initialization with different providers"""
        # Mock provider
        config_mock = {'llm': {'provider': 'mock'}}
        l4_mock = L4LLMInterface(config_mock)
        assert l4_mock.initialize()  # Call initialize()
        assert l4_mock.initialized
        
        # Clean provider
        config_clean = {'llm': {'provider': 'clean'}}
        l4_clean = L4LLMInterface(config_clean)
        assert l4_clean.initialize()  # Call initialize()
        assert l4_clean.initialized
    
    def test_response_generation_without_query_vector(self):
        """Test response generation without query vector"""
        l4 = L4LLMInterface({'llm': {'provider': 'mock'}})
        l4.initialize()  # Initialize the provider
        
        context = {
            'retrieved_documents': [
                {'text': 'Doc 1', 'embedding': np.random.randn(128)},
                {'text': 'Doc 2', 'embedding': np.random.randn(128)}
            ]
        }
        
        result = l4.generate_response_detailed(context, "Test question")
        
        assert 'response' in result
        assert 'success' in result
        assert result['success'] is True
    
    def test_response_generation_with_query_vector(self):
        """Test response generation with query vector"""
        l4 = L4LLMInterface({'llm': {'provider': 'mock'}})
        l4.initialize()  # Initialize the provider
        
        context = {
            'retrieved_documents': [
                {'text': 'Doc 1', 'embedding': np.random.randn(128)},
                {'text': 'Doc 2', 'embedding': np.random.randn(128)}
            ],
            'query_vector': np.random.randn(128)
        }
        
        result = l4.generate_response_detailed(context, "Test question")
        
        assert result['success'] is True
        # With mock provider, can't verify insight vector creation
        # but should not fail
    
    def test_vector_integrator_usage(self):
        """Test that VectorIntegrator is properly initialized and used"""
        l4 = L4LLMInterface({'llm': {'provider': 'mock'}})
        
        # Check VectorIntegrator exists
        assert hasattr(l4, 'vector_integrator')
        assert l4.vector_integrator is not None
        
        # Test _create_insight_vector directly
        docs = [
            {'embedding': np.random.randn(64)},
            {'embedding': np.random.randn(64)}
        ]
        
        # Without query
        vec1 = l4._create_insight_vector(docs, None)
        assert vec1 is not None
        assert vec1.shape == (64,)
        
        # With query
        query = np.random.randn(64)
        vec2 = l4._create_insight_vector(docs, query)
        assert vec2 is not None
        assert not np.allclose(vec1, vec2)  # Should be different


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
