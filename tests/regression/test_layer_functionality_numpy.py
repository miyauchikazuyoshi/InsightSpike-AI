"""
Regression tests for individual layers using NumPy backend to avoid Faiss issues
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


class TestLayer3WithNumPy:
    """Test Layer3 with NumPy backend to avoid Faiss segfault"""
    
    def test_layer3_graph_analysis_without_message_passing(self):
        """Test graph analysis without message passing enabled"""
        config = {
            'graph': {
                'use_faiss': False,  # Force NumPy backend
                'enable_message_passing': False,
                'enable_graph_search': False,
                'use_gnn': False
            },
            'reasoning': {
                'algorithm_selector': 'default'
            }
        }
        
        l2 = L2MemoryManager()
        l3 = L3GraphReasoner(config)
        
        # Add knowledge and build graph
        texts = ["Apple is a fruit", "Orange is a fruit", "Fruit is healthy"]
        for text in texts:
            episode = l2.add_episode(text)
        
        # Get episodes - accessing internal list directly
        episodes = l2.episodes
        documents = [{"embedding": ep.vec, "text": ep.text} for ep in episodes]
        
        # Build graph
        l3.build_graph(documents)
        
        # Analyze without message passing
        question = "What is a fruit?"
        query_embedding = l2.embedding_model.encode(question)
        
        # Use analyze_documents method with query context
        context = {'query_embedding': query_embedding}
        result = l3.analyze_documents(documents, context)
        
        # Validate result structure
        assert result is not None
        assert 'spike_detected' in result
        assert 'metrics' in result
        assert 'graph_context' in result
    
    def test_layer3_graph_analysis_with_message_passing(self):
        """Test graph analysis with message passing enabled"""
        config = {
            'graph': {
                'use_faiss': False,  # Force NumPy backend
                'enable_message_passing': True,
                'message_passing': {
                    'alpha': 0.3,
                    'iterations': 3
                },
                'enable_graph_search': False,
                'use_gnn': False
            },
            'reasoning': {
                'algorithm_selector': 'default'
            }
        }
        
        l2 = L2MemoryManager()
        l3 = L3GraphReasoner(config)
        
        # Add knowledge and build graph
        texts = ["Apple is a fruit", "Orange is a fruit", "Fruit is healthy", "Healthy food is good"]
        for text in texts:
            episode = l2.add_episode(text)
        
        # Get episodes and build graph - accessing internal list directly  
        episodes = l2.episodes
        documents = [{"embedding": ep.vec, "text": ep.text} for ep in episodes]
        l3.build_graph(documents)
        
        # Analyze with message passing
        question = "What is healthy?"
        query_embedding = l2.embedding_model.encode(question)
        
        # Use analyze_documents method with query context
        context = {'query_embedding': query_embedding, 'query_vector': query_embedding}
        result = l3.analyze_documents(documents, context)
        
        # Validate result
        assert result is not None
        assert 'spike_detected' in result
        assert 'metrics' in result
        assert 'graph_context' in result
        
        # Check if message passing was applied
        if 'message_passing_applied' in result.get('metrics', {}):
            assert result['metrics']['message_passing_applied'] is True
    
    def test_layer3_spike_detection_with_threshold(self):
        """Test spike detection with different thresholds"""
        config = {
            'graph': {
                'use_faiss': False,
                'enable_message_passing': False,
                'enable_graph_search': False,
                'use_gnn': False
            },
            'reasoning': {
                'algorithm_selector': 'default',
                'spike_threshold': 0.5  # Custom threshold
            }
        }
        
        l2 = L2MemoryManager()
        l3 = L3GraphReasoner(config)
        
        # Add simple knowledge
        texts = ["A leads to B", "B leads to C", "C leads to D"]
        for text in texts:
            l2.add_episode(text)
        
        episodes = l2.episodes
        documents = [{"embedding": ep.vec, "text": ep.text} for ep in episodes]
        l3.build_graph(documents)
        
        # Test with question that should trigger spike
        question = "How does A connect to D?"
        query_embedding = l2.embedding_model.encode(question)
        
        # Use analyze_documents method with query context
        context = {'query_embedding': query_embedding}
        result = l3.analyze_documents(documents, context)
        
        assert result is not None
        assert isinstance(result['spike_detected'], bool)
        assert 'metrics' in result


class TestLayerIntegrationWithNumPy:
    """Test layer integration with NumPy backend"""
    
    def test_layer2_to_layer3_flow(self):
        """Test data flow from Layer2 to Layer3"""
        config = {
            'graph': {
                'use_faiss': False,
                'enable_message_passing': True,
                'message_passing': {
                    'alpha': 0.3,
                    'iterations': 2
                }
            }
        }
        
        l2 = L2MemoryManager()
        l3 = L3GraphReasoner(config)
        
        # Add knowledge via L2
        knowledge = [
            "Python is a programming language",
            "Java is a programming language", 
            "Programming languages are used to write software"
        ]
        
        for text in knowledge:
            l2.add_episode(text)
        
        # Get documents for L3
        episodes = l2.episodes
        documents = [{"embedding": ep.vec, "text": ep.text} for ep in episodes]
        
        # Build graph in L3
        l3.build_graph(documents)
        
        # Search in L2 and use result for L3
        search_results = l2.search_episodes("What languages are used for programming?", k=2)
        query_embedding = l2.embedding_model.encode("What languages are used for programming?")
        
        # Analyze with L3
        context = {'query_embedding': query_embedding, 'query_vector': query_embedding}
        analysis = l3.analyze_documents(documents, context)
        
        # Validate flow
        assert len(search_results) == 2
        assert analysis is not None
        assert 'graph_context' in analysis
    
    def test_query_embedding_propagation(self):
        """Test query embedding propagates through layers"""
        config = {
            'graph': {
                'use_faiss': False,
                'enable_message_passing': True,
                'message_passing': {
                    'alpha': 0.5,
                    'iterations': 3
                }
            }
        }
        
        l2 = L2MemoryManager()
        l3 = L3GraphReasoner(config)
        l4 = L4LLMInterface({'llm': {'provider': 'mock'}})
        l4.initialize()  # Initialize provider
        
        # Add knowledge
        l2.add_episode("The sky is blue")
        l2.add_episode("The ocean is blue")
        l2.add_episode("Blue is a color")
        
        # Build graph
        episodes = l2.episodes
        documents = [{"embedding": ep.vec, "text": ep.text} for ep in episodes]
        l3.build_graph(documents)
        
        # Create query
        question = "Why are things blue?"
        query_embedding = l2.embedding_model.encode(question)
        
        # Search with L2
        search_results = l2.search_episodes(question, k=2)
        
        # Analyze with L3
        graph_analysis = l3.analyze_graph(query_embedding, query_vector=query_embedding)
        
        # Generate response with L4
        response = l4.generate_response(
            question=question,
            documents=search_results,
            graph_analysis=graph_analysis,
            query_vector=query_embedding
        )
        
        # Validate propagation
        assert response is not None
        assert 'response' in response
        assert 'insight_vector' in response
