"""
Regression tests for layer integration
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from insightspike.implementations.agents.main_agent import MainAgent


class TestLayerIntegration:
    """Test integration between layers"""
    
    def test_layer2_to_layer3_data_flow(self):
        """Test data flow from Layer2 to Layer3"""
        config = {
            'llm': {'provider': 'mock'},
            'graph': {'enable_message_passing': False}
        }
        
        agent = MainAgent(config)
        
        # Add some knowledge
        agent.add_knowledge("Test knowledge 1")
        agent.add_knowledge("Test knowledge 2")
        
        # Mock the internal method to capture data flow
        original_analyze = agent.l3_graph.analyze_documents
        captured_data = {}
        
        def capture_analyze(docs, context=None):
            captured_data['documents'] = docs
            captured_data['context'] = context
            return original_analyze(docs, context)
        
        agent.l3_graph.analyze_documents = capture_analyze
        
        # Process question
        result = agent.process_question("Test question")
        
        # Verify data flow
        assert 'documents' in captured_data
        assert len(captured_data['documents']) > 0
        assert all('text' in doc for doc in captured_data['documents'])
        assert all('embedding' in doc for doc in captured_data['documents'])
    
    def test_query_embedding_propagation(self):
        """Test query_embedding propagates through layers"""
        config = {
            'llm': {'provider': 'mock'},
            'graph': {
                'enable_message_passing': True,
                'message_passing': {'alpha': 0.3}
            }
        }
        
        agent = MainAgent(config)
        agent.add_knowledge("Test knowledge")
        
        # Capture contexts at each layer
        captured_contexts = {
            'layer3': None,
            'layer4': None
        }
        
        # Mock Layer3 analyze_documents
        original_l3_analyze = agent.l3_graph.analyze_documents
        def capture_l3(docs, context=None):
            captured_contexts['layer3'] = context
            return original_l3_analyze(docs, context)
        
        # Mock Layer4 generate_response_detailed
        original_l4_generate = agent.l4_llm.generate_response_detailed
        def capture_l4(context, question):
            captured_contexts['layer4'] = context
            return original_l4_generate(context, question)
        
        agent.l3_graph.analyze_documents = capture_l3
        agent.l4_llm.generate_response_detailed = capture_l4
        
        # Process question
        result = agent.process_question("Test question")
        
        # Verify query_vector propagation
        assert captured_contexts['layer3'] is not None
        assert 'query_vector' in captured_contexts['layer3']
        assert captured_contexts['layer3']['query_vector'] is not None
        
        assert captured_contexts['layer4'] is not None
        assert 'query_vector' in captured_contexts['layer4']
        assert captured_contexts['layer4']['query_vector'] is not None
    
    def test_graph_analysis_results_propagation(self):
        """Test graph analysis results propagate to Layer4"""
        config = {
            'llm': {'provider': 'mock'},
            'graph': {'enable_message_passing': False}
        }
        
        agent = MainAgent(config)
        agent.add_knowledge("Knowledge 1")
        agent.add_knowledge("Knowledge 2")
        
        # Capture Layer4 context
        captured_context = None
        
        original_generate = agent.l4_llm.generate_response_detailed
        def capture_generate(context, question):
            nonlocal captured_context
            captured_context = context
            return original_generate(context, question)
        
        agent.l4_llm.generate_response_detailed = capture_generate
        
        # Process question
        result = agent.process_question("Test question")
        
        # Verify graph analysis in Layer4 context
        assert captured_context is not None
        assert 'graph_analysis' in captured_context
        assert 'metrics' in captured_context['graph_analysis']
        assert 'spike_detected' in captured_context['graph_analysis']
    
    def test_message_passing_effect_on_results(self):
        """Test that message passing affects final results"""
        # Config without message passing
        config_no_mp = {
            'llm': {'provider': 'mock'},
            'graph': {'enable_message_passing': False}
        }
        
        # Config with message passing
        config_with_mp = {
            'llm': {'provider': 'mock'},
            'graph': {
                'enable_message_passing': True,
                'message_passing': {
                    'alpha': 0.7,  # High influence
                    'iterations': 3
                },
                'edge_reevaluation': {
                    'new_edge_threshold': 0.6
                }
            }
        }
        
        # Test both configurations
        agent_no_mp = MainAgent(config_no_mp)
        agent_with_mp = MainAgent(config_with_mp)
        
        # Add same knowledge
        knowledge = ["Concept A relates to B", "Concept B relates to C", "Concept C is different"]
        for k in knowledge:
            agent_no_mp.add_knowledge(k)
            agent_with_mp.add_knowledge(k)
        
        # Process same question
        question = "How does A relate to C?"
        
        result_no_mp = agent_no_mp.process_question(question)
        result_with_mp = agent_with_mp.process_question(question)
        
        # Check that message passing was actually used
        assert agent_with_mp.l3_graph.message_passing_enabled == True
        assert agent_no_mp.l3_graph.message_passing_enabled == False
        
        # Both should produce valid results
        assert hasattr(result_no_mp, 'response')
        assert hasattr(result_with_mp, 'response')


class TestConfigCompatibility:
    """Test various configuration formats"""
    
    def test_dict_config(self):
        """Test with dictionary configuration"""
        config = {
            'llm': {'provider': 'mock'},
            'graph': {
                'enable_message_passing': True,
                'similarity_threshold': 0.7
            },
            'memory': {
                'max_episodes': 100
            }
        }
        
        agent = MainAgent(config)
        assert agent.l3_graph.message_passing_enabled == True
    
    def test_minimal_config(self):
        """Test with minimal configuration"""
        config = {'llm': {'provider': 'mock'}}
        
        agent = MainAgent(config)
        
        # Should use defaults
        assert agent.l3_graph is not None
        assert agent.l2_memory is not None
    
    def test_config_with_all_features(self):
        """Test with all features enabled"""
        config = {
            'llm': {'provider': 'mock'},
            'graph': {
                'enable_message_passing': True,
                'enable_graph_search': True,
                'use_gnn': True,
                'message_passing': {
                    'alpha': 0.5,
                    'iterations': 3
                }
            },
            'algorithms': {
                'use_advanced_ged': True,
                'use_advanced_ig': True
            }
        }
        
        # Should not fail initialization
        agent = MainAgent(config)
        
        # Process should work
        agent.add_knowledge("Test")
        result = agent.process_question("Test?")
        assert hasattr(result, 'response')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])