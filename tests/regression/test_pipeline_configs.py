"""
Regression tests for complete pipeline with various configurations
"""

import pytest
import os
from insightspike.implementations.agents.main_agent import MainAgent


class TestPipelineConfigurations:
    """Test complete pipeline with different configurations"""
    
    @pytest.fixture
    def knowledge_base(self):
        """Common knowledge base for tests"""
        return [
            "The sun is a star at the center of our solar system.",
            "Photosynthesis is how plants convert sunlight into energy.",
            "Solar panels convert sunlight into electricity.",
            "Neural networks are inspired by biological neurons.",
            "Deep learning uses multiple layers of neural networks.",
            "Machine learning is a subset of artificial intelligence."
        ]
    
    @pytest.fixture
    def test_questions(self):
        """Common test questions"""
        return [
            "How does the sun produce energy?",
            "What is the relationship between photosynthesis and solar panels?",
            "How are neural networks related to AI?"
        ]
    
    def test_baseline_configuration(self, knowledge_base, test_questions):
        """Test with baseline configuration (all features disabled)"""
        config = {
            'llm': {'provider': 'mock'},
            'graph': {
                'enable_message_passing': False,
                'enable_graph_search': False,
                'use_gnn': False,
                'similarity_threshold': 0.6,
                'spike_ged_threshold': 0.3,
                'spike_ig_threshold': 0.7
            }
        }
        
        agent = MainAgent(config)
        
        # Add knowledge
        for k in knowledge_base:
            agent.add_knowledge(k)
        
        # Test each question
        results = []
        for q in test_questions:
            result = agent.process_question(q)
            results.append({
                'question': q,
                'has_response': hasattr(result, 'response'),
                'spike_detected': getattr(result, 'spike_detected', False),
                'graph_nodes': self._get_graph_nodes(result)
            })
        
        # Verify baseline behavior
        assert all(r['has_response'] for r in results)
        assert agent.l3_graph.message_passing_enabled == False
    
    def test_message_passing_only(self, knowledge_base, test_questions):
        """Test with message passing enabled only"""
        config = {
            'llm': {'provider': 'mock'},
            'graph': {
                'enable_message_passing': True,
                'message_passing': {
                    'alpha': 0.3,
                    'iterations': 3
                },
                'edge_reevaluation': {
                    'new_edge_threshold': 0.8,
                    'max_new_edges_per_node': 5
                },
                'enable_graph_search': False,
                'use_gnn': False
            }
        }
        
        agent = MainAgent(config)
        
        # Add knowledge
        for k in knowledge_base:
            agent.add_knowledge(k)
        
        # Process questions
        for q in test_questions:
            result = agent.process_question(q)
            
            # Check message passing was used
            assert agent.l3_graph.message_passing_enabled == True
            assert hasattr(result, 'graph_analysis')
            
            # Check for edge re-evaluation effects
            graph = result.graph_analysis.get('graph')
            if graph and hasattr(graph, 'edge_info'):
                # Edge info indicates re-evaluation happened
                assert graph.edge_info is not None
    
    def test_graph_search_only(self, knowledge_base, test_questions):
        """Test with graph search enabled only"""
        config = {
            'llm': {'provider': 'mock'},
            'graph': {
                'enable_message_passing': False,
                'enable_graph_search': True,
                'search_depth': 2,
                'use_gnn': False
            }
        }
        
        agent = MainAgent(config)
        
        # Add knowledge
        for k in knowledge_base:
            agent.add_knowledge(k)
        
        # Process questions
        for q in test_questions:
            result = agent.process_question(q)
            assert hasattr(result, 'response')
    
    def test_all_features_enabled(self, knowledge_base, test_questions):
        """Test with all features enabled"""
        config = {
            'llm': {'provider': 'mock'},
            'graph': {
                'enable_message_passing': True,
                'message_passing': {
                    'alpha': 0.5,
                    'iterations': 3,
                    'aggregation': 'weighted_mean'
                },
                'edge_reevaluation': {
                    'new_edge_threshold': 0.7,
                    'similarity_threshold': 0.6
                },
                'enable_graph_search': True,
                'use_gnn': True,
                'gnn_hidden_dim': 128
            },
            'algorithms': {
                'use_advanced_ged': True,
                'use_advanced_ig': True
            }
        }
        
        agent = MainAgent(config)
        
        # Add knowledge
        for k in knowledge_base:
            agent.add_knowledge(k)
        
        # Should not crash with all features
        for q in test_questions:
            result = agent.process_question(q)
            assert hasattr(result, 'response')
    
    def test_sensitive_spike_detection(self, knowledge_base):
        """Test with sensitive spike detection settings"""
        config = {
            'llm': {'provider': 'mock'},
            'graph': {
                'enable_message_passing': True,
                'message_passing': {
                    'alpha': 0.4,
                    'iterations': 2
                },
                'spike_ged_threshold': 0.1,  # Very sensitive
                'spike_ig_threshold': 0.9,   # Very sensitive
                'conflict_threshold': 0.3
            }
        }
        
        agent = MainAgent(config)
        
        # Add initial knowledge
        agent.add_knowledge(knowledge_base[0])
        
        # First question
        result1 = agent.process_question("What is the sun?")
        
        # Add very different knowledge
        agent.add_knowledge("Quantum computing uses qubits.")
        agent.add_knowledge("Blockchain is a distributed ledger.")
        
        # Question about new topic
        result2 = agent.process_question("What is quantum computing?")
        
        # With sensitive settings, might detect spikes
        # Check both results are valid
        assert hasattr(result1, 'response')
        assert hasattr(result2, 'response')
    
    def test_edge_cases(self):
        """Test edge cases"""
        config = {
            'llm': {'provider': 'mock'},
            'graph': {
                'enable_message_passing': True,
                'message_passing': {'alpha': 0.5}
            }
        }
        
        agent = MainAgent(config)
        
        # Empty knowledge base
        result1 = agent.process_question("Test question?")
        assert hasattr(result1, 'response')
        
        # Single knowledge item
        agent.add_knowledge("Single fact")
        result2 = agent.process_question("What do you know?")
        assert hasattr(result2, 'response')
        
        # Very long question
        long_question = "What " * 100 + "is this?"
        result3 = agent.process_question(long_question)
        assert hasattr(result3, 'response')
    
    def test_config_variations_dont_break_pipeline(self):
        """Test that various config combinations don't break the pipeline"""
        config_variations = [
            # High alpha, low iterations
            {
                'llm': {'provider': 'mock'},
                'graph': {
                    'enable_message_passing': True,
                    'message_passing': {'alpha': 0.9, 'iterations': 1}
                }
            },
            # Low alpha, high iterations
            {
                'llm': {'provider': 'mock'},
                'graph': {
                    'enable_message_passing': True,
                    'message_passing': {'alpha': 0.1, 'iterations': 10}
                }
            },
            # Different aggregation methods
            {
                'llm': {'provider': 'mock'},
                'graph': {
                    'enable_message_passing': True,
                    'message_passing': {
                        'alpha': 0.5,
                        'aggregation': 'max'
                    }
                }
            },
            # Low edge thresholds
            {
                'llm': {'provider': 'mock'},
                'graph': {
                    'enable_message_passing': True,
                    'edge_reevaluation': {
                        'new_edge_threshold': 0.3,
                        'similarity_threshold': 0.2
                    }
                }
            }
        ]
        
        for i, config in enumerate(config_variations):
            agent = MainAgent(config)
            agent.add_knowledge(f"Test knowledge {i}")
            result = agent.process_question(f"Test question {i}?")
            
            # Should not crash
            assert hasattr(result, 'response')
            assert hasattr(result, 'spike_detected')
    
    def _get_graph_nodes(self, result):
        """Helper to extract graph node count from result"""
        if hasattr(result, 'graph_analysis'):
            graph = result.graph_analysis.get('graph')
            if graph and hasattr(graph, 'num_nodes'):
                return graph.num_nodes
        return 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])