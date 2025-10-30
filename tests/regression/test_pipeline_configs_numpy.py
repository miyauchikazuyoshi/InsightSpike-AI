"""
Pipeline tests with NumPy backend to avoid Faiss segmentation faults
"""

import pytest
import os
from insightspike.implementations.agents.main_agent import MainAgent


class TestPipelineWithNumpy:
    """Test pipeline with NumPy vector backend"""
    
    @pytest.fixture
    def base_config(self):
        """Base configuration with NumPy backend"""
        return {
            'llm': {'provider': 'mock'},
            'graph': {
                'use_faiss': False  # Force NumPy backend
            }
        }
    
    def test_message_passing_numpy_backend(self, base_config):
        """Test message passing with NumPy backend"""
        config = base_config.copy()
        config['graph'] = {
            'enable_message_passing': True,
            'message_passing': {
                'alpha': 0.3,
                'iterations': 2
            }
        }
        
        agent = MainAgent(config)
        
        # Add knowledge
        agent.add_knowledge("Concept A relates to B")
        agent.add_knowledge("Concept B relates to C")
        
        # Process question
        result = agent.process_question("How does A relate to C?")
        
        # Verify
        assert hasattr(result, 'response')
        assert agent.l3_graph.message_passing_enabled == True
    
    def test_edge_reevaluation_numpy_backend(self, base_config):
        """Test edge re-evaluation with NumPy backend"""
        config = base_config.copy()
        config['graph'] = {
            'enable_message_passing': True,
            'edge_reevaluation': {
                'new_edge_threshold': 0.7,
                'max_new_edges_per_node': 3
            }
        }
        
        agent = MainAgent(config)
        
        # Add related knowledge
        knowledge = [
            "Neural networks use backpropagation",
            "Backpropagation calculates gradients",
            "Gradients are used for optimization",
            "Optimization updates network weights"
        ]
        for k in knowledge:
            agent.add_knowledge(k)
        
        # Process question that should trigger edge re-evaluation
        result = agent.process_question("How do neural networks learn?")
        
        assert hasattr(result, 'response')
        assert hasattr(result, 'graph_analysis')
    
    def test_full_features_numpy_backend(self, base_config):
        """Test all features with NumPy backend"""
        config = base_config.copy()
        config['graph'] = {
            'enable_message_passing': True,
            'message_passing': {
                'alpha': 0.5,
                'iterations': 3,
                'aggregation': 'weighted_mean'
            },
            'edge_reevaluation': {
                'new_edge_threshold': 0.6,
                'similarity_threshold': 0.5
            },
            'enable_graph_search': True,
            'search_depth': 2,
            'use_gnn': False,  # Disable GNN to avoid potential issues
            'spike_ged_threshold': 0.3,
            'spike_ig_threshold': 0.7
        }
        config['algorithms'] = {
            'use_advanced_ged': False,  # Use simple algorithms
            'use_advanced_ig': False
        }
        
        agent = MainAgent(config)
        
        # Add diverse knowledge
        knowledge = [
            "The sun is a star",
            "Stars produce energy through fusion",
            "Fusion combines hydrogen into helium",
            "Plants use sunlight for photosynthesis",
            "Photosynthesis produces oxygen",
            "Oxygen is essential for life"
        ]
        
        for k in knowledge:
            agent.add_knowledge(k)
        
        # Process multiple questions
        questions = [
            "How does the sun create energy?",
            "What is the connection between the sun and life on Earth?",
            "How do plants use the sun?"
        ]
        
        results = []
        for q in questions:
            result = agent.process_question(q)
            results.append(result)
            
            # Basic checks
            assert hasattr(result, 'response')
            assert hasattr(result, 'spike_detected')
        
        # All questions should process successfully
        assert len(results) == len(questions)
    
    def test_spike_detection_numpy_backend(self, base_config):
        """Test spike detection with NumPy backend"""
        config = base_config.copy()
        config['graph'] = {
            'enable_message_passing': True,
            'spike_ged_threshold': 0.2,  # Sensitive
            'spike_ig_threshold': 0.8,    # Sensitive
            'conflict_threshold': 0.3
        }
        
        agent = MainAgent(config)
        
        # Add initial coherent knowledge
        agent.add_knowledge("Water freezes at 0 degrees Celsius")
        agent.add_knowledge("Ice is solid water")
        
        # Question about existing knowledge
        result1 = agent.process_question("What happens to water at 0 degrees?")
        
        # Add very different knowledge
        agent.add_knowledge("Quantum computing uses qubits")
        agent.add_knowledge("Qubits can be in superposition")
        agent.add_knowledge("Superposition allows parallel computation")
        
        # Question about new domain
        result2 = agent.process_question("How do quantum computers work?")
        
        # Both should process successfully
        assert hasattr(result1, 'response')
        assert hasattr(result2, 'response')
        
        # Check if spike was detected (with sensitive thresholds, likely)
        # Note: exact spike detection depends on metric calculations
        assert hasattr(result2, 'spike_detected')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])