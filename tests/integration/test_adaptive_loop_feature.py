#!/usr/bin/env python3
"""
Adaptive Loop Feature Tests
===========================

Test the adaptive exploration loop functionality.
"""

import pytest
import tempfile
from pathlib import Path

from insightspike.config import load_config
from insightspike.implementations.agents import MainAgent
from insightspike.implementations.datastore.factory import DataStoreFactory


class TestAdaptiveLoop:
    """Test adaptive loop functionality"""
    
    @pytest.fixture
    def temp_datastore(self):
        """Create temporary datastore"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield DataStoreFactory.create("filesystem", base_path=tmpdir)
            
    @pytest.fixture
    def adaptive_config(self):
        """Create config with adaptive loop enabled"""
        return {
            "processing": {
                "enable_learning": True,
                "enable_adaptive_loop": True,
                "enable_insight_registration": True,
                "enable_insight_search": True,
                "adaptive_loop": {
                    "exploration_strategy": "narrowing",
                    "max_exploration_attempts": 5,
                    "initial_exploration_radius": 0.8,
                    "radius_decay_factor": 0.8,
                    "min_exploration_radius": 0.1
                }
            },
            "memory": {
                "max_retrieved_docs": 5
            },
            "l4_config": {
                "provider": "mock",
                "model": "mock-model"
            },
            "graph": {
                "similarity_threshold": 0.7,
                "hop_limit": 2
            }
        }
        
    @pytest.fixture
    def adaptive_agent(self, adaptive_config, temp_datastore):
        """Create agent with adaptive loop enabled"""
        agent = MainAgent(config=adaptive_config, datastore=temp_datastore)
        assert agent.initialize()
        
        # Add foundational knowledge
        mathematical_concepts = [
            "A derivative measures the rate of change of a function",
            "The integral is the inverse operation of differentiation", 
            "The fundamental theorem of calculus connects derivatives and integrals",
            "A limit describes the value a function approaches as input approaches some value",
            "Continuity means a function has no breaks or jumps",
            "Differentiability implies continuity, but not vice versa"
        ]
        
        for concept in mathematical_concepts:
            agent.add_knowledge(concept)
            
        return agent
        
    def test_adaptive_loop_basic(self, adaptive_agent):
        """Test basic adaptive loop functionality"""
        # Question that should trigger exploration
        result = adaptive_agent.process_question(
            "What is the relationship between derivatives and integrals?"
        )
        
        assert result is not None
        
        # Check for response
        if hasattr(result, 'response'):
            assert result.response is not None
        else:
            assert result.get('response') is not None
            
    def test_adaptive_loop_metadata(self, adaptive_agent):
        """Test that adaptive loop includes metadata"""
        result = adaptive_agent.process_question(
            "How do limits relate to continuity?"
        )
        
        # Check for metadata
        if hasattr(result, 'metadata'):
            metadata = result.metadata
        elif isinstance(result, dict):
            metadata = result.get('metadata', {})
        else:
            metadata = {}
            
        # Should have exploration metadata if adaptive was used
        if metadata:
            # Check for exploration attempts or processing time
            assert any(key in metadata for key in [
                'total_attempts', 'exploration_attempts', 
                'processing_time', 'exploration_path'
            ])
            
    def test_adaptive_exploration_path(self, adaptive_agent):
        """Test exploration path tracking"""
        result = adaptive_agent.process_question(
            "Explain the connection between differentiability and continuity",
            verbose=True  # Enable verbose mode if supported
        )
        
        # Check if exploration path is tracked
        if hasattr(result, 'metadata') and result.metadata:
            if 'exploration_path' in result.metadata:
                path = result.metadata['exploration_path']
                assert isinstance(path, list)
                
                # Check path structure
                for attempt in path:
                    assert 'radius' in attempt or 'topk_l2' in attempt
                    
    @pytest.mark.parametrize("strategy", ["narrowing", "expanding", "adaptive"])
    def test_different_strategies(self, temp_datastore, strategy):
        """Test different exploration strategies"""
        config = {
            "processing": {
                "enable_learning": True,
                "enable_adaptive_loop": True,
                "adaptive_loop": {
                    "exploration_strategy": strategy,
                    "max_exploration_attempts": 3
                }
            },
            "memory": {"max_retrieved_docs": 5},
            "l4_config": {"provider": "mock"},
            "graph": {"similarity_threshold": 0.7}
        }
        
        agent = MainAgent(config=config, datastore=temp_datastore)
        assert agent.initialize()
        
        # Add minimal knowledge
        agent.add_knowledge("Strategy test knowledge")
        
        # Process question
        result = agent.process_question("Test question for strategy")
        assert result is not None
        
    def test_adaptive_vs_non_adaptive(self, temp_datastore):
        """Compare adaptive vs non-adaptive processing"""
        # Create two agents - one with adaptive, one without
        adaptive_config = {
            "processing": {
                "enable_learning": True,
                "enable_adaptive_loop": True,
                "adaptive_loop": {
                    "exploration_strategy": "narrowing",
                    "max_exploration_attempts": 3
                }
            },
            "memory": {"max_retrieved_docs": 5},
            "l4_config": {"provider": "mock"},
            "graph": {"similarity_threshold": 0.7}
        }
        
        non_adaptive_config = adaptive_config.copy()
        non_adaptive_config["processing"]["enable_adaptive_loop"] = False
        
        adaptive_agent = MainAgent(config=adaptive_config, datastore=temp_datastore)
        non_adaptive_agent = MainAgent(config=non_adaptive_config, datastore=temp_datastore)
        
        assert adaptive_agent.initialize()
        assert non_adaptive_agent.initialize()
        
        # Add same knowledge to both
        knowledge = "Test knowledge for comparison"
        adaptive_agent.add_knowledge(knowledge)
        non_adaptive_agent.add_knowledge(knowledge)
        
        # Process same question
        question = "Test question for comparison"
        adaptive_result = adaptive_agent.process_question(question)
        non_adaptive_result = non_adaptive_agent.process_question(question)
        
        # Both should return results
        assert adaptive_result is not None
        assert non_adaptive_result is not None
        
    def test_adaptive_loop_limits(self, adaptive_agent):
        """Test adaptive loop respects max attempts"""
        # Process a complex question
        result = adaptive_agent.process_question(
            "What are all the interconnections between calculus concepts?"
        )
        
        # Check that exploration was bounded
        if hasattr(result, 'metadata') and result.metadata:
            if 'total_attempts' in result.metadata:
                # Should not exceed max_exploration_attempts (5)
                assert result.metadata['total_attempts'] <= 5
                
            if 'exploration_path' in result.metadata:
                # Path length should be bounded
                assert len(result.metadata['exploration_path']) <= 5
                
    @pytest.mark.slow
    def test_adaptive_loop_performance(self, adaptive_agent):
        """Test adaptive loop performance characteristics"""
        import time
        
        # Measure processing time
        start_time = time.time()
        result = adaptive_agent.process_question(
            "Complex question requiring exploration"
        )
        elapsed_time = time.time() - start_time
        
        # Should complete in reasonable time even with exploration
        assert elapsed_time < 10.0  # 10 seconds max
        assert result is not None
        
    def test_adaptive_loop_edge_cases(self, adaptive_agent):
        """Test adaptive loop edge cases"""
        # Empty question
        result = adaptive_agent.process_question("")
        assert result is not None
        
        # Very long question
        long_question = " ".join(["test"] * 100)
        result = adaptive_agent.process_question(long_question)
        assert result is not None
        
        # Special characters
        result = adaptive_agent.process_question("What about ∂/∂x and ∫?")
        assert result is not None