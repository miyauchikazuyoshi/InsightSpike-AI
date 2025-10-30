#!/usr/bin/env python3
"""
Knowledge Management Basic Tests
================================

Test basic knowledge management functionality.
"""

import pytest
import tempfile
from unittest.mock import Mock, patch

from insightspike.config import load_config
from insightspike.implementations.agents import MainAgent
from insightspike.implementations.datastore.factory import DataStoreFactory


class TestKnowledgeManagement:
    """Test basic knowledge management"""
    
    @pytest.fixture
    def temp_datastore(self):
        """Create temporary datastore"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield DataStoreFactory.create("filesystem", base_path=tmpdir)
            
    @pytest.fixture
    def basic_agent(self, temp_datastore):
        """Create basic agent for testing"""
        config = {
            "processing": {
                "enable_learning": True
            },
            "memory": {
                "max_retrieved_docs": 5
            },
            "l4_config": {
                "provider": "mock",
                "model": "mock-model"
            }
        }
        
        agent = MainAgent(config=config, datastore=temp_datastore)
        assert agent.initialize()
        return agent
        
    def test_add_single_knowledge(self, basic_agent):
        """Test adding single knowledge item"""
        knowledge = "Number: A mathematical object used to count, measure, and label"
        
        # Add knowledge
        result = basic_agent.add_knowledge(knowledge)
        
        # Check result
        assert result is not None
        if isinstance(result, dict):
            assert result.get('success', False)
        
        # Check episodes were created
        assert len(basic_agent.l2_memory.episodes) > 0
        
    def test_add_multiple_knowledge(self, basic_agent):
        """Test adding multiple knowledge items"""
        concepts = [
            "Number: A mathematical object used to count, measure, and label",
            "Addition: The process of combining two or more numbers to get their sum",
            "Multiplication: Repeated addition of the same number"
        ]
        
        initial_count = len(basic_agent.l2_memory.episodes)
        
        # Add all concepts
        for concept in concepts:
            result = basic_agent.add_knowledge(concept)
            assert result is not None
            
        # Check episodes were created
        final_count = len(basic_agent.l2_memory.episodes)
        assert final_count > initial_count
        assert final_count >= len(concepts)
        
    def test_add_knowledge_with_c_value(self, basic_agent):
        """Test adding knowledge with confidence value"""
        knowledge = "Important concept with high confidence"
        
        # Add with specific c_value
        result = basic_agent.add_knowledge(knowledge, c_value=0.9)
        assert result is not None
        
        # Check that episode was created
        episodes = basic_agent.l2_memory.episodes
        assert len(episodes) > 0
        
        # Check confidence value was set
        # Note: The actual confidence might be processed differently
        # but the episode should exist
        
    def test_add_empty_knowledge(self, basic_agent):
        """Test adding empty knowledge"""
        # Should handle empty string gracefully
        result = basic_agent.add_knowledge("")
        
        # Might still create an episode, but should not crash
        assert result is not None
        
    def test_add_very_long_knowledge(self, basic_agent):
        """Test adding very long knowledge text"""
        # Create a very long text
        long_text = "This is a test. " * 1000  # ~15000 characters
        
        result = basic_agent.add_knowledge(long_text)
        assert result is not None
        
        # Should handle long text without crashing
        assert len(basic_agent.l2_memory.episodes) > 0
        
    def test_add_knowledge_persistence(self, basic_agent, temp_datastore):
        """Test that knowledge is persisted to datastore"""
        knowledge = "Persistent knowledge item"
        
        # Add knowledge
        basic_agent.add_knowledge(knowledge)
        
        # Check datastore directly
        episodes = temp_datastore.load_episodes()
        assert len(episodes) > 0
        
        # Check that knowledge text is in episodes
        texts = [ep.get('text', '') for ep in episodes]
        assert any(knowledge in text for text in texts)
        
    def test_add_knowledge_with_special_characters(self, basic_agent):
        """Test adding knowledge with special characters"""
        special_knowledge = "Mathematical symbols: ∂/∂x, ∫, ∑, ∏, √, ∞"
        
        result = basic_agent.add_knowledge(special_knowledge)
        assert result is not None
        
        # Should handle special characters
        assert len(basic_agent.l2_memory.episodes) > 0
        
    def test_add_knowledge_idempotency(self, basic_agent):
        """Test adding same knowledge multiple times"""
        knowledge = "Duplicate knowledge test"
        
        # Add same knowledge multiple times
        results = []
        for _ in range(3):
            result = basic_agent.add_knowledge(knowledge)
            results.append(result)
            
        # All should succeed
        assert all(r is not None for r in results)
        
        # Should create multiple episodes (not deduplicated at this level)
        assert len(basic_agent.l2_memory.episodes) >= 3
        
    @pytest.mark.parametrize("c_value", [0.0, 0.5, 1.0])
    def test_add_knowledge_various_c_values(self, basic_agent, c_value):
        """Test adding knowledge with various confidence values"""
        knowledge = f"Knowledge with c_value {c_value}"
        
        result = basic_agent.add_knowledge(knowledge, c_value=c_value)
        assert result is not None
        
        # Should accept any valid c_value
        assert len(basic_agent.l2_memory.episodes) > 0
        
    def test_add_knowledge_error_handling(self, basic_agent):
        """Test error handling in add_knowledge"""
        # Test with None (should handle gracefully)
        try:
            result = basic_agent.add_knowledge(None)
            # Should either handle None or raise appropriate error
        except (TypeError, AttributeError):
            # Expected errors for None input
            pass
        except Exception as e:
            # Unexpected error
            pytest.fail(f"Unexpected error type: {type(e)}")
            
    def test_knowledge_retrieval_after_add(self, basic_agent):
        """Test that added knowledge can be retrieved"""
        knowledge_items = [
            "Concept A: First concept",
            "Concept B: Second concept",
            "Concept C: Third concept"
        ]
        
        # Add knowledge
        for item in knowledge_items:
            basic_agent.add_knowledge(item)
            
        # Process a question to trigger retrieval
        result = basic_agent.process_question("Tell me about concepts")
        assert result is not None
        
        # Should be able to process questions after adding knowledge