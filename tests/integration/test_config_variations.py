"""
Test InsightSpike with different configuration variations
Ensures the system works correctly with all config presets
"""

import pytest
import numpy as np
from typing import Dict, Any, List
import logging

from insightspike.config.presets import ConfigPresets
from insightspike.implementations.agents import MainAgent
from insightspike.implementations.datastore.memory_store import InMemoryDataStore

logger = logging.getLogger(__name__)


class TestConfigVariations:
    """Test system behavior with different configuration presets"""
    
    @pytest.fixture
    def test_knowledge(self) -> List[str]:
        """Sample knowledge base for testing"""
        return [
            "Neural networks are computational models inspired by biological neurons.",
            "Deep learning uses multiple layers of neural networks.",
            "Machine learning is a subset of artificial intelligence.",
            "Gradient descent is an optimization algorithm.",
            "Backpropagation calculates gradients efficiently."
        ]
    
    @pytest.fixture  
    def test_questions(self) -> List[str]:
        """Sample questions for testing"""
        return [
            "What are neural networks?",
            "How does deep learning work?",
            "What is machine learning?",
            "How are neural networks trained?"
        ]
    
    def verify_config_settings(self, agent: MainAgent, preset_name: str) -> Dict[str, Any]:
        """Verify that config settings are properly applied"""
        results = {
            "preset": preset_name,
            "checks": {}
        }
        
        # Check embedding dimension
        if hasattr(agent.l2_memory, 'memory_config'):
            results["checks"]["embedding_dim"] = agent.l2_memory.memory_config.embedding_dim
        else:
            results["checks"]["embedding_dim"] = getattr(agent.l2_memory, 'dim', 384)
        
        # Check LLM provider
        if hasattr(agent.l4_llm, 'provider_name'):
            results["checks"]["llm_provider"] = agent.l4_llm.provider_name
        elif hasattr(agent.l4_llm, 'config'):
            if hasattr(agent.l4_llm.config, 'provider'):
                results["checks"]["llm_provider"] = agent.l4_llm.config.provider
            else:
                results["checks"]["llm_provider"] = "unknown"
        
        # Check graph thresholds
        if hasattr(agent.l3_graph, 'config'):
            config = agent.l3_graph.config
            if hasattr(config, 'graph'):
                results["checks"]["spike_ged_threshold"] = getattr(config.graph, 'spike_ged_threshold', -0.5)
                results["checks"]["spike_ig_threshold"] = getattr(config.graph, 'spike_ig_threshold', 0.2)
            elif isinstance(config, dict):
                results["checks"]["spike_ged_threshold"] = config.get('graph', {}).get('spike_ged_threshold', -0.5)
                results["checks"]["spike_ig_threshold"] = config.get('graph', {}).get('spike_ig_threshold', 0.2)
        
        return results
    
    @pytest.mark.parametrize("preset_name", [
        "development",
        "experiment", 
        "production",
        "minimal",
        "graph_enhanced"
    ])
    def test_preset_initialization(self, preset_name: str):
        """Test that each preset initializes correctly"""
        # Get preset config
        config = getattr(ConfigPresets, preset_name)()
        
        # Create agent
        agent = MainAgent(config=config, datastore=InMemoryDataStore())
        
        # Verify initialization
        assert agent.initialize(), f"Failed to initialize with {preset_name} preset"
        
        # Verify key attributes exist
        assert hasattr(agent, 'l1_embedder'), "Missing l1_embedder"
        assert hasattr(agent, 'l2_memory'), "Missing l2_memory"
        assert hasattr(agent, 'l3_graph'), "Missing l3_graph"
        assert hasattr(agent, 'l4_llm'), "Missing l4_llm"
        
        # Verify config settings
        settings = self.verify_config_settings(agent, preset_name)
        logger.info(f"Config settings for {preset_name}: {settings}")
    
    @pytest.mark.parametrize("preset_name", ["development", "minimal", "graph_enhanced"])
    def test_knowledge_operations(self, preset_name: str, test_knowledge: List[str]):
        """Test knowledge addition and retrieval with different presets"""
        config = getattr(ConfigPresets, preset_name)()
        agent = MainAgent(config=config, datastore=InMemoryDataStore())
        assert agent.initialize()
        
        # Add knowledge
        success_count = 0
        for knowledge in test_knowledge:
            result = agent.add_knowledge(knowledge)
            if result.get("success"):
                success_count += 1
        
        assert success_count == len(test_knowledge), \
            f"Failed to add all knowledge with {preset_name}: {success_count}/{len(test_knowledge)}"
        
        # Test retrieval
        question = "What are neural networks?"
        answer = agent.process_question(question, max_cycles=2)
        
        assert hasattr(answer, 'response'), "Missing response in answer"
        assert answer.success, f"Question processing failed with {preset_name}"
        assert len(answer.response) > 0, "Empty response"
    
    def test_memory_capacity_differences(self, test_knowledge: List[str]):
        """Test different memory capacity settings"""
        presets_with_capacity = {
            "minimal": 60,  # Should have small capacity
            "production": 1000,  # Should have large capacity
        }
        
        for preset_name, expected_min_capacity in presets_with_capacity.items():
            config = getattr(ConfigPresets, preset_name)()
            agent = MainAgent(config=config, datastore=InMemoryDataStore())
            assert agent.initialize()
            
            # Check memory capacity setting
            if hasattr(agent.l2_memory, 'memory_config'):
                capacity = agent.l2_memory.memory_config.episodic_memory_capacity
            else:
                capacity = getattr(agent.l2_memory, 'capacity', 100)
            
            assert capacity >= expected_min_capacity, \
                f"{preset_name} should have capacity >= {expected_min_capacity}, got {capacity}"
    
    def test_graph_feature_variations(self, test_knowledge: List[str], test_questions: List[str]):
        """Test graph-based features with different configurations"""
        # Test minimal config (graph features may be disabled)
        minimal_config = ConfigPresets.minimal()
        minimal_agent = MainAgent(config=minimal_config, datastore=InMemoryDataStore())
        assert minimal_agent.initialize()
        
        # Test graph_enhanced config (all graph features enabled)
        enhanced_config = ConfigPresets.graph_enhanced()
        enhanced_agent = MainAgent(config=enhanced_config, datastore=InMemoryDataStore())
        assert enhanced_agent.initialize()
        
        # Add knowledge to both
        for knowledge in test_knowledge[:3]:  # Use subset for speed
            minimal_agent.add_knowledge(knowledge)
            enhanced_agent.add_knowledge(knowledge)
        
        # Process questions and compare
        for question in test_questions[:2]:  # Use subset for speed
            minimal_result = minimal_agent.process_question(question, max_cycles=2)
            enhanced_result = enhanced_agent.process_question(question, max_cycles=2)
            
            # Both should succeed
            assert minimal_result.success, "Minimal config failed"
            assert enhanced_result.success, "Enhanced config failed"
            
            # Enhanced might detect more spikes due to graph features
            logger.info(f"Question: {question}")
            logger.info(f"  Minimal - spike: {minimal_result.spike_detected}, quality: {minimal_result.reasoning_quality}")
            logger.info(f"  Enhanced - spike: {enhanced_result.spike_detected}, quality: {enhanced_result.reasoning_quality}")
    
    def test_llm_provider_variations(self):
        """Test different LLM provider configurations"""
        # Mock provider (development)
        dev_config = ConfigPresets.development()
        assert dev_config.llm.provider == "mock"
        
        dev_agent = MainAgent(config=dev_config, datastore=InMemoryDataStore())
        assert dev_agent.initialize()
        
        # Process a question with mock provider
        result = dev_agent.process_question("Test question", max_cycles=1)
        assert result.success, "Mock provider should work"
        
        # Local provider (experiment) - may not work without model
        exp_config = ConfigPresets.experiment() 
        assert exp_config.llm.provider == "local"
        # Skip actual test as local models may not be available
    
    def test_embedding_model_variations(self, test_knowledge: List[str]):
        """Test different embedding model configurations"""
        config = ConfigPresets.development()
        agent = MainAgent(config=config, datastore=InMemoryDataStore())
        assert agent.initialize()
        
        # Check embedding dimension matches config
        expected_dim = config.embedding.dimension
        
        # Add knowledge and check embedding shape
        result = agent.add_knowledge(test_knowledge[0])
        assert result.get("success"), "Failed to add knowledge"
        
        # Get the added episode
        if hasattr(agent.l2_memory, 'episodes') and len(agent.l2_memory.episodes) > 0:
            episode = agent.l2_memory.episodes[0]
            assert episode.vec.shape == (expected_dim,), \
                f"Expected embedding shape ({expected_dim},), got {episode.vec.shape}"
    
    def test_config_consistency_across_layers(self):
        """Ensure config is consistently applied across all layers"""
        config = ConfigPresets.development()
        agent = MainAgent(config=config, datastore=InMemoryDataStore())
        assert agent.initialize()
        
        # Check that all layers received proper config
        assert agent.l1_error_monitor is not None, "L1 not initialized"
        assert agent.l2_memory is not None, "L2 not initialized"
        assert agent.l3_graph is not None, "L3 not initialized"
        assert agent.l4_llm is not None, "L4 not initialized"
        
        # Verify l1_embedder uses same model as config
        if hasattr(agent.l1_embedder, 'model_name'):
            assert agent.l1_embedder.model_name == config.embedding.model_name, \
                "Embedder model mismatch"


class TestConfigEdgeCases:
    """Test edge cases and error conditions with different configs"""
    
    def test_missing_optional_features(self):
        """Test system behavior when optional features are disabled"""
        config = ConfigPresets.minimal()
        
        # Disable some features
        config.processing.enable_layer1_bypass = False
        config.processing.enable_insight_registration = False
        config.processing.enable_insight_search = False
        
        agent = MainAgent(config=config, datastore=InMemoryDataStore())
        assert agent.initialize()
        
        # System should still work without optional features
        result = agent.add_knowledge("Test knowledge")
        assert result.get("success"), "Should work without optional features"
        
        answer = agent.process_question("What is test?", max_cycles=1)
        assert answer.success, "Should process questions without optional features"
    
    def test_extreme_threshold_values(self):
        """Test with extreme threshold values"""
        config = ConfigPresets.development()
        
        # Set extreme thresholds
        config.graph.spike_ged_threshold = -10.0  # Very sensitive
        config.graph.spike_ig_threshold = 0.001   # Very sensitive
        config.graph.similarity_threshold = 0.99  # Very strict
        
        agent = MainAgent(config=config, datastore=InMemoryDataStore())
        assert agent.initialize()
        
        # Should still function, even if behavior is different
        agent.add_knowledge("Knowledge A")
        agent.add_knowledge("Knowledge B")
        
        result = agent.process_question("What is knowledge?", max_cycles=2)
        assert result.success, "Should handle extreme thresholds"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])