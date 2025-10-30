"""
Integration tests for July 2024 features
"""

import pytest
from insightspike.config import load_config
from insightspike.implementations.agents.main_agent import MainAgent


class TestJuly2024Features:
    """Test suite for all July 2024 feature integrations"""
    
    def test_layer1_bypass_integration(self):
        """Test Layer1 bypass with low uncertainty queries"""
        config = load_config(preset="production_optimized")
        agent = MainAgent(config)
        agent.initialize()
        
        # Add simple knowledge
        agent.add_knowledge("Water is H2O")
        
        # Query with low uncertainty
        result = agent.process_question("What is water?")
        
        assert result.success
        assert "H2O" in result.response
        # Should be fast (bypassed)
        
    def test_insight_registration_integration(self):
        """Test automatic insight registration on spike detection"""
        config = load_config(preset="experiment")
        config.processing.enable_insight_registration = True
        
        agent = MainAgent(config)
        agent.initialize()
        
        # Add knowledge that will trigger spike
        agent.add_knowledge("Energy cannot be created or destroyed")
        agent.add_knowledge("Mass and energy are interchangeable E=mc²")
        agent.add_knowledge("Conservation laws govern the universe")
        
        # Query that should trigger spike
        result = agent.process_question("What is the fundamental conservation principle?")
        
        # Check if insights were captured
        insights = agent.insight_registry.search_insights(["conservation", "energy"])
        assert len(insights) > 0
        
    def test_graph_search_integration(self):
        """Test graph-based memory search"""
        config = load_config(preset="graph_enhanced")
        agent = MainAgent(config)
        agent.initialize()
        
        # Add interconnected knowledge
        agent.add_knowledge("Neural networks process information")
        agent.add_knowledge("Information theory measures entropy")
        agent.add_knowledge("Entropy relates to disorder")
        
        # Query requiring multi-hop
        result = agent.process_question("How do neural networks relate to disorder?")
        
        # Should find connection through information → entropy → disorder
        assert result.success
        assert any(doc.get("hop", 0) > 0 for doc in result.retrieved_documents)
        
    def test_adaptive_learning_integration(self):
        """Test learning mechanism integration"""
        config = load_config(preset="adaptive_learning")
        agent = MainAgent(config)
        agent.initialize()
        
        # Add knowledge
        for i in range(5):
            agent.add_knowledge(f"Fact {i} about topic A")
        
        # Multiple similar queries
        initial_config = agent._get_config_snapshot()
        
        for i in range(3):
            agent.process_question("Tell me about topic A")
        
        # Check if parameters adapted
        final_config = agent._get_config_snapshot()
        
        # At least one parameter should have changed
        assert any(
            initial_config[k] != final_config[k] 
            for k in initial_config 
            if k in final_config
        )
        
    def test_mode_aware_prompts_integration(self):
        """Test prompt adaptation for different model sizes"""
        # Test minimal mode
        config_minimal = load_config(preset="experiment")
        agent_minimal = MainAgent(config_minimal)
        agent_minimal.initialize()
        
        # Test detailed mode
        config_detailed = load_config(preset="research")
        config_detailed.llm.prompt_style = "detailed"
        agent_detailed = MainAgent(config_detailed)
        agent_detailed.initialize()
        
        # Add same knowledge to both
        knowledge = ["Fact 1", "Fact 2", "Fact 3"]
        for k in knowledge:
            agent_minimal.add_knowledge(k)
            agent_detailed.add_knowledge(k)
        
        # Process same question
        question = "What are the facts?"
        
        result_minimal = agent_minimal.process_question(question)
        result_detailed = agent_detailed.process_question(question)
        
        # Both should succeed
        assert result_minimal.success
        assert result_detailed.success
        
        # Detailed should have more context
        # (Would need to inspect actual prompts to verify)
        
    @pytest.mark.parametrize("preset", [
        "production_optimized",
        "minimal", 
        "graph_enhanced",
        "adaptive_learning"
    ])
    def test_preset_compatibility(self, preset):
        """Test that all presets work correctly"""
        config = load_config(preset=preset)
        agent = MainAgent(config)
        
        # Should initialize without errors
        assert agent.initialize()
        
        # Should handle basic operations
        agent.add_knowledge("Test knowledge")
        result = agent.process_question("What do you know?")
        
        assert result.success
        assert isinstance(result.response, str)