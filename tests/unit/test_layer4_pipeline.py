"""
Tests for Layer 4 Pipeline Architecture
======================================

Tests the new Layer 4 (PromptBuilder) + Layer 4.1 (LLM Polish) pipeline.
"""

import pytest
from unittest.mock import Mock, patch
from insightspike.core.layers.layer4_pipeline import Layer4Pipeline
from insightspike.core.layers.layer4_prompt_builder import L4PromptBuilder
from insightspike.core.layers.layer4_1_llm_polish import L4_1LLMPolish


class TestLayer4Pipeline:
    """Test the integrated Layer 4 pipeline."""
    
    def test_pipeline_initialization(self):
        """Test pipeline initializes correctly."""
        pipeline = Layer4Pipeline()
        
        assert isinstance(pipeline.layer4, L4PromptBuilder)
        assert isinstance(pipeline.layer4_1, L4_1LLMPolish)
        assert pipeline.enable_polish is True
        assert pipeline.force_direct is False
    
    def test_direct_generation_mode(self):
        """Test direct generation through pipeline."""
        config = Mock()
        config.llm.use_direct_generation = True
        config.llm.direct_generation_threshold = 0.7
        config.llm.enable_polish = False
        
        pipeline = Layer4Pipeline(config)
        pipeline.enable_polish = False
        
        context = {
            "reasoning_quality": 0.8,
            "retrieved_documents": [{"text": "Test doc", "c_value": 0.9}],
            "graph_analysis": {"spike_detected": True}
        }
        
        result = pipeline.generate_response(context, "Test question", mode="direct")
        
        assert result["success"] is True
        assert "response" in result
        assert result["pipeline"]["layer4_mode"] == "direct"
        assert result["pipeline"]["polish_applied"] is False
    
    def test_prompt_mode(self):
        """Test prompt generation mode."""
        pipeline = Layer4Pipeline()
        
        context = {
            "reasoning_quality": 0.3,
            "retrieved_documents": []
        }
        
        result = pipeline.generate_response(context, "Test question", mode="prompt")
        
        assert result["success"] is True
        assert "response" in result
        assert result["pipeline"]["layer4_mode"] == "prompt"
    
    @pytest.mark.skip(reason="Need to fix config mock behavior")
    def test_auto_mode_selection(self):
        """Test automatic mode selection based on quality."""
        config = Mock()
        config.llm.use_direct_generation = True
        config.llm.direct_generation_threshold = 0.7
        config.llm.enable_polish = True
        config.llm.polish_threshold = 0.6
        config.llm.always_polish_below = 0.4
        
        pipeline = Layer4Pipeline(config)
        
        # High quality -> direct mode
        high_quality_context = {"reasoning_quality": 0.85}
        result = pipeline.generate_response(high_quality_context, "Test")
        assert result["pipeline"]["layer4_mode"] == "direct"
        
        # For low quality, direct generation should not be used
        # But since config.llm.use_direct_generation is True and mode is not forced,
        # it will still try direct mode but with low confidence
        low_quality_context = {"reasoning_quality": 0.3}  # Well below threshold
        
        # Explicitly disable direct generation for low quality test
        config2 = Mock()
        config2.llm.use_direct_generation = False  # Disable direct generation
        config2.llm.enable_polish = True
        config2.llm.polish_threshold = 0.6
        config2.llm.always_polish_below = 0.4
        pipeline2 = Layer4Pipeline(config2)
        
        result = pipeline2.generate_response(low_quality_context, "Test")
        # Now it should use prompt mode
        assert result["pipeline"]["layer4_mode"] == "prompt"
    
    def test_polish_application(self):
        """Test Layer 4.1 polish is applied when appropriate."""
        config = Mock()
        config.llm.use_direct_generation = True
        config.llm.direct_generation_threshold = 0.5
        config.llm.enable_polish = True
        config.llm.polish_threshold = 0.6
        config.llm.always_polish_below = 0.4
        
        pipeline = Layer4Pipeline(config)
        
        # Mock the polish method
        with patch.object(pipeline.layer4_1, 'polish') as mock_polish:
            mock_polish.return_value = "Polished text"
            
            # Low confidence should trigger polish
            context = {"reasoning_quality": 0.35}
            result = pipeline.generate_response(context, "Test", mode="direct")
            
            # Polish should be called for low confidence
            mock_polish.assert_called_once()
            assert result["pipeline"]["polish_applied"] is True
    
    def test_pipeline_modes(self):
        """Test different pipeline modes."""
        pipeline = Layer4Pipeline()
        
        # Test direct mode
        pipeline.set_mode("direct")
        assert pipeline.force_direct is True
        assert pipeline.enable_polish is True
        
        # Test prompt mode
        pipeline.set_mode("prompt")
        assert pipeline.force_direct is False
        assert pipeline.enable_polish is False
        
        # Test no polish mode
        pipeline.set_mode("no_polish")
        assert pipeline.enable_polish is False
        
        # Test polish all mode
        pipeline.set_mode("polish_all")
        assert pipeline.enable_polish is True
        assert pipeline.layer4_1.polish_threshold == 1.0
        
        # Test auto mode
        pipeline.set_mode("auto")
        assert pipeline.force_direct is False
        assert pipeline.enable_polish is True
    
    def test_get_stats(self):
        """Test pipeline statistics."""
        config = Mock()
        config.llm.use_direct_generation = True
        config.llm.direct_generation_threshold = 0.75
        
        pipeline = Layer4Pipeline(config)
        stats = pipeline.get_stats()
        
        assert stats["layer4_config"]["direct_generation_enabled"] is True
        assert stats["layer4_config"]["threshold"] == 0.75
        assert "polish_enabled" in stats["layer4_1_config"]
    
    def test_backward_compatibility(self):
        """Test backward compatibility with old interface."""
        pipeline = Layer4Pipeline()
        
        context = {"reasoning_quality": 0.5}
        result = pipeline.generate_response(context, "Test question")
        
        # Old interface expects 'response' key
        assert "response" in result
        assert result["success"] is True
        
        # New interface provides 'output' key
        assert result["response"] == result.get("output", "")