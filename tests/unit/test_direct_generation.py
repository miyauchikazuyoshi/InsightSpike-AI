"""
Tests for Layer 4 direct generation mode
========================================

Tests the new direct response generation capability without LLM.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from insightspike.core.layers.layer4_llm_provider import LocalProvider
from insightspike.core.layers.layer4_prompt_builder import L4PromptBuilder


class TestDirectGeneration:
    """Test direct generation mode in Layer 4."""

    def test_direct_generation_disabled_by_default(self):
        """Test that direct generation is disabled by default."""
        provider = LocalProvider()

        # Mock the initialization
        provider._initialized = True
        provider._generate_sync = Mock(return_value="LLM response")

        context = {"reasoning_quality": 0.9}
        response = provider.generate_response(str(context), "Test question")

        # Should use LLM, not direct generation
        provider._generate_sync.assert_called_once()
        assert isinstance(response, str)

    def test_direct_generation_enabled_high_quality(self):
        """Test direct generation with high reasoning quality."""
        # Create config with direct generation enabled
        config = Mock()
        config.llm.use_direct_generation = True
        config.llm.direct_generation_threshold = 0.7

        provider = LocalProvider(config)
        provider._initialized = True
        provider._generate_sync = Mock(return_value="LLM response")

        context = {
            "reasoning_quality": 0.8,  # Above threshold
            "retrieved_documents": [
                {"text": "Test document 1.", "c_value": 0.9},
                {"text": "Test document 2.", "c_value": 0.8},
            ],
            "graph_analysis": {
                "spike_detected": True,
                "metrics": {"delta_ged": -0.5, "delta_ig": 0.3},
            },
        }

        # Use detailed method to get full response with metadata
        response = provider.generate_response_detailed(context, "Test question")

        # Should NOT call LLM
        provider._generate_sync.assert_not_called()

        # Should have direct generation flag
        assert response["direct_generation"] is True
        assert response["confidence"] == 0.8
        assert "🧠 **INSIGHT SPIKE DETECTED**" in response["response"]
        assert "ΔGED" in response["response"]

    def test_direct_generation_low_quality_uses_llm(self):
        """Test that low quality falls back to LLM."""
        config = Mock()
        config.llm.use_direct_generation = True
        config.llm.direct_generation_threshold = 0.7

        provider = LocalProvider(config)
        provider._initialized = True
        provider._generate_sync = Mock(return_value="LLM response")

        context = {"reasoning_quality": 0.5}  # Below threshold
        response = provider.generate_response(str(context), "Test question")

        # Should use LLM
        provider._generate_sync.assert_called_once()
        assert isinstance(response, str)

    def test_prompt_builder_direct_response(self):
        """Test PromptBuilder's direct response generation."""
        builder = L4PromptBuilder()

        context = {
            "retrieved_documents": [
                {"text": "The sky is blue due to Rayleigh scattering.", "c_value": 0.9},
                {"text": "Light wavelengths affect color perception.", "c_value": 0.8},
            ],
            "graph_analysis": {
                "spike_detected": False,
                "metrics": {"delta_ged": -0.1, "delta_ig": 0.1},
            },
            "reasoning_quality": 0.75,
        }

        response = builder.build_direct_response(context, "Why is the sky blue?")

        assert "## Answer" in response
        assert "The sky is blue due to Rayleigh scattering." in response
        assert "light wavelengths affect color perception." in response
        assert "**Confidence Level**:" in response
        assert "Medium Confidence" in response

    def test_direct_response_with_no_documents(self):
        """Test direct response when no documents are retrieved."""
        builder = L4PromptBuilder()

        context = {
            "retrieved_documents": [],
            "graph_analysis": {},
            "reasoning_quality": 0.8,
        }

        response = builder.build_direct_response(context, "Test question")

        assert "Based on the analysis of the knowledge graph structure" in response
        assert "**Confidence Level**:" in response

    def test_direct_response_with_error_handling(self):
        """Test error handling in direct response generation."""
        builder = L4PromptBuilder()

        # Malformed context
        context = None

        response = builder.build_direct_response(context, "Test question")

        assert "Unable to generate a direct response" in response
        assert "Please try using standard LLM generation mode" in response

    def test_direct_generation_not_available_for_streaming(self):
        """Test that direct generation is not used for streaming requests."""
        config = Mock()
        config.llm.use_direct_generation = True
        config.llm.direct_generation_threshold = 0.7

        provider = LocalProvider(config)
        provider._initialized = True
        provider._generate_streaming = Mock(
            return_value=iter(["Streaming", " response"])
        )

        context = {"reasoning_quality": 0.9}  # High quality
        response = provider.generate_response_detailed(
            context, "Test question", streaming=True
        )

        # Should use streaming LLM, not direct generation
        provider._generate_streaming.assert_called_once()
        assert response["streaming"] is True
        assert "direct_generation" not in response
