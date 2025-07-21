"""Test that Query Transformation insights are included in prompts"""

from unittest.mock import Mock

import pytest

from insightspike.config.models import LLMConfig
from insightspike.implementations.layers.layer4_llm_interface import L4LLMInterface


class TestQueryInsightsInPrompt:
    """Test Query Transformation insights inclusion in prompts"""

    def setup_method(self):
        """Setup test environment"""
        self.config = LLMConfig(
            provider="mock",
            model="test",
            prompt_style="standard",
            include_metadata=True,
        )
        self.llm = L4LLMInterface(config=self.config)
        self.llm.initialize()

    def test_insights_included_in_standard_prompt(self):
        """Test that insights are included in standard prompts"""
        # Create mock query state with insights
        query_state = Mock()
        query_state.insights_discovered = [
            "Thermodynamics and information theory share mathematical structure",
            "Entropy bridges physical and informational domains",
            "Maxwell's demon connects computation and physics",
        ]
        query_state.absorbed_concepts = [
            "entropy",
            "information",
            "reversibility",
            "computation",
            "energy",
            "disorder",
        ]

        # Create context with query state
        context = {
            "retrieved_documents": [
                {"text": "Entropy measures disorder in thermodynamics"},
                {"text": "Shannon entropy quantifies information content"},
            ],
            "query_state": query_state,
        }

        # Build prompt
        prompt = self.llm._build_prompt(context, "How are entropy concepts related?")

        # Verify insights are included
        assert "[Discovered Insights from Query Evolution]" in prompt
        assert (
            "Thermodynamics and information theory share mathematical structure"
            in prompt
        )
        assert "Entropy bridges physical and informational domains" in prompt
        assert "Maxwell's demon connects computation and physics" in prompt

        # Verify absorbed concepts are included
        assert "[Key Concepts Absorbed]" in prompt
        assert "entropy, information, reversibility, computation, energy" in prompt

    def test_insights_included_in_simple_prompt(self):
        """Test that insights are included in simple prompts for lightweight models"""
        # Configure for simple prompt
        self.llm.config.use_simple_prompt = True

        # Create mock query state
        query_state = Mock()
        query_state.insights_discovered = [
            "Entropy connects physics and information theory fundamentally"
        ]
        query_state.absorbed_concepts = ["entropy", "information"]

        # Create context
        context = {
            "retrieved_documents": [
                {"text": "Entropy is a fundamental concept in physics"}
            ],
            "query_state": query_state,
        }

        # Build prompt
        prompt = self.llm._build_simple_prompt(context, "What is entropy?")

        # Verify insight is included (truncated for lightweight models)
        assert "[Entropy connects physics and information theory fund" in prompt

    def test_no_insights_when_query_state_missing(self):
        """Test graceful handling when query_state is not present"""
        context = {"retrieved_documents": [{"text": "Basic document about entropy"}]}

        # Should not raise error
        prompt = self.llm._build_prompt(context, "What is entropy?")

        # Should not contain insight sections
        assert "[Discovered Insights" not in prompt
        assert "[Key Concepts Absorbed]" not in prompt

    def test_empty_insights_handled_gracefully(self):
        """Test handling of empty insights list"""
        query_state = Mock()
        query_state.insights_discovered = []
        query_state.absorbed_concepts = []

        context = {
            "retrieved_documents": [{"text": "Document"}],
            "query_state": query_state,
        }

        prompt = self.llm._build_prompt(context, "Question?")

        # Should not include empty sections
        assert "[Discovered Insights" not in prompt
        assert "[Key Concepts Absorbed]" not in prompt

    def test_insight_truncation(self):
        """Test that insights are properly truncated"""
        query_state = Mock()
        # Create many insights
        query_state.insights_discovered = [
            f"Insight {i}: " + "x" * 100 for i in range(10)
        ]
        query_state.absorbed_concepts = [f"concept_{i}" for i in range(20)]

        context = {"retrieved_documents": [], "query_state": query_state}

        prompt = self.llm._build_prompt(context, "Question?")

        # Should only include first 3 insights
        assert "Insight 0:" in prompt
        assert "Insight 1:" in prompt
        assert "Insight 2:" in prompt
        assert "Insight 3:" not in prompt

        # Should only include first 5 concepts
        assert "concept_0" in prompt
        assert "concept_4" in prompt
        assert "concept_5" not in prompt
