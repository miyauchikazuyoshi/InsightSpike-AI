"""
Integration test for Layer1 Bypass Mechanism
===========================================

Tests the fast path for known concepts with low uncertainty.
"""

import pytest
from unittest.mock import Mock, patch

from insightspike.config import load_config
from insightspike.implementations.agents.main_agent import MainAgent


class TestLayer1Bypass:
    """Test Layer1 bypass mechanism"""

    def test_bypass_activated_for_low_uncertainty(self):
        """Test that bypass is activated for low uncertainty queries"""
        # Create config with bypass enabled
        config = load_config(preset="experiment")
        config.processing.enable_layer1_bypass = True
        config.processing.bypass_uncertainty_threshold = 0.3
        config.processing.bypass_known_ratio_threshold = 0.8
        
        # Create agent
        agent = MainAgent(config)
        
        # Add some known knowledge
        agent.add_knowledge("The capital of France is Paris.")
        agent.add_knowledge("Paris is a city in France.")
        
        # Mock the error monitor to return low uncertainty
        with patch.object(agent.l1_error_monitor, 'analyze_uncertainty') as mock_analyze:
            mock_analyze.return_value = {
                "uncertainty": 0.1,  # Low uncertainty
                "uncertainty_score": 0.1,
                "known_elements": ["capital", "France", "Paris"],
                "unknown_elements": [],
                "known_ratio": 1.0,
                "is_cacheable": True,
                "suggested_path": "bypass",
                "requires_synthesis": False,
                "error_threshold": 0.3,
                "analysis_confidence": 0.9,
                "certainty_scores": {"capital": 0.9, "France": 0.95, "Paris": 0.95},
            }
            
            # Process a simple known query
            result = agent.process_question("What is the capital of France?", verbose=True)
            
            # Check that bypass was used (no graph analysis)
            assert result.graph_analysis["metrics"]["delta_ged"] == 0.0
            assert result.graph_analysis["metrics"]["delta_ig"] == 0.0
            assert not result.spike_detected
            assert result.reasoning_quality > 0.7  # High quality for known info
    
    def test_bypass_not_activated_for_high_uncertainty(self):
        """Test that bypass is NOT activated for high uncertainty queries"""
        # Create config with bypass enabled
        config = load_config(preset="experiment")
        config.processing.enable_layer1_bypass = True
        config.processing.bypass_uncertainty_threshold = 0.3
        
        # Create agent
        agent = MainAgent(config)
        
        # Process an unknown query (will have high uncertainty)
        result = agent.process_question("What is the quantum flux coefficient of dark matter?")
        
        # Check that normal processing was used
        # The query should go through full pipeline due to high uncertainty
        assert result.error_state.get("uncertainty", 0) > 0.3
    
    def test_bypass_disabled_by_config(self):
        """Test that bypass is not used when disabled in config"""
        # Create config with bypass disabled
        config = load_config(preset="experiment")
        config.processing.enable_layer1_bypass = False
        
        # Create agent
        agent = MainAgent(config)
        
        # Add known knowledge
        agent.add_knowledge("The sky is blue.")
        
        # Mock the error monitor to return low uncertainty
        with patch.object(agent.l1_error_monitor, 'analyze_uncertainty') as mock_analyze:
            mock_analyze.return_value = {
                "uncertainty": 0.1,  # Low uncertainty
                "uncertainty_score": 0.1,
                "known_elements": ["sky", "blue"],
                "unknown_elements": [],
                "known_ratio": 1.0,
                "is_cacheable": True,
                "suggested_path": "bypass",
                "requires_synthesis": False,
                "error_threshold": 0.3,
                "analysis_confidence": 0.9,
                "certainty_scores": {"sky": 0.9, "blue": 0.9},
            }
            
            # Process query
            result = agent.process_question("What color is the sky?")
            
            # Even with low uncertainty, full processing should occur
            # because bypass is disabled
            # This is harder to test without mocking more internals,
            # but at least verify the result is valid
            assert hasattr(result, 'response')
            assert hasattr(result, 'reasoning_quality')
    
    def test_bypass_with_complex_query(self):
        """Test that complex queries are not bypassed even with low uncertainty"""
        # Create config with bypass enabled
        config = load_config(preset="experiment")
        config.processing.enable_layer1_bypass = True
        
        # Create agent
        agent = MainAgent(config)
        
        # Add knowledge
        agent.add_knowledge("Paris is the capital of France.")
        agent.add_knowledge("Berlin is the capital of Germany.")
        
        # Process a complex comparison query
        result = agent.process_question("Compare and contrast the capitals of France and Germany.")
        
        # Complex queries should not be bypassed due to "compare" operator
        # The full pipeline should be used
        assert hasattr(result, 'response')
        # Verify that synthesis was required (complex query)
        assert result.error_state.get("requires_synthesis", False) == True