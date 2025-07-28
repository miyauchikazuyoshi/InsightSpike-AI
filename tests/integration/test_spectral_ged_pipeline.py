"""
Test Spectral GED in Full Pipeline
=================================

Integration test for spectral GED feature in the complete InsightSpike pipeline.
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from insightspike.config.loader import load_config
from insightspike.implementations.agents import MainAgent


class TestSpectralGEDPipeline:
    """Test spectral GED feature in full pipeline."""
    
    def test_spectral_ged_with_mainagent(self):
        """Test that MainAgent works with spectral GED enabled."""
        # Create temporary config with spectral enabled
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_data = {
                'llm': {'provider': 'mock'},
                'datastore': {'type': 'in_memory'},
                'metrics': {
                    'spectral_evaluation': {
                        'enabled': True,
                        'weight': 0.3
                    }
                }
            }
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            # Load config and create agent
            config = load_config(config_path=config_path)
            agent = MainAgent(config=config)
            agent.initialize()
            
            # Add some knowledge
            agent.add_knowledge("The Earth revolves around the Sun.")
            agent.add_knowledge("The Moon revolves around the Earth.")
            
            # Process a question that should create connections
            result = agent.process_question("What is the relationship between Earth, Sun, and Moon?")
            
            # Basic assertions
            assert result is not None
            assert hasattr(result, 'response')
            assert len(result.response) > 0
            
            # The mock LLM won't produce real spikes, but the pipeline should work
            # CycleResult has spike_detected, not has_spike
            assert hasattr(result, 'spike_detected')
            
        finally:
            # Clean up
            Path(config_path).unlink()
    
    def test_spectral_ged_backward_compatibility(self):
        """Test that disabling spectral GED maintains backward compatibility."""
        # Test with spectral disabled (default)
        config_off = {
            'llm': {'provider': 'mock'},
            'datastore': {'type': 'in_memory'}
        }
        
        agent_off = MainAgent(config=config_off)
        agent_off.initialize()
        
        # Test with spectral explicitly disabled
        config_explicit = {
            'llm': {'provider': 'mock'},
            'datastore': {'type': 'in_memory'},
            'metrics': {
                'spectral_evaluation': {
                    'enabled': False
                }
            }
        }
        
        agent_explicit = MainAgent(config=config_explicit)
        agent_explicit.initialize()
        
        # Both should work identically
        knowledge = "Test knowledge for comparison."
        agent_off.add_knowledge(knowledge)
        agent_explicit.add_knowledge(knowledge)
        
        question = "What is the test knowledge?"
        result_off = agent_off.process_question(question)
        result_explicit = agent_explicit.process_question(question)
        
        # Both should produce results
        assert result_off is not None
        assert result_explicit is not None
        
        # Mock LLM produces consistent results, so responses should be similar
        assert result_off.response == result_explicit.response


if __name__ == "__main__":
    pytest.main([__file__, "-v"])