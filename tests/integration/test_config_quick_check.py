"""
Quick Configuration Check Test
==============================

Simplified test to verify basic configurations work.
"""

import pytest
import yaml
import tempfile
from pathlib import Path

from insightspike.config.loader import load_config
from insightspike.implementations.agents import MainAgent


class TestConfigQuickCheck:
    """Quick check for different configurations."""
    
    @pytest.mark.parametrize("config_name,config_dict", [
        # Minimal
        ("minimal", {
            "llm": {"provider": "mock"},
            "datastore": {"type": "in_memory"}
        }),
        
        # Spectral GED
        ("spectral", {
            "llm": {"provider": "mock"},
            "datastore": {"type": "in_memory"},
            "metrics": {
                "spectral_evaluation": {
                    "enabled": True,
                    "weight": 0.3
                }
            }
        }),
        
        # Multi-hop
        ("multihop", {
            "llm": {"provider": "mock"},
            "datastore": {"type": "in_memory"},
            "metrics": {
                "use_multihop_gedig": True,
                "multihop_config": {
                    "max_hops": 2,
                    "decay_factor": 0.7
                }
            }
        }),
        
        # Layer1 bypass
        ("layer1_bypass", {
            "llm": {"provider": "mock"},
            "datastore": {"type": "in_memory"},
            "processing": {
                "enable_layer1_bypass": True,
                "bypass_uncertainty_threshold": 0.2
            }
        }),
        
        # Graph search
        ("graph_search", {
            "llm": {"provider": "mock"},
            "datastore": {"type": "in_memory"},
            "graph": {
                "enable_graph_search": True,
                "hop_limit": 2
            }
        }),
    ])
    def test_config_works(self, config_name: str, config_dict: dict):
        """Test that a configuration works."""
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_dict, f)
            config_path = f.name
        
        try:
            # Load config
            config = load_config(config_path=config_path)
            
            # Create and initialize agent
            agent = MainAgent(config=config)
            agent.initialize()
            
            # Add knowledge
            agent.add_knowledge("Test knowledge.")
            
            # Process question
            result = agent.process_question("What is test?")
            
            # Basic checks
            assert result is not None
            assert hasattr(result, 'response')
            assert len(result.response) > 0
            
        finally:
            Path(config_path).unlink()
    
    def test_all_features_config(self):
        """Test configuration with all features enabled."""
        config_dict = {
            "llm": {"provider": "mock"},
            "datastore": {"type": "in_memory"},
            "processing": {
                "enable_layer1_bypass": True,
                "enable_insight_registration": True,
                "enable_insight_search": True
            },
            "metrics": {
                "use_normalized_ged": True,
                "spectral_evaluation": {
                    "enabled": True,
                    "weight": 0.3
                },
                "use_multihop_gedig": True,
                "multihop_config": {
                    "max_hops": 2,
                    "decay_factor": 0.5
                }
            },
            "graph": {
                "enable_graph_search": True
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_dict, f)
            config_path = f.name
        
        try:
            config = load_config(config_path=config_path)
            agent = MainAgent(config=config)
            agent.initialize()
            
            # Multiple operations
            agent.add_knowledge("Knowledge A.")
            agent.add_knowledge("Knowledge B.")
            agent.add_knowledge("Knowledge C connects A and B.")
            
            result1 = agent.process_question("What is knowledge A?")
            assert result1 is not None
            
            result2 = agent.process_question("How are A and B connected?")
            assert result2 is not None
            assert hasattr(result2, 'spike_detected')
            
        finally:
            Path(config_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])