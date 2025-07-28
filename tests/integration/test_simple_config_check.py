"""
Simple Configuration Check
=========================

Basic test to verify different configurations work.
"""

import yaml
import tempfile
from pathlib import Path

from insightspike.config.loader import load_config
from insightspike.implementations.agents import MainAgent


def test_configuration(name: str, config_dict: dict):
    """Test a single configuration."""
    print(f"\n{'='*50}")
    print(f"Testing: {name}")
    print(f"{'='*50}")
    
    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_dict, f)
        config_path = f.name
    
    try:
        # Load config
        config = load_config(config_path=config_path)
        print("✓ Config loaded successfully")
        
        # Create agent
        agent = MainAgent(config=config)
        print("✓ Agent created successfully")
        
        # Initialize
        agent.initialize()
        print("✓ Agent initialized successfully")
        
        # Add knowledge
        agent.add_knowledge("Test knowledge for verification.")
        print("✓ Knowledge added successfully")
        
        # Process question
        result = agent.process_question("What is test knowledge?")
        print("✓ Question processed successfully")
        
        # Check result
        if hasattr(result, 'response'):
            print(f"✓ Response received: {len(result.response)} chars")
        if hasattr(result, 'spike_detected'):
            print(f"✓ Spike detection: {result.spike_detected}")
            
        print(f"\n✅ {name} configuration: PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ {name} configuration: FAILED")
        print(f"   Error: {str(e)}")
        return False
    finally:
        Path(config_path).unlink()


def main():
    """Test key configurations."""
    
    configs = [
        # 1. Minimal
        ("Minimal", {
            "llm": {"provider": "mock"},
            "datastore": {"type": "in_memory"}
        }),
        
        # 2. With spectral GED
        ("Spectral GED", {
            "llm": {"provider": "mock"},
            "datastore": {"type": "in_memory"},
            "metrics": {
                "spectral_evaluation": {
                    "enabled": True,
                    "weight": 0.3
                }
            }
        }),
        
        # 3. Multi-hop
        ("Multi-hop", {
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
        
        # 4. Layer1 bypass
        ("Layer1 Bypass", {
            "llm": {"provider": "mock"},
            "datastore": {"type": "in_memory"},
            "processing": {
                "enable_layer1_bypass": True,
                "bypass_uncertainty_threshold": 0.2
            }
        }),
        
        # 5. Graph search
        ("Graph Search", {
            "llm": {"provider": "mock"},
            "datastore": {"type": "in_memory"},
            "graph": {
                "enable_graph_search": True,
                "hop_limit": 2
            }
        }),
        
        # 6. All features
        ("All Features", {
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
                }
            },
            "graph": {
                "enable_graph_search": True
            }
        })
    ]
    
    print("Configuration Validation Test")
    print("="*50)
    
    results = []
    for name, config in configs:
        success = test_configuration(name, config)
        results.append((name, success))
    
    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"Total configurations tested: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    
    print("\nDetailed Results:")
    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {name}: {status}")
    
    if passed == total:
        print(f"\n🎉 All configurations passed!")
    else:
        print(f"\n⚠️  Some configurations failed.")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    main()