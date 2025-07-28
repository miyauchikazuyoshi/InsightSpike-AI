"""
Comprehensive Configuration Variations Test
==========================================

Tests all possible configuration combinations that don't require external resources.
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import pytest
import yaml

from insightspike.config.loader import load_config
from insightspike.implementations.agents import MainAgent


class TestAllConfigVariations:
    """Test pipeline with all configuration variations."""
    
    @staticmethod
    def get_all_config_variations() -> List[Tuple[str, Dict]]:
        """Get comprehensive list of configuration variations."""
        variations = []
        
        # 1. Basic configurations
        variations.extend([
            ("minimal", {
                "llm": {"provider": "mock"},
                "datastore": {"type": "in_memory"}
            }),
            
            ("with_filesystem", {
                "llm": {"provider": "mock"},
                "datastore": {
                    "type": "filesystem",
                    "root_path": "./test_data/insight_store"
                }
            }),
        ])
        
        # 2. Metrics configurations
        variations.extend([
            ("normalized_ged", {
                "llm": {"provider": "mock"},
                "datastore": {"type": "in_memory"},
                "metrics": {
                    "use_normalized_ged": True,
                    "use_entropy_variance_ig": False,
                    "use_multihop_gedig": False
                }
            }),
            
            ("entropy_variance_ig", {
                "llm": {"provider": "mock"},
                "datastore": {"type": "in_memory"},
                "metrics": {
                    "use_normalized_ged": False,
                    "use_entropy_variance_ig": True,
                    "use_multihop_gedig": False
                }
            }),
            
            ("multihop_gedig", {
                "llm": {"provider": "mock"},
                "datastore": {"type": "in_memory"},
                "metrics": {
                    "use_normalized_ged": False,
                    "use_entropy_variance_ig": False,
                    "use_multihop_gedig": True,
                    "multihop_config": {
                        "max_hops": 2,
                        "decay_factor": 0.5
                    }
                }
            }),
            
            ("spectral_ged", {
                "llm": {"provider": "mock"},
                "datastore": {"type": "in_memory"},
                "metrics": {
                    "spectral_evaluation": {
                        "enabled": True,
                        "weight": 0.4
                    }
                }
            }),
            
            ("all_metrics", {
                "llm": {"provider": "mock"},
                "datastore": {"type": "in_memory"},
                "metrics": {
                    "use_normalized_ged": True,
                    "use_entropy_variance_ig": True,
                    "use_multihop_gedig": True,
                    "spectral_evaluation": {
                        "enabled": True,
                        "weight": 0.3
                    },
                    "multihop_config": {
                        "max_hops": 3,
                        "decay_factor": 0.6
                    }
                }
            }),
        ])
        
        # 3. Processing configurations
        variations.extend([
            ("layer1_bypass", {
                "llm": {"provider": "mock"},
                "datastore": {"type": "in_memory"},
                "processing": {
                    "enable_layer1_bypass": True,
                    "bypass_uncertainty_threshold": 0.3,
                    "bypass_known_ratio_threshold": 0.8
                }
            }),
            
            ("insight_features", {
                "llm": {"provider": "mock"},
                "datastore": {"type": "in_memory"},
                "processing": {
                    "enable_insight_registration": True,
                    "enable_insight_search": True,
                    "max_insights_per_query": 3,
                    "insight_relevance_boost": 0.3
                }
            }),
            
            ("dynamic_doc_adjustment", {
                "llm": {"provider": "mock"},
                "datastore": {"type": "in_memory"},
                "processing": {
                    "dynamic_doc_adjustment": True,
                    "max_docs_with_insights": 3
                }
            }),
        ])
        
        # 4. Graph configurations
        variations.extend([
            ("graph_search", {
                "llm": {"provider": "mock"},
                "datastore": {"type": "in_memory"},
                "graph": {
                    "enable_graph_search": True,
                    "hop_limit": 2,
                    "neighbor_threshold": 0.5,
                    "path_decay": 0.8
                }
            }),
            
            ("ged_simple", {
                "llm": {"provider": "mock"},
                "datastore": {"type": "in_memory"},
                "graph": {
                    "ged_algorithm": "simple"
                }
            }),
            
            ("ged_hybrid", {
                "llm": {"provider": "mock"},
                "datastore": {"type": "in_memory"},
                "graph": {
                    "ged_algorithm": "hybrid",
                    "hybrid_weights": {
                        "structure": 0.33,
                        "semantic": 0.33,
                        "quality": 0.34
                    }
                }
            }),
        ])
        
        # 5. Output configurations
        variations.extend([
            ("output_concise", {
                "llm": {"provider": "mock"},
                "datastore": {"type": "in_memory"},
                "output": {
                    "response_style": "concise",
                    "show_reasoning": False,
                    "show_metadata": False
                }
            }),
            
            ("output_detailed", {
                "llm": {"provider": "mock"},
                "datastore": {"type": "in_memory"},
                "output": {
                    "response_style": "detailed",
                    "show_reasoning": True,
                    "show_metadata": True
                }
            }),
        ])
        
        # 6. Memory configurations
        variations.extend([
            ("small_memory", {
                "llm": {"provider": "mock"},
                "datastore": {"type": "in_memory"},
                "memory": {
                    "max_retrieved_docs": 5,
                    "short_term_capacity": 5,
                    "working_memory_capacity": 10,
                    "episodic_memory_capacity": 20
                }
            }),
            
            ("large_memory", {
                "llm": {"provider": "mock"},
                "datastore": {"type": "in_memory"},
                "memory": {
                    "max_retrieved_docs": 20,
                    "short_term_capacity": 20,
                    "working_memory_capacity": 50,
                    "episodic_memory_capacity": 200
                }
            }),
        ])
        
        # 7. Reasoning configurations
        variations.extend([
            ("quick_reasoning", {
                "llm": {"provider": "mock"},
                "datastore": {"type": "in_memory"},
                "reasoning": {
                    "max_cycles": 3,
                    "convergence_threshold": 0.7,
                    "spike_threshold": 0.6
                }
            }),
            
            ("deep_reasoning", {
                "llm": {"provider": "mock"},
                "datastore": {"type": "in_memory"},
                "reasoning": {
                    "max_cycles": 20,
                    "convergence_threshold": 0.9,
                    "spike_threshold": 0.8
                }
            }),
        ])
        
        # 8. Combined configurations
        variations.extend([
            ("performance_optimized", {
                "llm": {"provider": "mock"},
                "datastore": {"type": "in_memory"},
                "processing": {
                    "enable_layer1_bypass": True,
                    "dynamic_doc_adjustment": True
                },
                "performance": {
                    "enable_cache": True,
                    "parallel_workers": 8
                },
                "memory": {
                    "max_retrieved_docs": 5
                }
            }),
            
            ("quality_optimized", {
                "llm": {"provider": "mock"},
                "datastore": {"type": "in_memory"},
                "processing": {
                    "enable_insight_registration": True,
                    "enable_insight_search": True
                },
                "metrics": {
                    "use_normalized_ged": True,
                    "use_entropy_variance_ig": True,
                    "spectral_evaluation": {
                        "enabled": True,
                        "weight": 0.4
                    }
                },
                "reasoning": {
                    "max_cycles": 15,
                    "convergence_threshold": 0.85
                }
            }),
        ])
        
        return variations
    
    def test_single_config(self, name: str, config: Dict) -> Dict:
        """Test a single configuration."""
        results = {
            "name": name,
            "config": config,
            "status": "pending",
            "errors": [],
            "operations": {}
        }
        
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name
        
        try:
            # Load and initialize
            loaded_config = load_config(config_path=config_path)
            agent = MainAgent(config=loaded_config)
            agent.initialize()
            results["status"] = "initialized"
            
            # Test 1: Add knowledge
            try:
                agent.add_knowledge("Paris is the capital of France.")
                agent.add_knowledge("The Eiffel Tower is in Paris.")
                results["operations"]["add_knowledge"] = "success"
            except Exception as e:
                results["operations"]["add_knowledge"] = "failed"
                results["errors"].append(f"add_knowledge: {str(e)}")
            
            # Test 2: Basic question
            try:
                result = agent.process_question("What is the capital of France?")
                assert result is not None
                assert hasattr(result, 'response')
                results["operations"]["basic_question"] = "success"
            except Exception as e:
                results["operations"]["basic_question"] = "failed"
                results["errors"].append(f"basic_question: {str(e)}")
            
            # Test 3: Complex question (potential spike)
            try:
                result = agent.process_question("How are Paris and the Eiffel Tower related?")
                assert hasattr(result, 'spike_detected')
                results["operations"]["complex_question"] = "success"
                results["spike_detected"] = result.spike_detected
            except Exception as e:
                results["operations"]["complex_question"] = "failed"
                results["errors"].append(f"complex_question: {str(e)}")
            
            # Test 4: Follow-up question
            try:
                result = agent.process_question("Tell me more about Paris.")
                results["operations"]["follow_up"] = "success"
            except Exception as e:
                results["operations"]["follow_up"] = "failed"
                results["errors"].append(f"follow_up: {str(e)}")
            
            results["status"] = "completed"
            
        except Exception as e:
            results["status"] = "failed"
            results["errors"].append(f"initialization: {str(e)}")
            
        finally:
            # Cleanup
            Path(config_path).unlink()
            if config.get("datastore", {}).get("type") == "filesystem":
                import shutil
                store_path = Path(config["datastore"]["root_path"])
                if store_path.exists():
                    shutil.rmtree(store_path)
        
        return results
    
    def test_all_configurations(self):
        """Test all configuration variations systematically."""
        variations = self.get_all_config_variations()
        all_results = []
        
        print(f"\nTesting {len(variations)} configuration variations...")
        print("=" * 80)
        
        for i, (name, config) in enumerate(variations, 1):
            print(f"\n[{i}/{len(variations)}] Testing: {name}")
            
            result = self.test_single_config(name, config)
            all_results.append(result)
            
            # Print immediate result
            if result["status"] == "completed":
                successful_ops = sum(1 for v in result["operations"].values() if v == "success")
                total_ops = len(result["operations"])
                print(f"  ✓ Status: {result['status']} ({successful_ops}/{total_ops} operations)")
                if result.get("spike_detected"):
                    print(f"  ✓ Spike detected: {result['spike_detected']}")
            else:
                print(f"  ✗ Status: {result['status']}")
                if result["errors"]:
                    print(f"  Errors: {result['errors'][0]}")
        
        # Generate summary report
        print("\n" + "=" * 80)
        print("CONFIGURATION TEST SUMMARY")
        print("=" * 80)
        
        total = len(all_results)
        completed = sum(1 for r in all_results if r["status"] == "completed")
        initialized = sum(1 for r in all_results if r["status"] in ["initialized", "completed"])
        
        print(f"\nOverall Statistics:")
        print(f"  Total configurations: {total}")
        print(f"  Successfully completed: {completed} ({completed/total*100:.1f}%)")
        print(f"  Successfully initialized: {initialized} ({initialized/total*100:.1f}%)")
        print(f"  Failed: {total - initialized} ({(total-initialized)/total*100:.1f}%)")
        
        # Operation statistics
        operation_stats = {}
        for result in all_results:
            if result["status"] == "completed":
                for op, status in result["operations"].items():
                    if op not in operation_stats:
                        operation_stats[op] = {"success": 0, "failed": 0}
                    operation_stats[op][status] += 1
        
        print(f"\nOperation Success Rates:")
        for op, stats in operation_stats.items():
            total_op = stats["success"] + stats["failed"]
            success_rate = stats["success"] / total_op * 100 if total_op > 0 else 0
            print(f"  {op}: {stats['success']}/{total_op} ({success_rate:.1f}%)")
        
        # Failed configurations
        failed_configs = [r for r in all_results if r["status"] == "failed"]
        if failed_configs:
            print(f"\nFailed Configurations:")
            for fc in failed_configs:
                print(f"  - {fc['name']}: {fc['errors'][0] if fc['errors'] else 'Unknown error'}")
        
        # Save detailed results
        output_dir = Path("test_results")
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / "all_config_variations_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\nDetailed results saved to: {output_dir / 'all_config_variations_results.json'}")
        
        # Assert all completed successfully
        assert completed == total, f"{total - completed} configurations failed to complete"
        
        return all_results


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    test = TestAllConfigVariations()
    try:
        test.test_all_configurations()
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()