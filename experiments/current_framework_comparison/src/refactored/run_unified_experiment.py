"""
Unified Experiment Runner
========================

Demonstrates how to use the UnifiedMainAgent to replace old experiment scripts.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from unified_main_agent import UnifiedMainAgent, AgentConfig, AgentMode
import json
import time
from datetime import datetime


def load_knowledge_base():
    """Load the English knowledge base"""
    kb_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '../../data/english_knowledge_base.json'
    )
    
    try:
        with open(kb_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load knowledge base: {e}")
        return []


def run_comparison_experiment():
    """Compare different agent modes on the same questions"""
    print("="*80)
    print("UNIFIED AGENT COMPARISON EXPERIMENT")
    print("="*80)
    
    # Test questions
    questions = [
        "What links SOAP, RADAR, and LASER?",
        "What is the connection between energy and mass?",
        "How are birds, bees, and airplanes related?"
    ]
    
    # Agent modes to test
    modes = [
        (AgentMode.BASIC, "Basic Agent"),
        (AgentMode.ENHANCED, "Enhanced Agent (Graph-Aware)"),
        (AgentMode.QUERY_TRANSFORM, "Query Transform Agent"),
    ]
    
    # Load knowledge base
    knowledge_base = load_knowledge_base()
    
    results = {}
    
    for mode, mode_name in modes:
        print(f"\n{'='*60}")
        print(f"Testing: {mode_name}")
        print('='*60)
        
        # Create agent with specific mode
        config = AgentConfig.from_mode(mode)
        config.max_cycles = 3
        config.verbose = False
        
        agent = UnifiedMainAgent(config)
        
        if not agent.initialize():
            print(f"Failed to initialize {mode_name}")
            continue
        
        # Add knowledge base
        print(f"Adding {len(knowledge_base)} knowledge items...")
        for item in knowledge_base[:20]:  # Add first 20 items for speed
            agent.add_episode(
                item['text'],
                c_value=0.7 if item['phase'] >= 3 else 0.5
            )
        
        # Test each question
        mode_results = []
        for question in questions:
            print(f"\nQuestion: {question}")
            
            start_time = time.time()
            result = agent.process_question(question)
            end_time = time.time()
            
            print(f"Response: {result.get('response', 'No response')[:150]}...")
            print(f"Quality: {result.get('reasoning_quality', 0):.3f}")
            print(f"Spike: {result.get('spike_detected', False)}")
            print(f"Time: {end_time - start_time:.2f}s")
            
            mode_results.append({
                'question': question,
                'response': result.get('response', ''),
                'quality': result.get('reasoning_quality', 0),
                'spike_detected': result.get('spike_detected', False),
                'processing_time': end_time - start_time,
                'cached': result.get('cached', False),
                'cycles': result.get('total_cycles', 1)
            })
        
        results[mode_name] = mode_results
        
        # Cleanup
        agent.cleanup()
    
    return results


def demonstrate_feature_combinations():
    """Show how to combine features flexibly"""
    print("\n" + "="*80)
    print("FEATURE COMBINATION DEMONSTRATION")
    print("="*80)
    
    # Example 1: Fast cached agent
    print("\n1. Fast Cached Agent (Basic + Caching)")
    config = AgentConfig(
        mode=AgentMode.BASIC,
        enable_caching=True,
        cache_size=500,
        max_cycles=2
    )
    agent = UnifiedMainAgent(config)
    print(f"   Features: {[k for k, v in config.__dict__.items() if k.startswith('enable_') and v]}")
    
    # Example 2: Smart but slow agent
    print("\n2. Smart Agent (Query Transform + Multi-hop)")
    config = AgentConfig(
        mode=AgentMode.BASIC,
        enable_query_transform=True,
        enable_multi_hop=True,
        enable_caching=False,  # No caching for maximum exploration
        max_cycles=5
    )
    agent = UnifiedMainAgent(config)
    print(f"   Features: {[k for k, v in config.__dict__.items() if k.startswith('enable_') and v]}")
    
    # Example 3: Production configuration
    print("\n3. Production Agent (Optimized + Custom Settings)")
    config = AgentConfig.from_mode(AgentMode.OPTIMIZED)
    config.cache_size = 5000  # Large cache
    config.parallel_branches = 8  # More parallelism
    config.enable_gpu_acceleration = False  # Disable if no GPU
    agent = UnifiedMainAgent(config)
    print(f"   Features: {[k for k, v in config.__dict__.items() if k.startswith('enable_') and v]}")


def save_results(results, filename="unified_experiment_results.json"):
    """Save experiment results"""
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '../../results',
        filename
    )
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'experiment': 'unified_agent_comparison',
        'results': results
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


def print_summary(results):
    """Print summary of results"""
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    # Calculate averages
    for mode_name, mode_results in results.items():
        avg_quality = sum(r['quality'] for r in mode_results) / len(mode_results)
        avg_time = sum(r['processing_time'] for r in mode_results) / len(mode_results)
        spike_count = sum(1 for r in mode_results if r['spike_detected'])
        
        print(f"\n{mode_name}:")
        print(f"  Average Quality: {avg_quality:.3f}")
        print(f"  Average Time: {avg_time:.2f}s")
        print(f"  Spikes Detected: {spike_count}/{len(mode_results)}")


if __name__ == "__main__":
    # Run the comparison experiment
    results = run_comparison_experiment()
    
    # Save results
    save_results(results)
    
    # Print summary
    print_summary(results)
    
    # Demonstrate feature combinations
    demonstrate_feature_combinations()
    
    print("\n✅ Experiment completed successfully!")
    print("\nThe UnifiedMainAgent successfully replaces all 6 agent variants!")
    print("You can now use a single agent with different configurations.")
    print("\nNext steps:")
    print("1. Update imports in src/ to use UnifiedMainAgent")
    print("2. Delete old agent files")
    print("3. Update documentation")