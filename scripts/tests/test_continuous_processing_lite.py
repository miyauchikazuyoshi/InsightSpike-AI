"""
Lightweight test for continuous processing - 10 data points per config
"""

import time
import psutil
import os
from insightspike.implementations.agents.main_agent import MainAgent

# Simplified configurations
configurations = [
    {
        "name": "Baseline",
        "config": {
            'llm': {'provider': 'mock'},
            'graph': {
                'enable_message_passing': False,
                'enable_graph_search': False,
                'use_gnn': False
            }
        }
    },
    {
        "name": "Message Passing",
        "config": {
            'llm': {'provider': 'mock'},
            'graph': {
                'enable_message_passing': True,
                'message_passing': {'alpha': 0.3, 'iterations': 2},
                'enable_graph_search': False,
                'use_gnn': False
            }
        }
    },
    {
        "name": "All Features",
        "config": {
            'llm': {'provider': 'mock'},
            'graph': {
                'enable_message_passing': True,
                'message_passing': {'alpha': 0.5, 'iterations': 2},
                'enable_graph_search': True,
                'use_gnn': False  # Disable GNN to speed up
            }
        }
    }
]

# Reduced test data - 10 items each
test_knowledge = [
    "The sun is a star.",
    "Water freezes at 0°C.",
    "AI learns from data.",
    "DNA stores information.",
    "Earth orbits the sun.",
    "Neurons send signals.",
    "Plants use photosynthesis.",
    "Gravity attracts objects.",
    "Atoms have electrons.",
    "Evolution changes species."
]

test_questions = [
    "What is the sun?",
    "When does water freeze?", 
    "How does AI work?",
    "What is DNA?",
    "How does Earth move?",
    "What do neurons do?",
    "How do plants grow?",
    "What is gravity?",
    "What are atoms?",
    "What is evolution?"
]

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def test_configuration(config_name, config):
    """Test a single configuration"""
    print(f"\n{'='*50}")
    print(f"Testing: {config_name}")
    print(f"{'='*50}")
    
    # Memory before
    mem_start = get_memory_usage()
    
    # Initialize agent
    agent = MainAgent(config)
    
    # Metrics
    start_time = time.time()
    spike_count = 0
    
    # Add knowledge
    print("Adding knowledge...")
    for i, knowledge in enumerate(test_knowledge):
        agent.add_knowledge(knowledge)
        print(f"  [{i+1}/10] {knowledge}")
    
    # Process questions
    print("\nProcessing questions...")
    for i, question in enumerate(test_questions):
        q_start = time.time()
        result = agent.process_question(question)
        q_time = time.time() - q_start
        
        # Check spike
        has_spike = getattr(result, 'has_spike', False)
        if has_spike:
            spike_count += 1
            
        print(f"  [{i+1}/10] {question} | Spike: {has_spike} | {q_time:.3f}s")
    
    # Summary
    total_time = time.time() - start_time
    mem_end = get_memory_usage()
    mem_used = mem_end - mem_start
    
    print(f"\nSummary:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Spikes: {spike_count}/10")
    print(f"  Memory used: {mem_used:.1f} MB")
    
    return {
        'name': config_name,
        'time': total_time,
        'spikes': spike_count,
        'memory': mem_used
    }

def main():
    """Run all tests"""
    print("Continuous Processing Test (Lite Version)")
    print("10 knowledge items + 10 questions per configuration")
    
    results = []
    
    for config_info in configurations:
        result = test_configuration(config_info['name'], config_info['config'])
        results.append(result)
    
    # Summary
    print(f"\n{'='*50}")
    print("FINAL RESULTS")
    print(f"{'='*50}")
    print(f"{'Config':<20} {'Time(s)':<10} {'Spikes':<10} {'Memory(MB)':<10}")
    print("-" * 50)
    
    for r in results:
        print(f"{r['name']:<20} {r['time']:<10.2f} {r['spikes']:<10} {r['memory']:<10.1f}")

if __name__ == "__main__":
    main()