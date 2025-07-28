"""
Test continuous processing with 20 data points for each configuration
"""

import time
import numpy as np
from insightspike.implementations.agents.main_agent import MainAgent

# Test configurations
configurations = [
    {
        "name": "Baseline (all disabled)",
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
        "name": "Message Passing only",
        "config": {
            'llm': {'provider': 'mock'},
            'graph': {
                'enable_message_passing': True,
                'message_passing': {
                    'alpha': 0.3,
                    'iterations': 3
                },
                'enable_graph_search': False,
                'use_gnn': False
            }
        }
    },
    {
        "name": "Graph Search only",
        "config": {
            'llm': {'provider': 'mock'},
            'graph': {
                'enable_message_passing': False,
                'enable_graph_search': True,
                'use_gnn': False
            }
        }
    },
    {
        "name": "All features enabled",
        "config": {
            'llm': {'provider': 'mock'},
            'graph': {
                'enable_message_passing': True,
                'message_passing': {
                    'alpha': 0.5,
                    'iterations': 3
                },
                'enable_graph_search': True,
                'use_gnn': True
            }
        }
    },
    {
        "name": "High sensitivity spike detection",
        "config": {
            'llm': {'provider': 'mock'},
            'graph': {
                'enable_message_passing': True,
                'message_passing': {
                    'alpha': 0.4,
                    'iterations': 2
                },
                'spike_ged_threshold': 0.1,  # Very sensitive
                'spike_ig_threshold': 0.9,   # Very sensitive
                'conflict_threshold': 0.3
            }
        }
    }
]

# Test data - 20 diverse knowledge items
test_knowledge = [
    "The sun is a star at the center of our solar system.",
    "Water freezes at 0 degrees Celsius.",
    "Machine learning uses algorithms to learn from data.",
    "DNA contains genetic information.",
    "The Earth orbits around the sun.",
    "Neurons transmit electrical signals in the brain.",
    "Photosynthesis converts light energy to chemical energy.",
    "Gravity pulls objects toward each other.",
    "Electrons orbit around the nucleus of atoms.",
    "Evolution occurs through natural selection.",
    "Quantum mechanics describes behavior at atomic scales.",
    "The speed of light is approximately 300,000 km/s.",
    "Proteins are made of amino acids.",
    "Climate change is caused by greenhouse gases.",
    "Black holes have extremely strong gravitational fields.",
    "Artificial intelligence can recognize patterns.",
    "The human brain has billions of neurons.",
    "Energy cannot be created or destroyed.",
    "Chemical reactions involve electron transfer.",
    "The universe is expanding."
]

# Test questions - 20 diverse questions
test_questions = [
    "What is the sun?",
    "How does water change with temperature?",
    "What is machine learning?",
    "What stores genetic information?",
    "How does Earth move?",
    "How do neurons work?",
    "What is photosynthesis?",
    "What is gravity?",
    "What are atoms made of?",
    "How does evolution work?",
    "What is quantum mechanics?",
    "How fast is light?",
    "What are proteins?",
    "What causes climate change?",
    "What are black holes?",
    "What can AI do?",
    "How complex is the brain?",
    "Can energy be destroyed?",
    "What happens in chemical reactions?",
    "Is the universe changing?"
]

def test_configuration(config_name, config):
    """Test a single configuration with 20 data points"""
    print(f"\n{'='*60}")
    print(f"Testing: {config_name}")
    print(f"{'='*60}")
    
    # Initialize agent
    agent = MainAgent(config)
    
    # Metrics
    start_time = time.time()
    spike_count = 0
    errors = []
    processing_times = []
    
    # Add all knowledge
    print("\nAdding 20 knowledge items...")
    for i, knowledge in enumerate(test_knowledge):
        try:
            knowledge_start = time.time()
            agent.add_knowledge(knowledge)
            knowledge_time = time.time() - knowledge_start
            print(f"  [{i+1}/20] Added: {knowledge[:50]}... ({knowledge_time:.3f}s)")
        except Exception as e:
            errors.append(f"Knowledge {i+1}: {str(e)}")
            print(f"  [{i+1}/20] ERROR: {str(e)}")
    
    # Process all questions
    print("\nProcessing 20 questions...")
    for i, question in enumerate(test_questions):
        try:
            question_start = time.time()
            result = agent.process_question(question)
            question_time = time.time() - question_start
            processing_times.append(question_time)
            
            # Check for spike
            has_spike = False
            if hasattr(result, 'has_spike'):
                has_spike = result.has_spike
            elif isinstance(result, dict) and 'has_spike' in result:
                has_spike = result['has_spike']
            
            if has_spike:
                spike_count += 1
                
            print(f"  [{i+1}/20] Q: {question[:40]}... | Spike: {has_spike} | Time: {question_time:.3f}s")
            
        except Exception as e:
            errors.append(f"Question {i+1}: {str(e)}")
            print(f"  [{i+1}/20] ERROR: {str(e)}")
            processing_times.append(0)
    
    # Summary
    total_time = time.time() - start_time
    avg_processing_time = np.mean(processing_times) if processing_times else 0
    
    print(f"\n{'-'*60}")
    print(f"Summary for {config_name}:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average question processing: {avg_processing_time:.3f}s")
    print(f"  Spikes detected: {spike_count}/20")
    print(f"  Errors: {len(errors)}")
    if errors:
        print(f"  Error details:")
        for error in errors[:5]:  # Show first 5 errors
            print(f"    - {error}")
    
    return {
        'config_name': config_name,
        'total_time': total_time,
        'avg_processing_time': avg_processing_time,
        'spike_count': spike_count,
        'error_count': len(errors),
        'errors': errors
    }

def main():
    """Run all configuration tests"""
    print("Starting continuous processing test with 20 data points per configuration")
    print(f"Total configurations to test: {len(configurations)}")
    print(f"Total operations: {len(configurations)} x 40 = {len(configurations) * 40}")
    
    results = []
    
    for config_info in configurations:
        result = test_configuration(config_info['name'], config_info['config'])
        results.append(result)
    
    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    
    print("\nPerformance comparison:")
    print(f"{'Configuration':<35} {'Total(s)':<10} {'Avg(s)':<10} {'Spikes':<10} {'Errors':<10}")
    print("-" * 75)
    
    for result in results:
        print(f"{result['config_name']:<35} "
              f"{result['total_time']:<10.2f} "
              f"{result['avg_processing_time']:<10.3f} "
              f"{result['spike_count']:<10} "
              f"{result['error_count']:<10}")
    
    # Check for memory leaks or performance degradation
    print("\nStability check:")
    baseline_time = results[0]['avg_processing_time']
    for result in results:
        degradation = ((result['avg_processing_time'] - baseline_time) / baseline_time) * 100
        print(f"  {result['config_name']}: {degradation:+.1f}% vs baseline")

if __name__ == "__main__":
    main()