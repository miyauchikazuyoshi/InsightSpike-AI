"""
Test all configurations with 20 data points each
"""

import time
import json
from datetime import datetime
from insightspike.implementations.agents.main_agent import MainAgent

# Test data - 20 items
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

configurations = [
    {
        "name": "1_baseline",
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
        "name": "2_message_passing",
        "config": {
            'llm': {'provider': 'mock'},
            'graph': {
                'enable_message_passing': True,
                'message_passing': {
                    'alpha': 0.3,
                    'iterations': 2  # Reduced from 3
                },
                'enable_graph_search': False,
                'use_gnn': False
            }
        }
    },
    {
        "name": "3_graph_search",
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
        "name": "4_all_features",
        "config": {
            'llm': {'provider': 'mock'},
            'graph': {
                'enable_message_passing': True,
                'message_passing': {
                    'alpha': 0.5,
                    'iterations': 2  # Reduced
                },
                'enable_graph_search': True,
                'use_gnn': False  # Disabled to avoid slowdown
            }
        }
    }
]

def test_configuration(config_name, config):
    """Test a single configuration with 20 data points"""
    print(f"\n{'='*70}")
    print(f"Testing: {config_name}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    
    result = {
        "config": config_name,
        "start_time": datetime.now().isoformat(),
        "errors": [],
        "timings": {
            "knowledge_addition": [],
            "question_processing": []
        },
        "spikes": []
    }
    
    try:
        # Initialize
        init_start = time.time()
        agent = MainAgent(config)
        result["init_time"] = time.time() - init_start
        print(f"✓ Agent initialized in {result['init_time']:.3f}s")
        
        # Add knowledge
        print(f"\nAdding {len(test_knowledge)} knowledge items...")
        total_add_start = time.time()
        
        for i, knowledge in enumerate(test_knowledge):
            k_start = time.time()
            try:
                agent.add_knowledge(knowledge)
                k_time = time.time() - k_start
                result["timings"]["knowledge_addition"].append(k_time)
                print(f"  [{i+1:2d}/20] {k_time:.3f}s - {knowledge[:40]}...")
            except Exception as e:
                result["errors"].append(f"Knowledge {i+1}: {str(e)}")
                print(f"  [{i+1:2d}/20] ERROR - {str(e)}")
        
        result["total_add_time"] = time.time() - total_add_start
        
        # Process questions
        print(f"\nProcessing {len(test_questions)} questions...")
        total_q_start = time.time()
        
        for i, question in enumerate(test_questions):
            q_start = time.time()
            try:
                response = agent.process_question(question)
                q_time = time.time() - q_start
                result["timings"]["question_processing"].append(q_time)
                
                has_spike = getattr(response, 'has_spike', False)
                if has_spike:
                    result["spikes"].append(i)
                
                print(f"  [{i+1:2d}/20] {q_time:.3f}s | Spike: {has_spike:5} | {question[:30]}...")
                
            except Exception as e:
                result["errors"].append(f"Question {i+1}: {str(e)}")
                print(f"  [{i+1:2d}/20] ERROR - {str(e)}")
        
        result["total_q_time"] = time.time() - total_q_start
        result["total_time"] = time.time() - init_start
        
        # Summary
        print(f"\n{'-'*70}")
        print(f"Summary for {config_name}:")
        print(f"  Total time: {result['total_time']:.2f}s")
        print(f"  Knowledge addition: {result['total_add_time']:.2f}s")
        print(f"  Question processing: {result['total_q_time']:.2f}s")
        print(f"  Avg per question: {result['total_q_time']/20:.3f}s")
        print(f"  Spikes detected: {len(result['spikes'])}/20")
        print(f"  Errors: {len(result['errors'])}")
        
        result["success"] = True
        
    except Exception as e:
        result["success"] = False
        result["fatal_error"] = str(e)
        print(f"\n❌ FATAL ERROR: {str(e)}")
    
    result["end_time"] = datetime.now().isoformat()
    return result

def main():
    """Run all configuration tests"""
    print("CONTINUOUS PROCESSING TEST - 20 DATA POINTS PER CONFIGURATION")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_results = []
    
    for config_info in configurations:
        result = test_configuration(config_info["name"], config_info["config"])
        all_results.append(result)
    
    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"{'Configuration':<25} {'Total(s)':<10} {'Avg Q(s)':<10} {'Spikes':<10} {'Status':<10}")
    print("-" * 70)
    
    for result in all_results:
        if result["success"]:
            avg_q = result["total_q_time"] / 20
            status = "✓ OK"
        else:
            avg_q = 0
            status = "✗ FAILED"
            
        print(f"{result['config']:<25} "
              f"{result.get('total_time', 0):<10.2f} "
              f"{avg_q:<10.3f} "
              f"{len(result.get('spikes', [])):<10} "
              f"{status:<10}")
    
    # Save results
    with open("continuous_test_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to continuous_test_results.json")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()