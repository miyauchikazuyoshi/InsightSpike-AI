#!/usr/bin/env python3
"""
Layer1 Bypass Demo
==================

Demonstrates the Layer1 bypass mechanism for known concepts with low uncertainty.
"""

import time
from insightspike.config import load_config


def test_bypass_performance():
    """Test performance improvement with Layer1 bypass"""
    
    print("=== Layer1 Bypass Performance Demo ===\n")
    
    # Test with bypass disabled
    print("1. Testing WITHOUT bypass...")
    config_no_bypass = load_config(preset="experiment")
    config_no_bypass.processing.enable_layer1_bypass = False
    
    from insightspike.implementations.agents.main_agent import MainAgent  # internal import inside function
    agent_no_bypass = MainAgent(config_no_bypass)
    
    # Add known knowledge
    knowledge_base = [
        "The capital of France is Paris.",
        "Paris is a city in France.",
        "The Eiffel Tower is in Paris.",
        "France is a country in Europe.",
        "The population of Paris is about 2.2 million.",
    ]
    
    for knowledge in knowledge_base:
        agent_no_bypass.add_knowledge(knowledge)
    
    # Test queries
    test_queries = [
        "What is the capital of France?",
        "Where is Paris located?",
        "What landmark is in Paris?",
        "What continent is France in?",
        "What is the population of Paris?",
    ]
    
    # Time without bypass
    start_time = time.time()
    for query in test_queries:
        result = agent_no_bypass.process_question(query, verbose=False)
        print(f"  Q: {query}")
        print(f"  A: {result.response[:100]}...")
        print(f"  Uncertainty: {result.error_state.get('uncertainty', 'N/A')}")
        print()
    
    time_no_bypass = time.time() - start_time
    print(f"Total time WITHOUT bypass: {time_no_bypass:.3f} seconds\n")
    
    # Test with bypass enabled
    print("2. Testing WITH bypass...")
    config_with_bypass = load_config(preset="experiment")
    config_with_bypass.processing.enable_layer1_bypass = True
    config_with_bypass.processing.bypass_uncertainty_threshold = 0.3
    config_with_bypass.processing.bypass_known_ratio_threshold = 0.8
    
    from insightspike.implementations.agents.main_agent import MainAgent  # internal import inside function
    agent_with_bypass = MainAgent(config_with_bypass)
    
    # Add same knowledge
    for knowledge in knowledge_base:
        agent_with_bypass.add_knowledge(knowledge)
    
    # Time with bypass
    start_time = time.time()
    bypass_count = 0
    for query in test_queries:
        result = agent_with_bypass.process_question(query, verbose=True)
        print(f"  Q: {query}")
        print(f"  A: {result.response[:100]}...")
        print(f"  Uncertainty: {result.error_state.get('uncertainty', 'N/A')}")
        if result.error_state.get('suggested_path') == 'bypass':
            print(f"  ✓ BYPASSED - Low uncertainty!")
            bypass_count += 1
        print()
    
    time_with_bypass = time.time() - start_time
    print(f"Total time WITH bypass: {time_with_bypass:.3f} seconds")
    print(f"Queries bypassed: {bypass_count}/{len(test_queries)}")
    
    # Calculate speedup
    if time_with_bypass > 0:
        speedup = time_no_bypass / time_with_bypass
        print(f"\n🚀 Speedup: {speedup:.2f}x faster with bypass enabled!")
    
    # Test complex query that shouldn't be bypassed
    print("\n3. Testing complex query (should NOT bypass)...")
    complex_query = "Compare and contrast the capital cities of France and Germany."
    
    result = agent_with_bypass.process_question(complex_query, verbose=True)
    print(f"  Q: {complex_query}")
    print(f"  A: {result.response[:100]}...")
    print(f"  Uncertainty: {result.error_state.get('uncertainty', 'N/A')}")
    if result.error_state.get('suggested_path') != 'bypass':
        print(f"  ✓ NOT BYPASSED - Complex query requiring full processing")
    else:
        print(f"  ⚠️  Unexpectedly bypassed!")


if __name__ == "__main__":
    test_bypass_performance()
