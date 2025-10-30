#!/usr/bin/env python3
"""
Test full spike detection with new GED
"""

import numpy as np
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)


def test_spike_detection():
    """Test spike detection with known graph transformation."""
    print("=== Testing Spike Detection ===\n")
    
    from insightspike.implementations.datastore.memory_store import InMemoryDataStore
    from insightspike.implementations.agents.main_agent import MainAgent
    
    config = {
        "processing": {
            "enable_learning": True,
        },
        "l4_config": {
            "provider": "mock",
        },
        "graph": {
            "use_new_ged_implementation": True,
            "spike_ged_threshold": -0.5,  # Expecting negative GED
            "spike_ig_threshold": 0.2
        }
    }
    
    datastore = InMemoryDataStore()
    agent = MainAgent(config=config, datastore=datastore)
    
    # 1. Build initial square-like knowledge
    print("1. Building initial square structure...")
    agent.add_knowledge("Node A connects to Node B")
    agent.add_knowledge("Node B connects to Node C")
    agent.add_knowledge("Node C connects to Node D")
    agent.add_knowledge("Node D connects to Node A")
    
    result1 = agent.process_question("What is the structure?")
    print(f"   Initial result:")
    print(f"   - Spike: {result1.spike_detected}")
    print(f"   - Graph analysis: {result1.graph_analysis.get('metrics', {})}")
    
    # 2. Add hub to trigger spike
    print("\n2. Adding hub node...")
    agent.add_knowledge("Node E is a central hub connecting to all nodes")
    agent.add_knowledge("Node E connects to A, B, C, and D")
    
    result2 = agent.process_question("What is the hub structure?")
    print(f"   Hub result:")
    print(f"   - Spike: {result2.spike_detected}")
    print(f"   - Graph analysis: {result2.graph_analysis.get('metrics', {})}")
    
    # 3. Check metrics directly
    print("\n3. Checking spike detection directly...")
    metrics = result2.graph_analysis.get('metrics', {})
    delta_ged = metrics.get('delta_ged', 0)
    delta_ig = metrics.get('delta_ig', 0)
    
    print(f"   Delta GED: {delta_ged}")
    print(f"   Delta IG: {delta_ig}")
    print(f"   GED < -0.5? {delta_ged < -0.5}")
    print(f"   IG > 0.2? {delta_ig > 0.2}")
    
    # 4. Try with extreme values
    print("\n4. Adding more dramatic change...")
    for i in range(10):
        agent.add_knowledge(f"Node F{i} connects to hub E")
    
    result3 = agent.process_question("What is the star structure?")
    print(f"   Star result:")
    print(f"   - Spike: {result3.spike_detected}")
    print(f"   - Metrics: {result3.graph_analysis.get('metrics', {})}")


if __name__ == "__main__":
    test_spike_detection()