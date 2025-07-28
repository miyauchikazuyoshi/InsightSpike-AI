#\!/usr/bin/env python3
"""
Test GED calculation through full pipeline
"""

import numpy as np
from typing import Dict, Any


def test_ged_calculation_flow():
    """Test if GED is actually calculated."""
    print("=== Testing GED Calculation Flow ===\n")
    
    from insightspike.implementations.datastore.memory_store import InMemoryDataStore
    from insightspike.implementations.agents.main_agent import MainAgent
    
    # Enable debug logging for metrics
    import logging
    logging.getLogger("insightspike.algorithms.metrics_selector").setLevel(logging.DEBUG)
    logging.getLogger("insightspike.implementations.layers.layer3_graph_reasoner").setLevel(logging.DEBUG)
    
    config = {
        "processing": {
            "enable_learning": True,
        },
        "l4_config": {
            "provider": "mock",
        },
        "graph": {
            "use_new_ged_implementation": True,
            "similarity_threshold": 0.7,
            "hop_limit": 2
        }
    }
    
    datastore = InMemoryDataStore()
    agent = MainAgent(config=config, datastore=datastore)
    
    # 1. First add some knowledge
    print("1. Adding initial knowledge...")
    agent.add_knowledge("Graph theory studies networks")
    agent.add_knowledge("Networks have nodes and edges")
    
    # Process first question
    print("\n2. Processing first question...")
    result1 = agent.process_question("What are graphs?")
    
    print(f"   Result 1:")
    print(f"   - Success: {result1.success}")
    print(f"   - Spike detected: {result1.spike_detected}")
    print(f"   - Graph analysis: {result1.graph_analysis}")
    
    # 2. Add more knowledge to create structural change
    print("\n3. Adding hub knowledge...")
    agent.add_knowledge("Hub nodes connect many other nodes")
    agent.add_knowledge("Star graphs have a central hub")
    
    # Process second question
    print("\n4. Processing second question...")
    result2 = agent.process_question("What are hub nodes?")
    
    print(f"   Result 2:")
    print(f"   - Success: {result2.success}")
    print(f"   - Spike detected: {result2.spike_detected}")
    print(f"   - Graph analysis: {result2.graph_analysis}")
    
    # Check if L3 has metrics
    print("\n5. Checking L3 state...")
    agent._ensure_l3_initialized()
    if hasattr(agent, 'l3_graph') and agent.l3_graph:
        # Check last analysis result
        if hasattr(agent.l3_graph, 'previous_graph'):
            print(f"   L3 has previous graph: {agent.l3_graph.previous_graph is not None}")
        
        # Try to get metrics directly
        from insightspike.algorithms.metrics_selector import MetricsSelector
        selector = MetricsSelector(config)
        
        print("\n6. Testing GED calculation directly...")
        # Create simple test graphs
        import networkx as nx
        g1 = nx.Graph([(0, 1), (1, 2)])
        g2 = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 0)])
        
        try:
            ged = selector.delta_ged(g1, g2)
            print(f"   Direct GED calculation: {ged}")
        except Exception as e:
            print(f"   GED calculation error: {e}")


def check_graph_analysis_in_cycle():
    """Check what's in graph_analysis during cycle."""
    print("\n\n=== Checking Graph Analysis in Cycle ===\n")
    
    # Patch process_question to see internal state
    from insightspike.implementations.agents import main_agent
    
    original_cycle = main_agent.MainAgent._reasoning_cycle
    
    def patched_cycle(self, question: str, context: Dict[str, Any] = None):
        result = original_cycle(self, question, context)
        
        print("\n=== Inside _reasoning_cycle ===")
        print(f"Retrieved docs: {len(result.retrieved_documents)}")
        print(f"Graph analysis keys: {list(result.graph_analysis.keys())}")
        print(f"Graph analysis: {result.graph_analysis}")
        
        return result
    
    main_agent.MainAgent._reasoning_cycle = patched_cycle
    
    # Test
    from insightspike.implementations.datastore.memory_store import InMemoryDataStore
    from insightspike.implementations.agents.main_agent import MainAgent
    
    config = {
        "processing": {"enable_learning": True},
        "l4_config": {"provider": "mock"},
        "graph": {"use_new_ged_implementation": True}
    }
    
    datastore = InMemoryDataStore()
    agent = MainAgent(config=config, datastore=datastore)
    
    agent.add_knowledge("Test 1")
    agent.add_knowledge("Test 2")
    
    result = agent.process_question("What is test?")
    
    # Restore
    main_agent.MainAgent._reasoning_cycle = original_cycle


if __name__ == "__main__":
    test_ged_calculation_flow()
    check_graph_analysis_in_cycle()