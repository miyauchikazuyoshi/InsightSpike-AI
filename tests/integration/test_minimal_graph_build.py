#!/usr/bin/env python3
"""
Minimal test to find the actual graph building error
"""

import numpy as np
import tempfile
import traceback

def test_minimal_graph_build():
    """Minimal test to isolate graph building error."""
    print("=== Minimal Graph Build Test ===\n")
    
    # Apply patches
    from insightspike.patches import apply_all_fixes
    apply_all_fixes()
    
    try:
        # Setup
        from insightspike.implementations.agents.main_agent import MainAgent
        from insightspike.implementations.datastore.memory_store import InMemoryDataStore
        
        config_dict = {
            "processing": {
                "enable_learning": True,
                "enable_insight_registration": True,
                "enable_insight_search": True,
                "max_insights_per_query": 10
            },
            "memory": {
                "max_retrieved_docs": 5
            },
            "llm": {
                "provider": "mock",
                "model": "mock-model"
            },
            "graph": {
                "similarity_threshold": 0.7,
                "hop_limit": 2,
                "use_new_ged_implementation": True
            },
            "embedder": {
                "model_name": "mock",
                "vector_dim": 384
            }
        }
        
        datastore = InMemoryDataStore()
        
        # Convert dict config to object with attributes
        from types import SimpleNamespace
        
        def dict_to_namespace(d):
            """Convert nested dict to nested SimpleNamespace."""
            if isinstance(d, dict):
                return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
            return d
        
        config_obj = dict_to_namespace(config_dict)
        
        print("Creating MainAgent...")
        agent = MainAgent(config=config_obj, datastore=datastore)
        print("MainAgent created successfully")
        
        # Initialize agent
        print("Initializing agent...")
        init_result = agent.initialize()
        print(f"Agent initialized: {init_result}")
        
        # Add knowledge
        print("\nAdding knowledge...")
        knowledge = [
            "Graph theory is fundamental to network analysis",
            "Hub nodes have high degree centrality"
        ]
        
        for text in knowledge:
            try:
                result = agent.add_knowledge(text)
                print(f"Added: '{text[:30]}...' - Result: {result}")
            except Exception as e:
                print(f"Failed to add: '{text[:30]}...' - Error: {e}")
                traceback.print_exc()
        
        # Process question to trigger graph building
        print("\nProcessing question to trigger graph building...")
        try:
            result = agent.process_question("What is graph theory?")
            print(f"Processing result: {type(result)}")
            print(f"Response: {getattr(result, 'response', str(result))[:100]}")
            
            # Check if graph was built
            if hasattr(agent, 'l3_graph') and agent.l3_graph:
                print("\nChecking graph state...")
                if hasattr(agent.l3_graph, 'graph_builder'):
                    gb = agent.l3_graph.graph_builder
                    print(f"Graph builder documents: {len(gb.documents) if gb.documents else 0}")
                    print(f"Graph builder index: {gb.index is not None}")
                
                if hasattr(agent.l3_graph, 'previous_graph'):
                    pg = agent.l3_graph.previous_graph
                    print(f"Previous graph: {pg}")
                    if pg:
                        print(f"  Nodes: {getattr(pg, 'num_nodes', 'N/A')}")
                        print(f"  Has edge_index: {hasattr(pg, 'edge_index')}")
                
        except Exception as e:
            print(f"Processing failed: {e}")
            traceback.print_exc()
            
    except Exception as e:
        print(f"Test failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_minimal_graph_build()