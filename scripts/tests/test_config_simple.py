#!/usr/bin/env python3
"""
Simple configuration test
"""

import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from insightspike.implementations.agents.main_agent import MainAgent

# Suppress most logs
logging.basicConfig(level=logging.ERROR)

os.environ["ANTHROPIC_API_KEY"] = "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"

def test_configuration(name, graph_config):
    """Test a single configuration"""
    print(f"\n--- {name} ---")
    
    config = {
        "llm": {
            "provider": "anthropic",
            "model": "claude-3-sonnet-20240229",
            "api_key": os.environ.get("ANTHROPIC_API_KEY"),
            "temperature": 0.7,
            "max_tokens": 300
        },
        "graph": graph_config
    }
    
    try:
        # Create agent
        agent = MainAgent(config)
        
        # Add knowledge
        agent.add_knowledge("The sun provides energy through radiation.")
        agent.add_knowledge("Solar panels convert sunlight into electricity.")
        
        # Ask question
        result = agent.process_question("How does the sun create electricity?")
        
        # Get results
        response = getattr(result, 'response', 'No response')
        spike = getattr(result, 'spike_detected', False)
        
        # Check graph stats
        graph_stats = {}
        graph_analysis = getattr(result, 'graph_analysis', {})
        if graph_analysis:
            graph = graph_analysis.get('graph')
            if graph:
                graph_stats['nodes'] = getattr(graph, 'num_nodes', 0)
                if hasattr(graph, 'edge_info') and graph.edge_info:
                    new_edges = sum(1 for e in graph.edge_info if e.get('type') == 'new')
                    graph_stats['new_edges'] = new_edges
        
        print(f"Response length: {len(response)}")
        print(f"Spike detected: {spike}")
        print(f"Graph stats: {graph_stats}")
        print("✓ Success")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    print("=== CONFIGURATION TEST ===")
    
    configs = [
        ("Baseline", {
            "enable_message_passing": False,
            "enable_graph_search": False
        }),
        
        ("Message Passing (Low)", {
            "enable_message_passing": True,
            "message_passing": {
                "alpha": 0.2,
                "iterations": 2
            }
        }),
        
        ("Message Passing (High)", {
            "enable_message_passing": True,
            "message_passing": {
                "alpha": 0.7,
                "iterations": 5
            },
            "edge_reevaluation": {
                "new_edge_threshold": 0.6
            }
        }),
        
        ("Graph Search Enabled", {
            "enable_message_passing": True,
            "enable_graph_search": True,
            "message_passing": {
                "alpha": 0.4,
                "iterations": 3
            }
        })
    ]
    
    results = []
    for name, config in configs:
        success = test_configuration(name, config)
        results.append((name, success))
    
    # Summary
    print("\n=== SUMMARY ===")
    successful = sum(1 for _, success in results if success)
    print(f"Total: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(results) - successful}")

if __name__ == "__main__":
    main()