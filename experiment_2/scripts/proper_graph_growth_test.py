#!/usr/bin/env python3
"""
Test proper graph growth using official InsightSpike API
"""

import sys
import json
import torch
from pathlib import Path
from datetime import datetime

# Add src directory to path
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from insightspike.core.agents.main_agent import MainAgent


def get_comprehensive_metrics():
    """Get detailed metrics about data state"""
    metrics = {}
    
    # Episodes
    episodes_path = Path("data/episodes.json")
    if episodes_path.exists():
        with open(episodes_path, 'r') as f:
            episodes = json.load(f)
        metrics['episodes'] = {
            'count': len(episodes),
            'size': episodes_path.stat().st_size
        }
    
    # Graph
    graph_path = Path("data/graph_pyg.pt")
    if graph_path.exists():
        try:
            graph = torch.load(graph_path)
            metrics['graph'] = {
                'size': graph_path.stat().st_size,
                'num_nodes': graph.num_nodes if hasattr(graph, 'num_nodes') else 
                            graph.x.shape[0] if hasattr(graph, 'x') else 0,
                'num_edges': graph.edge_index.shape[1] // 2 if hasattr(graph, 'edge_index') else 0
            }
        except:
            metrics['graph'] = {'size': graph_path.stat().st_size, 'error': 'Could not load graph'}
    
    return metrics


def main():
    """Test proper graph growth"""
    print("Testing Proper Graph Growth with Official API")
    print("=" * 50)
    
    # Get initial metrics
    initial_metrics = get_comprehensive_metrics()
    print(f"\\nInitial state:")
    print(f"  Episodes: {initial_metrics.get('episodes', {}).get('count', 0)}")
    print(f"  Graph nodes: {initial_metrics.get('graph', {}).get('num_nodes', 0)}")
    print(f"  Graph edges: {initial_metrics.get('graph', {}).get('num_edges', 0)}")
    
    # Initialize MainAgent properly
    print("\\nInitializing MainAgent...")
    agent = MainAgent()
    agent.initialize()
    
    # Load test data
    test_data_path = Path("experiment_2/dynamic_growth/test_knowledge.json")
    with open(test_data_path, 'r') as f:
        knowledge_items = json.load(f)
    
    print(f"\\nLoaded {len(knowledge_items)} knowledge items")
    
    # Add episodes with graph updates
    print("\\nAdding episodes with graph updates...")
    success_count = 0
    
    for i, item in enumerate(knowledge_items[:10]):  # Test with first 10 items
        try:
            print(f"\\n[{i+1}/10] Processing: {item['text'][:50]}...")
            
            # Use the proper API method
            result = agent.add_episode_with_graph_update(
                text=item['text'],
                c_value=0.5  # Default confidence
            )
            
            if result:
                success_count += 1
                print(f"  ✓ Added successfully")
                
                # Check if graph is growing
                if i % 3 == 2:  # Check every 3 items
                    current_metrics = get_comprehensive_metrics()
                    print(f"  Current graph nodes: {current_metrics.get('graph', {}).get('num_nodes', 0)}")
            else:
                print(f"  ✗ Failed to add")
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    # Save state to persist everything
    print("\\nSaving agent state...")
    try:
        save_result = agent.save_state()
        print(f"Save result: {'✓ Success' if save_result else '✗ Failed'}")
    except Exception as e:
        print(f"Save error: {e}")
    
    # Get final metrics
    final_metrics = get_comprehensive_metrics()
    
    # Analysis
    print("\\n" + "=" * 50)
    print("RESULTS ANALYSIS")
    
    print(f"\\nEpisode growth: {initial_metrics.get('episodes', {}).get('count', 0)} → {final_metrics.get('episodes', {}).get('count', 0)}")
    print(f"Graph node growth: {initial_metrics.get('graph', {}).get('num_nodes', 0)} → {final_metrics.get('graph', {}).get('num_nodes', 0)}")
    print(f"Graph edge growth: {initial_metrics.get('graph', {}).get('num_edges', 0)} → {final_metrics.get('graph', {}).get('num_edges', 0)}")
    
    # Check if graph actually grew
    graph_node_growth = final_metrics.get('graph', {}).get('num_nodes', 0) - initial_metrics.get('graph', {}).get('num_nodes', 0)
    
    if graph_node_growth > 0:
        print(f"\\n✅ SUCCESS: Graph grew by {graph_node_growth} nodes!")
        print(f"Graph file size: {initial_metrics.get('graph', {}).get('size', 0)} → {final_metrics.get('graph', {}).get('size', 0)} bytes")
    else:
        print(f"\\n❌ Graph did not grow")
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'initial_metrics': initial_metrics,
        'final_metrics': final_metrics,
        'items_processed': 10,
        'items_succeeded': success_count,
        'graph_node_growth': graph_node_growth
    }
    
    results_path = Path("experiment_2/results/proper_graph_growth_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()