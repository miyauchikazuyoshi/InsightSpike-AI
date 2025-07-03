#!/usr/bin/env python3
"""
Load data into InsightSpike with explicit persistence
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Import using relative path like CLI does
from insightspike.core.agents.main_agent import MainAgent


def get_data_metrics():
    """Get comprehensive data metrics"""
    metrics = {}
    
    # Episodes
    episodes_path = Path("data/episodes.json")
    if episodes_path.exists():
        with open(episodes_path, 'r') as f:
            episodes = json.load(f)
        metrics['episodes'] = {
            'count': len(episodes),
            'size': episodes_path.stat().st_size,
            'size_per_item': episodes_path.stat().st_size / len(episodes) if episodes else 0
        }
    
    # Graph
    graph_path = Path("data/graph_pyg.pt")
    if graph_path.exists():
        metrics['graph'] = {
            'size': graph_path.stat().st_size
        }
    
    # Index
    index_path = Path("data/index.faiss")
    if index_path.exists():
        metrics['index'] = {
            'size': index_path.stat().st_size
        }
    
    # Total size
    total_size = 0
    for key in ['episodes', 'graph', 'index']:
        if key in metrics and 'size' in metrics[key]:
            total_size += metrics[key]['size']
    metrics['total_size'] = total_size
    
    return metrics


def load_and_persist_data():
    """Load data and explicitly save it"""
    print("Loading data with explicit persistence")
    print("=" * 50)
    
    # Get initial metrics
    initial_metrics = get_data_metrics()
    print(f"\\nInitial metrics:")
    print(f"  Episodes: {initial_metrics.get('episodes', {}).get('count', 0)}")
    print(f"  Total size: {initial_metrics['total_size']:,} bytes")
    
    # Build agent
    print("\\nInitializing agent...")
    agent = MainAgent()
    
    # Load test knowledge
    test_data_path = Path("experiment_2/dynamic_growth/test_knowledge.json")
    with open(test_data_path, 'r') as f:
        knowledge_items = json.load(f)
    
    print(f"\\nLoaded {len(knowledge_items)} knowledge items")
    
    # Track growth
    growth_data = [{
        'step': 0,
        'added': 0,
        'metrics': initial_metrics
    }]
    
    # Add data in batches
    batch_size = 5
    total_added = 0
    
    for i in range(0, min(len(knowledge_items), 30), batch_size):  # Limit to 30 items
        batch = knowledge_items[i:i+batch_size]
        print(f"\\n[Batch {i//batch_size + 1}] Processing {len(batch)} items...")
        
        batch_added = 0
        for item in batch:
            try:
                # Add document
                agent.add_document(item['text'])
                batch_added += 1
                total_added += 1
                print(f"  ✓ Added: {item['text'][:50]}...")
            except Exception as e:
                print(f"  ✗ Error: {e}")
        
        # Save after each batch
        print(f"  Saving batch {i//batch_size + 1}...")
        try:
            save_success = agent.l2_memory.save()
            if save_success:
                print(f"  ✓ Batch saved successfully")
            else:
                print(f"  ✗ Save failed")
        except Exception as e:
            print(f"  ✗ Save error: {e}")
        
        # Get updated metrics
        current_metrics = get_data_metrics()
        growth_data.append({
            'step': i//batch_size + 1,
            'added': total_added,
            'metrics': current_metrics
        })
        
        # Show progress
        episode_growth = current_metrics.get('episodes', {}).get('count', 0) - initial_metrics.get('episodes', {}).get('count', 0)
        size_growth = current_metrics['total_size'] - initial_metrics['total_size']
        
        print(f"  Current state: {current_metrics.get('episodes', {}).get('count', 0)} episodes (+{episode_growth}), "
              f"{current_metrics['total_size']:,} bytes (+{size_growth:,})")
    
    # Final save
    print("\\nPerforming final save...")
    final_save = agent.l2_memory.save()
    print(f"Final save: {'✓ Success' if final_save else '✗ Failed'}")
    
    # Get final metrics
    final_metrics = get_data_metrics()
    
    # Analysis
    print("\\n" + "=" * 50)
    print("RESULTS ANALYSIS")
    
    total_episode_growth = final_metrics.get('episodes', {}).get('count', 0) - initial_metrics.get('episodes', {}).get('count', 0)
    total_size_growth = final_metrics['total_size'] - initial_metrics['total_size']
    
    print(f"\\nTotal growth:")
    print(f"  Episodes: +{total_episode_growth}")
    print(f"  Size: +{total_size_growth:,} bytes")
    
    if total_episode_growth > 0:
        bytes_per_episode = total_size_growth / total_episode_growth
        print(f"  Bytes per episode: {bytes_per_episode:.1f}")
        
        # Estimate compression ratio
        avg_input_length = sum(len(item['text']) for item in knowledge_items[:total_added]) / total_added
        estimated_raw_size = total_episode_growth * avg_input_length
        compression_ratio = estimated_raw_size / total_size_growth if total_size_growth > 0 else 0
        print(f"  Average input length: {avg_input_length:.0f} characters")
        print(f"  Estimated compression ratio: {compression_ratio:.1f}x")
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'initial_metrics': initial_metrics,
        'final_metrics': final_metrics,
        'growth_data': growth_data,
        'summary': {
            'total_added': total_added,
            'episode_growth': total_episode_growth,
            'size_growth': total_size_growth,
            'compression_ratio': compression_ratio if 'compression_ratio' in locals() else 0
        }
    }
    
    results_path = Path("experiment_2/results/persistent_loader_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\\nResults saved to: {results_path}")
    
    return results


def main():
    try:
        results = load_and_persist_data()
        
        if results['summary']['episode_growth'] > 0:
            print("\\n✅ SUCCESS: Data was loaded and persisted!")
            print(f"Final compression ratio: {results['summary']['compression_ratio']:.1f}x")
        else:
            print("\\n❌ No data growth detected")
            
    except Exception as e:
        print(f"\\nError in main: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()