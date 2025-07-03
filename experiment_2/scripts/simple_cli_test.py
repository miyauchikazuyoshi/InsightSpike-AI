#!/usr/bin/env python3
"""
Simple test to add data using InsightSpike CLI
"""

import subprocess
import json
import time
from pathlib import Path
from datetime import datetime


def get_data_stats():
    """Get current data statistics"""
    stats = {}
    
    # Check episodes.json
    episodes_path = Path("data/episodes.json")
    if episodes_path.exists():
        with open(episodes_path, 'r') as f:
            episodes = json.load(f)
        stats['episodes_count'] = len(episodes)
        stats['episodes_size'] = episodes_path.stat().st_size
    
    # Check graph_pyg.pt
    graph_path = Path("data/graph_pyg.pt")
    if graph_path.exists():
        stats['graph_size'] = graph_path.stat().st_size
    
    return stats


def add_knowledge_via_cli(text: str):
    """Add knowledge using the CLI"""
    try:
        # Use poetry run insightspike ask command
        cmd = ['poetry', 'run', 'insightspike', 'ask', text]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print(f"✓ Added: {text[:60]}...")
            return True
        else:
            print(f"✗ Failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("✗ Command timed out")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    """Run simple CLI test"""
    print("Testing InsightSpike CLI data addition")
    print("=" * 50)
    
    # Get initial stats
    initial_stats = get_data_stats()
    print(f"Initial state: {initial_stats}")
    
    # Load test data
    test_data_path = Path("experiment_2/dynamic_growth/test_knowledge.json")
    if not test_data_path.exists():
        print("Test data not found. Creating...")
        subprocess.run(['python', 'experiment_2/scripts/create_test_data.py'])
    
    with open(test_data_path, 'r') as f:
        knowledge_items = json.load(f)
    
    print(f"\\nLoaded {len(knowledge_items)} test items")
    
    # Test with first 5 items
    test_items = knowledge_items[:5]
    success_count = 0
    
    print("\\nAdding knowledge items...")
    for i, item in enumerate(test_items):
        print(f"\\n[{i+1}/{len(test_items)}]")
        if add_knowledge_via_cli(item['text']):
            success_count += 1
        time.sleep(1)  # Small delay between additions
    
    # Get final stats
    final_stats = get_data_stats()
    print(f"\\nFinal state: {final_stats}")
    
    # Calculate growth
    if 'episodes_count' in initial_stats and 'episodes_count' in final_stats:
        growth = final_stats['episodes_count'] - initial_stats['episodes_count']
        print(f"\\nEpisode growth: {growth}")
        print(f"Success rate: {success_count}/{len(test_items)}")
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'initial_stats': initial_stats,
        'final_stats': final_stats,
        'items_attempted': len(test_items),
        'items_succeeded': success_count,
        'growth': final_stats.get('episodes_count', 0) - initial_stats.get('episodes_count', 0)
    }
    
    results_path = Path("experiment_2/results/cli_test_results.json")
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()