#!/usr/bin/env python3
"""
Test the legacy embed command to see if it persists data
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
    
    # Check index.faiss
    index_path = Path("data/index.faiss")
    if index_path.exists():
        stats['index_size'] = index_path.stat().st_size
    
    return stats


def test_embed_command():
    """Test the embed command with a test file"""
    print("Testing InsightSpike embed command")
    print("=" * 50)
    
    # Get initial stats
    initial_stats = get_data_stats()
    print(f"\\nInitial state: {initial_stats}")
    
    # Create a simple test file
    test_file = Path("experiment_2/dynamic_growth/embed_test.txt")
    test_content = """
The embed command test begins here.

Artificial Intelligence is transforming how we interact with technology.
Machine learning algorithms can identify patterns in vast amounts of data.
Deep learning has enabled breakthroughs in computer vision and natural language processing.

Python is the most popular language for data science and machine learning.
TensorFlow and PyTorch are leading frameworks for deep learning research.
Transformer models have revolutionized natural language understanding.

This is test content to verify if the embed command persists data.
We expect to see changes in episodes.json and related files after embedding.
"""
    
    with open(test_file, 'w') as f:
        f.write(test_content)
    
    print(f"\\nCreated test file: {test_file}")
    print(f"Content length: {len(test_content)} characters")
    
    # Run embed command
    print("\\nRunning embed command...")
    try:
        cmd = ['poetry', 'run', 'insightspike', 'embed', '--path', str(test_file)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        print(f"Return code: {result.returncode}")
        if result.stdout:
            print(f"Output: {result.stdout[:200]}...")
        if result.stderr:
            print(f"Error: {result.stderr[:200]}...")
            
    except Exception as e:
        print(f"Error running embed: {e}")
    
    # Wait a bit for processing
    print("\\nWaiting for processing...")
    time.sleep(3)
    
    # Check final stats
    final_stats = get_data_stats()
    print(f"\\nFinal state: {final_stats}")
    
    # Calculate changes
    print("\\nChanges:")
    if 'episodes_count' in initial_stats and 'episodes_count' in final_stats:
        episode_change = final_stats['episodes_count'] - initial_stats['episodes_count']
        print(f"  Episodes: {initial_stats['episodes_count']} → {final_stats['episodes_count']} ({'+' if episode_change >= 0 else ''}{episode_change})")
    
    if 'episodes_size' in initial_stats and 'episodes_size' in final_stats:
        size_change = final_stats['episodes_size'] - initial_stats['episodes_size']
        print(f"  Episodes size: {initial_stats['episodes_size']:,} → {final_stats['episodes_size']:,} ({'+' if size_change >= 0 else ''}{size_change:,} bytes)")
    
    if 'graph_size' in initial_stats and 'graph_size' in final_stats:
        graph_change = final_stats['graph_size'] - initial_stats['graph_size']
        print(f"  Graph size: {initial_stats['graph_size']:,} → {final_stats['graph_size']:,} ({'+' if graph_change >= 0 else ''}{graph_change:,} bytes)")
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'initial_stats': initial_stats,
        'final_stats': final_stats,
        'test_file': str(test_file),
        'content_length': len(test_content),
        'changes': {
            'episodes': final_stats.get('episodes_count', 0) - initial_stats.get('episodes_count', 0),
            'size': final_stats.get('episodes_size', 0) - initial_stats.get('episodes_size', 0)
        }
    }
    
    results_path = Path("experiment_2/results/embed_test_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\\nResults saved to: {results_path}")
    
    # If data grew, show more details
    if results['changes']['episodes'] > 0:
        print("\\n✅ SUCCESS: Data was embedded and persisted!")
        print(f"Added {results['changes']['episodes']} new episodes")
        print(f"Storage increased by {results['changes']['size']:,} bytes")
        
        if results['changes']['size'] > 0:
            compression = len(test_content) / results['changes']['size']
            print(f"Compression ratio: {compression:.2f}x")
    else:
        print("\\n❌ No data growth detected")


def main():
    test_embed_command()


if __name__ == "__main__":
    main()