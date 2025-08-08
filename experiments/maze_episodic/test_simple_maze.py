#!/usr/bin/env python3
"""
Test pure memory navigation on a simple fixed maze
"""

import numpy as np
from pure_episodic_movement_memory import PureEpisodicMovementMemory


def test_simple_maze():
    """Test on a simple 5x5 maze"""
    # Simple maze with clear path
    maze = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ])
    
    print("Testing Pure Memory Navigation")
    print("Maze (0=path, 1=wall):")
    for row in maze:
        print(' '.join(['.' if x == 0 else '#' for x in row]))
    print(f"\nStart: (0,0), Goal: (4,4)")
    print("-" * 30)
    
    # Create navigator
    navigator = PureEpisodicMovementMemory(maze, max_depth=5)
    
    # Navigate
    result = navigator.navigate(max_steps=500)
    
    # Print results
    if result['success']:
        print(f"\n✅ SUCCESS!")
        print(f"  Steps taken: {result['steps']}")
        print(f"  Total episodes: {result['total_episodes']}")
        print(f"  Wall hits: {result['wall_hits']}")
        print(f"  Path length: {len(result['path'])}")
        
        # Show path
        print(f"\nPath taken:")
        for i in range(0, len(result['path']), 10):
            segment = result['path'][i:i+10]
            print(f"  {' -> '.join([f'({x},{y})' for x, y in segment])}")
        
        # Show visit counts
        print(f"\nMost visited positions:")
        sorted_visits = sorted(result['visit_counts'].items(), 
                             key=lambda x: x[1], reverse=True)[:5]
        for pos, count in sorted_visits:
            print(f"  {pos}: {count} visits")
    else:
        print(f"\n❌ Failed after {result['steps']} steps")
        print(f"  Total episodes: {result['total_episodes']}")
        print(f"  Wall hits: {result['wall_hits']}")
    
    # Show episode distribution
    if 'index_stats' in result:
        stats = result['index_stats']
        print(f"\nIndex Statistics:")
        print(f"  Graph nodes: {stats.get('graph_nodes', 0)}")
        print(f"  Graph edges: {stats.get('graph_edges', 0)}")
        if stats.get('graph_density'):
            print(f"  Graph density: {stats['graph_density']:.3f}")
    
    return result


if __name__ == "__main__":
    test_simple_maze()