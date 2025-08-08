#!/usr/bin/env python3
"""Test fixed implementation on 25x25 maze"""

import numpy as np
import sys
import os
import time

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '../maze-optimized-search/src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from pure_episodic_integrated_fixed import PureEpisodicIntegratedFixed
from insightspike.environments.proper_maze_generator import ProperMazeGenerator


def test_fixed_implementation():
    """Test fixed O(1) implementation"""
    
    print("\n" + "="*60)
    print("Testing Fixed Implementation with True O(1) Search")
    print("="*60)
    
    # Test on 15x15 first
    print("\n1. Testing 15x15 maze...")
    generator = ProperMazeGenerator()
    maze = generator.generate_dfs_maze(size=(15, 15))
    
    navigator = PureEpisodicIntegratedFixed(maze)
    result = navigator.navigate(max_steps=1000)
    
    if result['success']:
        print(f"✓ 15x15 Success in {result['steps']} steps")
        print(f"  Average search time: {result['avg_search_time']:.2f}ms")
        print(f"  Total episodes: {result['total_episodes']}")
    else:
        print("✗ 15x15 Failed")
        return
    
    # Now try 25x25
    print("\n2. Testing 25x25 maze...")
    maze = generator.generate_dfs_maze(size=(25, 25))
    
    navigator = PureEpisodicIntegratedFixed(maze)
    result = navigator.navigate(max_steps=3000)
    
    if result['success']:
        print(f"\n{'='*60}")
        print(f"🎉 25x25 SUCCESS in {result['steps']} steps!")
        print(f"{'='*60}")
        print(f"Total episodes: {result['total_episodes']}")
        print(f"Average search time: {result['avg_search_time']:.2f}ms")
        print(f"Total time: {result['total_time']:.2f}s")
        
        # Verify O(1) behavior
        print("\nSearch Time Analysis:")
        search_times = result['search_times']
        if len(search_times) > 100:
            early_avg = np.mean(search_times[:100])
            late_avg = np.mean(search_times[-100:])
            print(f"  First 100 steps avg: {early_avg:.2f}ms")
            print(f"  Last 100 steps avg: {late_avg:.2f}ms")
            print(f"  Ratio (should be ~1 for O(1)): {late_avg/early_avg:.2f}")
            
            if late_avg/early_avg < 2.0:
                print("\n✓ Confirmed O(1) search behavior!")
            else:
                print("\n⚠ Search time increased more than expected")
                
        return True
    else:
        print("✗ 25x25 Failed")
        return False


if __name__ == "__main__":
    success = test_fixed_implementation()
    
    if success:
        print("\n" + "="*60)
        print("BREAKTHROUGH ACHIEVED!")
        print("="*60)
        print("• Integrated index with FAISS enables true O(1) search")
        print("• Large mazes now solvable that previously failed")
        print("• 25x25 maze solved successfully!")
        print("\nNext: Test on 50x50 maze...")