#!/usr/bin/env python3
"""
Quick TopK Test
"""

import numpy as np
import time
from pure_episodic_navigator import PureEpisodicNavigator, create_complex_maze

class QuickTopKNavigator(PureEpisodicNavigator):
    """Quick test of TopK idea"""
    
    def __init__(self, maze: np.ndarray, k: int = 30):
        super().__init__(maze, message_depth=3)
        self.k = k
        self.query_count = 0
        self.total_query_time = 0
    
    def get_n_hop_episodes(self, start_pos, n_hops):
        """Override with TopK approach"""
        if not self.episodes:
            return []
        
        start_time = time.time()
        
        # Simple TopK based on distance
        distances = []
        for i, ep in enumerate(self.episodes):
            dist = abs(ep['pos'][0] - start_pos[0]) + abs(ep['pos'][1] - start_pos[1])
            distances.append((dist, i))
        
        # Get top K
        distances.sort(key=lambda x: x[0])
        topk_indices = [idx for _, idx in distances[:self.k]]
        
        # Always include goal episodes
        for i, ep in enumerate(self.episodes):
            if ep['reached_goal'] and i not in topk_indices:
                topk_indices.append(i)
        
        self.query_count += 1
        self.total_query_time += time.time() - start_time
        
        return topk_indices[:self.k]

def test_quick():
    """Quick test"""
    print("="*60)
    print("QUICK TOPK TEST")
    print("="*60)
    
    # Test on 20x20
    size = 20
    maze = create_complex_maze(size, seed=42)
    
    print(f"\nTesting {size}×{size} maze with TopK approach")
    nav = QuickTopKNavigator(maze, k=30)
    
    start = time.time()
    result = nav.navigate(max_steps=2000)
    elapsed = time.time() - start
    
    if result['success']:
        print(f"\n✓ SUCCESS!")
        print(f"Steps: {result['steps']}")
        print(f"Total time: {elapsed:.2f}s")
        print(f"Queries: {nav.query_count}")
        print(f"Avg query time: {nav.total_query_time/nav.query_count*1000:.2f}ms")
    else:
        print(f"\n✗ Failed after {result['steps']} steps")
        print(f"Coverage: {len(nav.visited)/(size*size)*100:.1f}%")
    
    # Compare with original
    print("\n" + "-"*40)
    print("Testing same maze with original approach")
    nav2 = PureEpisodicNavigator(maze, message_depth=3)
    
    start = time.time()
    result2 = nav2.navigate(max_steps=2000)
    elapsed2 = time.time() - start
    
    if result2['success']:
        print(f"\n✓ Original SUCCESS!")
        print(f"Steps: {result2['steps']}")
        print(f"Total time: {elapsed2:.2f}s")
    else:
        print(f"\n✗ Original failed")
    
    # Compare
    print("\n" + "="*60)
    print("COMPARISON:")
    print(f"TopK time: {elapsed:.2f}s")
    print(f"Original time: {elapsed2:.2f}s")
    if elapsed2 > 0:
        print(f"Speedup: {elapsed2/elapsed:.2f}x")


if __name__ == "__main__":
    test_quick()