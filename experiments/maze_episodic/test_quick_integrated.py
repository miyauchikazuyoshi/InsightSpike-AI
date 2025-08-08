#!/usr/bin/env python3
"""Quick test for integrated index navigator"""

import numpy as np
import sys
import os
import time

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '../maze-optimized-search/src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from integrated_vector_graph_index import IntegratedVectorGraphIndex

def test_basic_functionality():
    """Test basic integrated index functionality"""
    print("Testing Integrated Index...")
    
    # Create index
    index = IntegratedVectorGraphIndex(dimension=6)
    
    # Add some episodes
    for i in range(100):
        vec = np.random.randn(6)
        episode = {
            'vec': vec,
            'text': f'Episode {i}',
            'pos': (i % 10, i // 10),
            'c_value': 0.5
        }
        idx = index.add_episode(episode)
        
    print(f"Added {len(index.metadata)} episodes")
    
    # Test search
    query = np.random.randn(6)
    
    start = time.time()
    indices, scores = index.search(query, k=10)
    search_time = (time.time() - start) * 1000
    
    print(f"Search time: {search_time:.2f}ms")
    print(f"Found {len(indices)} results")
    
    # Compare with original pure episodic
    print("\nComparing with O(n²) approach:")
    print(f"Integrated Index: {search_time:.2f}ms for {len(index.metadata)} episodes")
    print(f"Estimated O(n²): {(100*100)/1000:.2f}ms (rough estimate)")
    

def test_small_maze():
    """Test on a very small maze"""
    from pure_episodic_integrated import PureEpisodicIntegrated
    from insightspike.environments.proper_maze_generator import ProperMazeGenerator
    
    print("\n\nTesting 5x5 maze...")
    
    generator = ProperMazeGenerator()
    maze = generator.generate_dfs_maze(size=(5, 5))
    
    navigator = PureEpisodicIntegrated(maze, message_depth=2)
    
    # Navigate for limited steps
    for step in range(50):
        if navigator.position == navigator.goal:
            print(f"Success! Reached goal in {step} steps")
            break
            
        action = navigator.get_action()
        if action:
            navigator.move(action)
            
        if step % 10 == 0:
            episodes = len(navigator.index.metadata)
            avg_search = np.mean(navigator.search_times) if navigator.search_times else 0
            print(f"Step {step}: pos={navigator.position}, episodes={episodes}, avg_search={avg_search:.2f}ms")


if __name__ == "__main__":
    test_basic_functionality()
    test_small_maze()