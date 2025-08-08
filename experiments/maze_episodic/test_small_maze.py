#!/usr/bin/env python3
"""
Small Maze Test for Pure Episodic Navigator
"""

import numpy as np
import time
from pure_episodic_navigator import PureEpisodicNavigator, create_complex_maze, visualize_maze_with_path

def test_small_maze():
    """Test on small maze first"""
    print("="*60)
    print("PURE EPISODIC NAVIGATION - SMALL MAZE TEST")
    print("Testing without visit counts")
    print("="*60)
    
    # Test on 15x15 maze
    size = 15
    maze = create_complex_maze(size, seed=42)
    
    # Count walkable cells
    walkable = np.sum(maze == 0)
    print(f"\nMaze statistics:")
    print(f"  Size: {size}×{size}")
    print(f"  Walkable cells: {walkable}")
    print(f"  Density: {walkable/(size*size)*100:.1f}%")
    
    # Create navigator
    nav = PureEpisodicNavigator(maze, message_depth=3)
    
    # Navigate
    result = nav.navigate(max_steps=1500)
    
    # Analysis
    if result['success']:
        print("\n✓ SUCCESS!")
        print(f"  Optimal estimate: ~{2*(size-2)} steps")
        print(f"  Actual: {result['steps']} steps")
        print(f"  Efficiency: {result['steps']/(2*(size-2)):.2f}x optimal")
        
        # Check goal episodes
        goal_episodes = sum(1 for ep in nav.episodes if ep['reached_goal'])
        print(f"\nGoal episodes: {goal_episodes}")
        
        # Visualize
        visualize_maze_with_path(maze, nav.path, 'test_small_maze.png')
    else:
        print("\n✗ Failed")
        print(f"  Explored {len(nav.visited)} cells")
        print(f"  Coverage: {len(nav.visited)/walkable*100:.1f}%")
    
    return result


def analyze_episodes(navigator):
    """Analyze episode distribution"""
    print("\nEpisode Analysis:")
    
    # Result distribution
    results = {'success': 0, 'wall': 0, 'visited': 0}
    for ep in navigator.episodes:
        results[ep['result']] += 1
    
    print("Result distribution:")
    for result, count in results.items():
        print(f"  {result}: {count}")
    
    # Goal signal propagation
    goal_signals = []
    for ep in navigator.episodes[-100:]:  # Last 100 episodes
        goal_signal = ep['embedding'][5]
        if goal_signal > 0:
            goal_signals.append((ep['id'], ep['pos'], goal_signal))
    
    if goal_signals:
        print(f"\nEpisodes with goal signal: {len(goal_signals)}")
        for ep_id, pos, signal in goal_signals[:5]:
            print(f"  Episode {ep_id} at {pos}: signal={signal:.1f}")


if __name__ == "__main__":
    result = test_small_maze()
    
    # If successful, try larger
    if result['success']:
        print("\n" + "="*60)
        print("Small maze successful! Try running the full test.")
    else:
        print("\nNeed to debug why small maze failed.")