#!/usr/bin/env python3
"""Test different threshold values for geDIG wiring."""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from navigation.maze_navigator import MazeNavigator


def test_thresholds():
    """Test different geDIG thresholds."""
    
    # Create a simple 5x5 maze
    maze = np.ones((5, 5), dtype=int)
    for i in range(1, 4):
        maze[2, i] = 0
        maze[i, 2] = 0
    
    start = (1, 2)
    goal = (3, 2)
    maze[start[0], start[1]] = 0
    maze[goal[0], goal[1]] = 0
    
    print("Testing different geDIG thresholds:")
    print("=" * 50)
    
    # Test different thresholds
    thresholds = [0.0, -0.1, -0.2, -0.3, -0.4, -0.5, -1.0]
    
    for thresh in thresholds:
        try:
            nav = MazeNavigator(
                maze=maze,
                start_pos=start,
                goal_pos=goal,
                wiring_strategy='gedig',
                gedig_threshold=thresh,
                simple_mode=True
            )
            
            steps = 0
            max_steps = 20
            
            while steps < max_steps:
                nav.step()
                steps += 1
                if nav.current_pos == goal:
                    break
            
            # Get graph info
            graph = nav.graph_manager.graph
            success = nav.current_pos == goal
            
            print(f"Threshold {thresh:5.1f}: Steps={steps:2d}, Success={success}, "
                  f"Nodes={graph.number_of_nodes():2d}, Edges={graph.number_of_edges():2d}")
            
            # Check actual geDIG values
            if hasattr(nav, 'gedig_history') and nav.gedig_history:
                values = nav.gedig_history
                print(f"  geDIG range: [{min(values):.3f}, {max(values):.3f}], mean={np.mean(values):.3f}")
            
        except Exception as e:
            print(f"Threshold {thresh:5.1f}: ERROR - {e}")
    
    print("\nConclusion:")
    print("The geDIG values are in range [-0.5, 0.0] approximately.")
    print("We need a threshold around -0.1 to -0.2 to actually create edges.")


if __name__ == '__main__':
    test_thresholds()