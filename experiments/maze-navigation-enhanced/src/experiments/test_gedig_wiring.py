#!/usr/bin/env python3
"""Test geDIG wiring strategy specifically to find the bug."""

import os
import sys
import numpy as np
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from navigation.maze_navigator import MazeNavigator
from core.graph_manager import GraphManager
from core.episode_manager import Episode


def test_wiring_strategies():
    """Test different wiring strategies to identify the issue."""
    
    print("=" * 60)
    print("Testing Wiring Strategies")
    print("=" * 60)
    
    # Create a simple 5x5 maze
    maze = np.ones((5, 5), dtype=int)
    # Carve a simple path
    for i in range(1, 4):
        maze[2, i] = 0  # Horizontal path
        maze[i, 2] = 0  # Vertical path
    
    start = (1, 2)
    goal = (3, 2)
    maze[start[0], start[1]] = 0
    maze[goal[0], goal[1]] = 0
    
    print("Maze created (5x5)")
    print("Start:", start)
    print("Goal:", goal)
    print()
    
    # Test each strategy
    strategies = ['simple', 'gedig']
    
    for strategy in strategies:
        print(f"\nTesting '{strategy}' strategy...")
        print("-" * 40)
        
        try:
            nav = MazeNavigator(
                maze=maze,
                start_pos=start,
                goal_pos=goal,
                wiring_strategy=strategy,
                gedig_threshold=-0.5,
                backtrack_threshold=-0.3,
                simple_mode=True,
                backtrack_debounce=True
            )
            
            steps = 0
            max_steps = 20
            path = []
            start_time = time.time()
            
            while steps < max_steps:
                print(f"  Step {steps}: pos={nav.current_pos}", end='')
                
                # Add timeout protection
                if time.time() - start_time > 5:
                    print(" [TIMEOUT]")
                    break
                
                try:
                    action = nav.step()
                    print(f" -> action={action}")
                except Exception as e:
                    print(f" [ERROR: {e}]")
                    import traceback
                    traceback.print_exc()
                    break
                
                path.append(nav.current_pos)
                steps += 1
                
                if nav.current_pos == goal:
                    print(f"  ✓ Goal reached in {steps} steps!")
                    break
            
            if nav.current_pos != goal and steps >= max_steps:
                print(f"  ✗ Max steps reached without finding goal")
            
            # Analyze graph structure
            if hasattr(nav, 'graph_manager'):
                graph = nav.graph_manager.graph
                print(f"  Graph nodes: {graph.number_of_nodes()}")
                print(f"  Graph edges: {graph.number_of_edges()}")
            
            # Check geDIG history
            if hasattr(nav, 'gedig_history'):
                history = nav.gedig_history
                if history:
                    print(f"  geDIG values: min={min(history):.3f}, max={max(history):.3f}, mean={np.mean(history):.3f}")
                else:
                    print("  No geDIG values recorded")
            
        except Exception as e:
            print(f"Failed to create navigator: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)


def test_graph_manager_directly():
    """Test GraphManager's gedig wiring directly."""
    
    print("\n" + "=" * 60)
    print("Testing GraphManager Directly")
    print("=" * 60)
    
    from core.gedig_evaluator import GeDIGEvaluator
    
    # Create graph manager
    gedig_eval = GeDIGEvaluator()
    graph_mgr = GraphManager(gedig_eval)
    
    # Create some dummy episodes
    episodes = []
    for i in range(5):
        ep = Episode(
            episode_id=i,
            position=(i, i),
            observation=np.zeros((3, 3)),
            timestamp=float(i)
        )
        graph_mgr.add_episode(ep)
        episodes.append(ep)
    
    print(f"Created {len(episodes)} episodes")
    print(f"Initial graph: {graph_mgr.graph.number_of_nodes()} nodes, {graph_mgr.graph.number_of_edges()} edges")
    
    # Test simple wiring
    print("\nTesting simple wiring...")
    graph_mgr._wire_simple(episodes)
    print(f"After simple: {graph_mgr.graph.number_of_nodes()} nodes, {graph_mgr.graph.number_of_edges()} edges")
    
    # Reset graph
    graph_mgr.graph.clear()
    for ep in episodes:
        graph_mgr.graph.add_node(ep.episode_id)
    
    # Test gedig wiring
    print("\nTesting gedig wiring...")
    try:
        graph_mgr._wire_with_gedig(episodes, threshold=0.3)
        print(f"After gedig: {graph_mgr.graph.number_of_nodes()} nodes, {graph_mgr.graph.number_of_edges()} edges")
    except Exception as e:
        print(f"Error in gedig wiring: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    test_wiring_strategies()
    test_graph_manager_directly()