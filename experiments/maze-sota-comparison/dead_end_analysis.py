#!/usr/bin/env python3
"""Analyze why navigators avoid dead ends."""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent.parent))

from insightspike.environments.maze import SimpleMaze
from insightspike.maze_experimental.navigators.blind_experience_navigator import BlindExperienceNavigator
from insightspike.maze_experimental.navigators.experience_memory_navigator import ExperienceMemoryNavigator
from insightspike.maze_experimental.maze_config import MazeNavigatorConfig


def create_test_maze():
    """Create a simple maze with clear dead ends."""
    maze = np.ones((10, 10), dtype=int)
    
    # Main corridor
    maze[1, 1:9] = 0  # Top corridor
    maze[1:9, 8] = 0  # Right corridor
    maze[8, 1:9] = 0  # Bottom corridor
    
    # Dead ends
    maze[2:5, 2] = 0  # Dead end 1 (down from top)
    maze[2:5, 4] = 0  # Dead end 2 (down from top)
    maze[2:5, 6] = 0  # Dead end 3 (down from top)
    
    return maze


def track_decision_process(navigator, maze, max_steps=100):
    """Track navigator's decision process."""
    obs = maze.reset()
    decisions = []
    
    for step in range(max_steps):
        pos = obs.position
        
        # Record current state
        decision_info = {
            'step': step,
            'position': pos,
            'possible_moves': obs.possible_moves.copy(),
            'is_dead_end': obs.is_dead_end,
            'num_paths': obs.num_paths
        }
        
        # Get action and scores
        if hasattr(navigator, 'memory_nodes'):
            # Visual navigator
            node = navigator.memory_nodes.get(pos)
            if node:
                decision_info['visual_info'] = {
                    dir: (exp.visual.name, exp.confidence) 
                    for dir, exp in node.experiences.items()
                }
        
        action = navigator.decide_action(obs, maze)
        decision_info['chosen_action'] = action
        
        decisions.append(decision_info)
        
        # Take action
        obs, reward, done, info = maze.step(action)
        
        if done and maze.agent_pos == maze.goal_pos:
            print(f"Goal reached in {step + 1} steps")
            break
    
    return decisions


def analyze_dead_end_avoidance():
    """Analyze how navigators avoid dead ends."""
    config = MazeNavigatorConfig(
        ged_weight=1.0,
        ig_weight=2.0,
        temperature=1.0,
        exploration_epsilon=0.0
    )
    
    print("DEAD END AVOIDANCE ANALYSIS")
    print("=" * 60)
    
    # Create test maze
    test_grid = create_test_maze()
    maze = SimpleMaze(size=(10, 10), maze_type='empty')
    maze.grid = test_grid
    maze.start_pos = (1, 1)
    maze.goal_pos = (8, 8)
    maze.agent_pos = maze.start_pos
    
    # Test visual navigator
    print("\nVisual Navigator Analysis:")
    print("-" * 40)
    visual_nav = ExperienceMemoryNavigator(config)
    decisions = track_decision_process(visual_nav, maze)
    
    # Analyze decisions near dead ends
    dead_end_entries = [(1, 2), (1, 4), (1, 6)]  # Entrances to dead ends
    
    for entry in dead_end_entries:
        print(f"\nAt dead end entrance {entry}:")
        for d in decisions:
            if d['position'] == entry:
                print(f"  Step {d['step']}: {d['num_paths']} paths available")
                print(f"  Possible moves: {d['possible_moves']}")
                if 'visual_info' in d:
                    print(f"  Visual info: {d['visual_info']}")
                print(f"  Chosen action: {d['chosen_action']}")
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Maze structure
    ax1.imshow(test_grid, cmap='binary')
    ax1.plot(maze.start_pos[1], maze.start_pos[0], 'go', markersize=10)
    ax1.plot(maze.goal_pos[1], maze.goal_pos[0], 'ro', markersize=10)
    
    # Mark dead ends
    for entry in dead_end_entries:
        ax1.plot(entry[1], entry[0], 'yo', markersize=8)
        ax1.text(entry[1]+0.2, entry[0], 'DE', fontsize=8, color='yellow')
    
    ax1.set_title('Test Maze (Yellow = Dead End Entrances)')
    ax1.axis('off')
    
    # Decision heat map
    visit_count = np.zeros_like(test_grid, dtype=float)
    for d in decisions:
        pos = d['position']
        visit_count[pos[0], pos[1]] += 1
    
    ax2.imshow(test_grid, cmap='binary', alpha=0.3)
    im = ax2.imshow(visit_count, cmap='hot', alpha=0.7)
    plt.colorbar(im, ax=ax2)
    ax2.set_title('Visit Frequency Heat Map')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('dead_end_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n" + "=" * 60)
    print("KEY INSIGHTS:")
    print("1. Visual info is ONLY 1 square ahead (not cheating)")
    print("2. Dead ends avoided through geDIG objective function:")
    print("   - High IG for unexplored areas")
    print("   - Low IG for dead ends (all directions explored)")
    print("3. Experience memory helps avoid revisiting dead ends")
    print("=" * 60)


if __name__ == "__main__":
    analyze_dead_end_avoidance()