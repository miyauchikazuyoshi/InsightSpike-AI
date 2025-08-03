#!/usr/bin/env python3
"""Create static visualization of maze solving."""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict, Any
import yaml

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from insightspike.environments.maze import SimpleMaze
from insightspike.navigators.pure_gediq_navigator import PureGeDIGNavigator
from insightspike.config.maze_config import MazeNavigatorConfig


def run_and_visualize():
    """Run maze solving and create visualization."""
    # Load config
    config_path = Path(__file__).parent / "maze_experiment_config.yaml"
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    nav_config = MazeNavigatorConfig(**config_dict['navigator'])
    
    # Create maze and navigator
    # Try different maze types
    maze_type = 'dfs'  # Can be 'simple', 'complex', 'spiral', 'rooms', 'dfs', 'kruskal', 'prim'
    maze = SimpleMaze(size=(15, 15), maze_type=maze_type)  # Must be odd dimensions
    navigator = PureGeDIGNavigator(nav_config)
    
    # Run episode
    obs = maze.reset()
    done = False
    step = 0
    max_steps = 1000
    
    print("Running maze solving...")
    
    # Store key moments
    snapshots = []
    snapshot_steps = [0, 10, 25, 50, 100, 200, 500]
    
    while not done and step < max_steps:
        # Store snapshot if needed
        if step in snapshot_steps or (done and step not in snapshot_steps):
            snapshots.append({
                'step': step,
                'position': maze.agent_pos,
                'trajectory': list(maze.trajectory),
                'memory_nodes': dict(navigator.memory_nodes)
            })
        
        # Decide and take action
        action = navigator.decide_action(obs, maze)
        obs, reward, done, info = maze.step(action)
        
        step += 1
    
    # Final snapshot
    snapshots.append({
        'step': step,
        'position': maze.agent_pos,
        'trajectory': list(maze.trajectory),
        'memory_nodes': dict(navigator.memory_nodes)
    })
    
    print(f"\nEpisode finished!")
    print(f"Success: {maze.agent_pos == maze.goal_pos}")
    print(f"Total steps: {step}")
    print(f"Total memory nodes: {len(navigator.memory_nodes)}")
    
    # Create visualization with snapshots
    n_snapshots = min(6, len(snapshots))
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx in range(n_snapshots):
        ax = axes[idx]
        snapshot = snapshots[min(idx, len(snapshots)-1)]
        
        # Create maze image
        img = np.ones((maze.height, maze.width, 3))
        
        # Draw walls
        for i in range(maze.height):
            for j in range(maze.width):
                if maze.grid[i, j] == 1:
                    img[i, j] = [0.2, 0.2, 0.2]
        
        # Draw memory nodes
        for pos, node in snapshot['memory_nodes'].items():
            if 0 <= pos[0] < maze.height and 0 <= pos[1] < maze.width:
                # Color by type
                if node.node_type == 'junction':
                    img[pos[0], pos[1]] = [0.9, 0.9, 0.5]  # Yellow
                elif node.node_type == 'wall':
                    img[pos[0], pos[1]] = [1.0, 0.7, 0.7]  # Light red
                else:
                    img[pos[0], pos[1]] = [0.9, 0.9, 0.9]  # Light gray
        
        # Draw trajectory
        for pos in snapshot['trajectory'][:-1]:
            if img[pos[0], pos[1], 0] > 0.8:  # Only if not wall or special
                img[pos[0], pos[1]] = [0.7, 0.7, 1.0]  # Light blue
        
        # Draw special positions
        img[maze.start_pos] = [0.0, 1.0, 0.0]  # Green
        img[maze.goal_pos] = [1.0, 0.0, 0.0]   # Red
        
        # Draw current position
        if snapshot['position'] != maze.goal_pos:
            img[snapshot['position']] = [0.0, 0.0, 1.0]  # Blue
        
        ax.imshow(img, interpolation='nearest')
        ax.set_title(f"Step {snapshot['step']}")
        ax.axis('off')
    
    plt.suptitle(f"Pure geDIG Maze Navigation (Success: {maze.agent_pos == maze.goal_pos})", 
                 fontsize=16)
    
    # Add text info
    info_text = f"Total steps: {step}\n"
    info_text += f"Memory nodes: {len(navigator.memory_nodes)}\n"
    info_text += f"Path length: {len(maze.trajectory)}\n"
    info_text += f"Unique positions: {len(set(maze.trajectory))}"
    
    plt.figtext(0.95, 0.05, info_text, ha='right', va='bottom', 
                fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(__file__).parent / "results" / "maze_solving_snapshots.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to {output_path}")
    
    plt.show()
    
    # Create path-only visualization
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Draw maze
    maze_img = np.ones((maze.height, maze.width, 3))
    for i in range(maze.height):
        for j in range(maze.width):
            if maze.grid[i, j] == 1:
                maze_img[i, j] = [0.2, 0.2, 0.2]
    
    maze_img[maze.start_pos] = [0.0, 1.0, 0.0]
    maze_img[maze.goal_pos] = [1.0, 0.0, 0.0]
    
    ax.imshow(maze_img, interpolation='nearest')
    
    # Draw path as line
    if maze.trajectory:
        path_y = [p[0] for p in maze.trajectory]
        path_x = [p[1] for p in maze.trajectory]
        ax.plot(path_x, path_y, 'b-', linewidth=3, alpha=0.7)
        ax.plot(path_x[0], path_y[0], 'go', markersize=12)
        ax.plot(path_x[-1], path_y[-1], 'ro', markersize=12)
    
    ax.set_title(f"Complete Path (Steps: {len(maze.trajectory)})")
    ax.axis('off')
    
    plt.tight_layout()
    
    # Save path visualization
    output_path2 = Path(__file__).parent / "results" / "maze_path.png"
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"Path visualization saved to {output_path2}")
    
    plt.show()


if __name__ == "__main__":
    run_and_visualize()