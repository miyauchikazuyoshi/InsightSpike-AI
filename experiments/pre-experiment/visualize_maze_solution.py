#!/usr/bin/env python3
"""Visualize maze solving process."""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from typing import List, Tuple, Dict, Any
import yaml

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from insightspike.environments.maze import SimpleMaze
from insightspike.navigators.pure_gediq_navigator import PureGeDIGNavigator
from insightspike.config.maze_config import MazeNavigatorConfig


def create_maze_image(maze: SimpleMaze, agent_pos: Tuple[int, int], 
                     trajectory: List[Tuple[int, int]], 
                     memory_nodes: Dict[Tuple[int, int], Any]) -> np.ndarray:
    """Create a colored image of the maze state."""
    # Create RGB image
    img = np.ones((maze.height, maze.width, 3))
    
    # Color scheme
    colors = {
        'wall': [0.2, 0.2, 0.2],        # Dark gray
        'empty': [1.0, 1.0, 1.0],       # White
        'start': [0.0, 1.0, 0.0],       # Green
        'goal': [1.0, 0.0, 0.0],        # Red
        'agent': [0.0, 0.0, 1.0],       # Blue
        'trajectory': [0.8, 0.8, 1.0],  # Light blue
        'memory': [1.0, 0.9, 0.7]       # Light yellow
    }
    
    # Draw maze structure
    for i in range(maze.height):
        for j in range(maze.width):
            if maze.grid[i, j] == 1:  # Wall
                img[i, j] = colors['wall']
            else:
                img[i, j] = colors['empty']
    
    # Draw memory nodes
    for pos in memory_nodes.keys():
        if 0 <= pos[0] < maze.height and 0 <= pos[1] < maze.width:
            img[pos[0], pos[1]] = colors['memory']
    
    # Draw trajectory
    for pos in trajectory:
        if pos != agent_pos:  # Don't overwrite agent position
            img[pos[0], pos[1]] = colors['trajectory']
    
    # Draw special positions
    img[maze.start_pos] = colors['start']
    img[maze.goal_pos] = colors['goal']
    img[agent_pos] = colors['agent']
    
    return img


def run_visualization(save_gif: bool = True):
    """Run and visualize a single maze solving episode."""
    # Load config
    config_path = Path(__file__).parent / "maze_experiment_config.yaml"
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    nav_config = MazeNavigatorConfig(**config_dict['navigator'])
    
    # Create maze and navigator
    maze = SimpleMaze(size=(20, 20))
    navigator = PureGeDIGNavigator(nav_config)
    
    # Storage for animation
    frames = []
    trajectories = []
    memory_states = []
    
    # Run episode
    obs = maze.reset()
    done = False
    step = 0
    max_steps = 1000
    
    print("Starting maze solving visualization...")
    
    while not done and step < max_steps:
        # Store current state
        trajectories.append(list(maze.trajectory))
        memory_states.append(dict(navigator.memory_nodes))
        
        # Create frame
        frame = create_maze_image(maze, maze.agent_pos, 
                                trajectories[-1], memory_states[-1])
        frames.append(frame)
        
        # Decide and take action
        action = navigator.decide_action(obs, maze)
        obs, reward, done, info = maze.step(action)
        
        step += 1
        
        # Print progress
        if step % 50 == 0:
            print(f"Step {step}: Position {maze.agent_pos}, "
                  f"Memory nodes: {len(navigator.memory_nodes)}")
    
    # Final frame
    trajectories.append(list(maze.trajectory))
    memory_states.append(dict(navigator.memory_nodes))
    frame = create_maze_image(maze, maze.agent_pos, 
                            trajectories[-1], memory_states[-1])
    frames.append(frame)
    
    print(f"\nEpisode finished!")
    print(f"Success: {maze.agent_pos == maze.goal_pos}")
    print(f"Total steps: {step}")
    print(f"Total memory nodes: {len(navigator.memory_nodes)}")
    
    # Create animation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Maze view
    im1 = ax1.imshow(frames[0], interpolation='nearest')
    ax1.set_title('Maze Navigation')
    ax1.axis('off')
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc=[0.2, 0.2, 0.2], label='Wall'),
        plt.Rectangle((0, 0), 1, 1, fc=[0.0, 1.0, 0.0], label='Start'),
        plt.Rectangle((0, 0), 1, 1, fc=[1.0, 0.0, 0.0], label='Goal'),
        plt.Rectangle((0, 0), 1, 1, fc=[0.0, 0.0, 1.0], label='Agent'),
        plt.Rectangle((0, 0), 1, 1, fc=[0.8, 0.8, 1.0], label='Path'),
        plt.Rectangle((0, 0), 1, 1, fc=[1.0, 0.9, 0.7], label='Memory')
    ]
    ax1.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Stats view
    ax2.axis('off')
    stats_text = ax2.text(0.1, 0.5, '', fontsize=12, 
                         verticalalignment='center', fontfamily='monospace')
    
    def update(frame_idx):
        # Update maze image
        im1.set_array(frames[frame_idx])
        
        # Update stats
        stats = f"Step: {frame_idx}\n"
        stats += f"Position: {trajectories[frame_idx][-1] if trajectories[frame_idx] else 'N/A'}\n"
        stats += f"Path length: {len(trajectories[frame_idx])}\n"
        stats += f"Memory nodes: {len(memory_states[frame_idx])}\n"
        stats += f"Unique positions visited: {len(set(trajectories[frame_idx]))}\n"
        
        # Memory node breakdown
        node_types = {}
        for node in memory_states[frame_idx].values():
            node_type = node.node_type
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        stats += "\nMemory breakdown:\n"
        for node_type, count in sorted(node_types.items()):
            stats += f"  {node_type}: {count}\n"
        
        stats_text.set_text(stats)
        
        return [im1, stats_text]
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=len(frames),
                                  interval=100, blit=True, repeat=True)
    
    # Save as GIF if requested
    if save_gif:
        output_path = Path(__file__).parent / "results" / "maze_solving.gif"
        output_path.parent.mkdir(exist_ok=True)
        
        print(f"\nSaving animation to {output_path}...")
        anim.save(output_path, writer='pillow', fps=10)
        print("Animation saved!")
    
    plt.tight_layout()
    plt.show()
    
    return frames, trajectories, memory_states


def create_summary_plot(trajectories: List[List[Tuple[int, int]]], 
                       memory_states: List[Dict],
                       maze: SimpleMaze):
    """Create a summary plot showing the complete path and final memory map."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Complete trajectory
    ax1.set_title('Complete Path Taken')
    
    # Create base maze image
    maze_img = np.ones((maze.height, maze.width, 3))
    for i in range(maze.height):
        for j in range(maze.width):
            if maze.grid[i, j] == 1:  # Wall
                maze_img[i, j] = [0.2, 0.2, 0.2]
    
    # Mark start and goal
    maze_img[maze.start_pos] = [0.0, 1.0, 0.0]
    maze_img[maze.goal_pos] = [1.0, 0.0, 0.0]
    
    ax1.imshow(maze_img, interpolation='nearest')
    
    # Draw path as line
    if trajectories and trajectories[-1]:
        path = trajectories[-1]
        path_y = [p[0] for p in path]
        path_x = [p[1] for p in path]
        ax1.plot(path_x, path_y, 'b-', linewidth=2, alpha=0.7)
        ax1.plot(path_x[0], path_y[0], 'go', markersize=10, label='Start')
        ax1.plot(path_x[-1], path_y[-1], 'ro', markersize=10, label='End')
    
    ax1.axis('off')
    ax1.legend()
    
    # Plot 2: Final memory map
    ax2.set_title('Final Memory Map')
    
    # Create memory map image
    memory_img = maze_img.copy()
    
    # Color memory nodes by type
    node_colors = {
        'wall': [0.8, 0.2, 0.2],
        'junction': [0.2, 0.2, 0.8],
        'corridor': [0.2, 0.8, 0.2],
        'dead_end': [0.8, 0.8, 0.2],
        'goal': [1.0, 0.5, 0.0]
    }
    
    if memory_states and memory_states[-1]:
        for pos, node in memory_states[-1].items():
            if 0 <= pos[0] < maze.height and 0 <= pos[1] < maze.width:
                color = node_colors.get(node.node_type, [0.5, 0.5, 0.5])
                memory_img[pos[0], pos[1]] = color
    
    ax2.imshow(memory_img, interpolation='nearest')
    ax2.axis('off')
    
    # Add legend for memory types
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc=color, label=node_type.replace('_', ' ').title())
        for node_type, color in node_colors.items()
    ]
    ax2.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    
    # Save summary
    output_path = Path(__file__).parent / "results" / "maze_solving_summary.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Summary plot saved to {output_path}")
    
    plt.show()


if __name__ == "__main__":
    # Run visualization
    frames, trajectories, memory_states = run_visualization(save_gif=True)
    
    # Create summary plot
    maze = SimpleMaze(size=(20, 20))
    create_summary_plot(trajectories, memory_states, maze)