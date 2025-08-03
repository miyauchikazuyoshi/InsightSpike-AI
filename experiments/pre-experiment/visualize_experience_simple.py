#!/usr/bin/env python3
"""Simple visualization of key frames."""

import sys
from pathlib import Path
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.append(str(Path(__file__).parent.parent.parent))

from insightspike.environments.maze import SimpleMaze
from insightspike.navigators.experience_memory_navigator import ExperienceMemoryNavigator, ExperienceType
from insightspike.config.maze_config import MazeNavigatorConfig


def visualize_key_frames():
    """Visualize key frames of navigation."""
    # Load config
    config_path = Path(__file__).parent / "maze_experiment_config.yaml"
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    nav_config = MazeNavigatorConfig(**config_dict['navigator'])
    
    # Create maze
    np.random.seed(42)
    maze = SimpleMaze(size=(15, 15), maze_type='dfs')
    
    # Create navigator
    navigator = ExperienceMemoryNavigator(nav_config)
    
    # Key steps to visualize
    key_steps = [0, 20, 50, 100, 150]
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, len(key_steps), figsize=(20, 4))
    
    # Run navigation
    obs = maze.reset()
    positions = []
    
    for step in range(200):
        positions.append(maze.agent_pos)
        
        # Visualize if key step
        if step in key_steps:
            idx = key_steps.index(step)
            ax = axes[idx]
            
            # Draw maze
            for i in range(maze.size[0]):
                for j in range(maze.size[1]):
                    if maze.grid[i, j] == 1:  # Wall
                        ax.add_patch(Rectangle((j, maze.size[0]-1-i), 1, 1, 
                                             facecolor='black'))
                    elif (i, j) in navigator.memory_nodes:
                        # Visited path
                        ax.add_patch(Rectangle((j, maze.size[0]-1-i), 1, 1, 
                                             facecolor='lightblue', alpha=0.5))
            
            # Draw start and goal
            sx, sy = maze.start_pos
            ax.text(sy + 0.5, maze.size[0]-1-sx + 0.5, 'S', 
                   ha='center', va='center', fontsize=12, color='green', weight='bold')
            
            gx, gy = maze.goal_pos
            ax.text(gy + 0.5, maze.size[0]-1-gx + 0.5, 'G', 
                   ha='center', va='center', fontsize=12, color='red', weight='bold')
            
            # Draw agent
            agent_x, agent_y = maze.agent_pos
            circle = plt.Circle((agent_y + 0.5, maze.size[0]-1-agent_x + 0.5), 0.3, 
                              color='blue', zorder=10)
            ax.add_patch(circle)
            
            # Setup
            ax.set_xlim(0, maze.size[1])
            ax.set_ylim(0, maze.size[0])
            ax.set_aspect('equal')
            ax.set_title(f'Step {step}')
            ax.axis('off')
        
        # Make decision
        action = navigator.decide_action(obs, maze)
        obs, reward, done, info = maze.step(action)
        
        if done and maze.agent_pos == maze.goal_pos:
            print(f"Goal reached in {step + 1} steps!")
            break
    
    plt.tight_layout()
    plt.savefig('experience_navigation_frames.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved visualization to experience_navigation_frames.png")


if __name__ == "__main__":
    visualize_key_frames()