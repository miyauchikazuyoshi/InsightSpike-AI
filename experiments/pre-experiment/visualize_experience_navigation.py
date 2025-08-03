#!/usr/bin/env python3
"""Visualize experience navigation as animated GIF."""

import sys
from pathlib import Path
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import imageio
import os

sys.path.append(str(Path(__file__).parent.parent.parent))

from insightspike.environments.maze import SimpleMaze
from insightspike.navigators.experience_memory_navigator import ExperienceMemoryNavigator, ExperienceType
from insightspike.config.maze_config import MazeNavigatorConfig


def create_navigation_gif():
    """Create GIF of navigation process."""
    # Load config
    config_path = Path(__file__).parent / "maze_experiment_config.yaml"
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    nav_config = MazeNavigatorConfig(**config_dict['navigator'])
    
    # Create maze
    np.random.seed(42)
    maze = SimpleMaze(size=(15, 15), maze_type='dfs')
    
    print("Creating navigation animation...")
    
    # Create navigator
    navigator = ExperienceMemoryNavigator(nav_config)
    
    # Store frames
    frames = []
    positions = []
    
    # Run navigation and capture frames
    obs = maze.reset()
    max_steps = 300
    done = False
    
    for step in range(max_steps):
        if not done:
            # Store current state
            positions.append(maze.agent_pos)
            
            # Create frame
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # Draw maze
            for i in range(maze.size[0]):
                for j in range(maze.size[1]):
                    if maze.grid[i, j] == 1:  # Wall
                        ax.add_patch(Rectangle((j, maze.size[0]-1-i), 1, 1, 
                                             facecolor='black'))
                    else:  # Path
                        # Color based on experience
                        if (i, j) in navigator.memory_nodes:
                            node = navigator.memory_nodes[(i, j)]
                            # Light blue for visited
                            ax.add_patch(Rectangle((j, maze.size[0]-1-i), 1, 1, 
                                                 facecolor='lightblue', alpha=0.5))
                            
                            # Show wall detection (visual experience)
                            for direction, exp in node.experiences.items():
                                if exp.visual == ExperienceType.VISUAL_WALL:
                                    # Draw small red line indicating detected wall
                                    cx, cy = j + 0.5, maze.size[0]-1-i + 0.5
                                    if direction == 0:  # Up
                                        ax.plot([cx-0.3, cx+0.3], [cy+0.4, cy+0.4], 'r-', linewidth=2)
                                    elif direction == 1:  # Right
                                        ax.plot([cx+0.4, cx+0.4], [cy-0.3, cy+0.3], 'r-', linewidth=2)
                                    elif direction == 2:  # Down
                                        ax.plot([cx-0.3, cx+0.3], [cy-0.4, cy-0.4], 'r-', linewidth=2)
                                    elif direction == 3:  # Left
                                        ax.plot([cx-0.4, cx-0.4], [cy-0.3, cy+0.3], 'r-', linewidth=2)
            
            # Draw start
            sx, sy = maze.start_pos
            ax.text(sy + 0.5, maze.size[0]-1-sx + 0.5, 'S', 
                   ha='center', va='center', fontsize=20, color='green', weight='bold')
            
            # Draw goal
            gx, gy = maze.goal_pos
            ax.text(gy + 0.5, maze.size[0]-1-gx + 0.5, 'G', 
                   ha='center', va='center', fontsize=20, color='red', weight='bold')
            
            # Draw agent
            agent_x, agent_y = maze.agent_pos
            circle = plt.Circle((agent_y + 0.5, maze.size[0]-1-agent_x + 0.5), 0.3, 
                              color='blue', zorder=10)
            ax.add_patch(circle)
            
            # Draw path trail
            if len(positions) > 1:
                path_x = [p[1] + 0.5 for p in positions[-20:]]  # Last 20 positions
                path_y = [maze.size[0]-1-p[0] + 0.5 for p in positions[-20:]]
                ax.plot(path_x, path_y, 'b-', alpha=0.3, linewidth=2)
            
            # Setup plot
            ax.set_xlim(0, maze.size[1])
            ax.set_ylim(0, maze.size[0])
            ax.set_aspect('equal')
            ax.set_xticks(range(maze.size[1]+1))
            ax.set_yticks(range(maze.size[0]+1))
            ax.grid(True, alpha=0.3)
            # Count walls detected
            wall_count = sum(1 for n in navigator.memory_nodes.values() 
                           for e in n.experiences.values() 
                           if e.visual == ExperienceType.VISUAL_WALL)
            
            ax.set_title(f'Experience Memory Navigation - Step {step}\n'
                        f'Positions visited: {len(navigator.memory_nodes)}, '
                        f'Walls detected: {wall_count}',
                        fontsize=16)
            
            # Save frame
            plt.tight_layout()
            fig.canvas.draw()
            # Use buffer_rgba() for compatibility
            buf = fig.canvas.buffer_rgba()
            image = np.asarray(buf)
            # Convert RGBA to RGB
            image = image[:,:,:3]
            frames.append(image)
            plt.close(fig)
            
            # Make decision
            action = navigator.decide_action(obs, maze)
            obs, reward, done, info = maze.step(action)
            
            if done and maze.agent_pos == maze.goal_pos:
                print(f"Goal reached in {step + 1} steps!")
                # Add a few frames at the goal
                for _ in range(10):
                    frames.append(frames[-1])
                break
    
    # Save as GIF
    output_path = Path(__file__).parent / "experience_navigation.gif"
    imageio.mimsave(output_path, frames, fps=5)
    print(f"Animation saved to: {output_path}")
    
    # Also save some key frames as PNG
    key_steps = [0, 20, 50, 100, len(frames)-1]
    for i, step in enumerate(key_steps):
        if step < len(frames):
            imageio.imwrite(f"frame_{i}_step_{step}.png", frames[step])


if __name__ == "__main__":
    create_navigation_gif()