#!/usr/bin/env python3
"""Create simple animated GIF of navigation."""

import sys
from pathlib import Path
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import imageio

sys.path.append(str(Path(__file__).parent.parent.parent))

from insightspike.environments.maze import SimpleMaze
from insightspike.navigators.blind_experience_navigator import BlindExperienceNavigator
from insightspike.navigators.experience_memory_navigator import ExperienceMemoryNavigator
from insightspike.config.maze_config import MazeNavigatorConfig

# Suppress verbose output
import logging
logging.getLogger().setLevel(logging.WARNING)


def create_simple_gif():
    """Create a simple navigation GIF."""
    # Load config
    config_path = Path(__file__).parent / "maze_experiment_config.yaml"
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    nav_config = MazeNavigatorConfig(**config_dict['navigator'])
    nav_config.exploration_epsilon = 0.0
    
    # Create maze
    np.random.seed(42)
    maze_size = (10, 10)
    maze = SimpleMaze(size=maze_size, maze_type='dfs')
    
    print(f"Creating Navigation GIF for {maze_size[0]}x{maze_size[1]} maze")
    print(f"Start: {maze.start_pos}, Goal: {maze.goal_pos}")
    
    # Run blind navigator
    blind_nav = BlindExperienceNavigator(nav_config)
    blind_trajectory = []
    
    obs = maze.reset()
    blind_trajectory.append(obs.position)
    
    max_steps = 200
    for step in range(max_steps):
        action = blind_nav.decide_action(obs, maze)
        obs, reward, done, info = maze.step(action)
        blind_trajectory.append(obs.position)
        
        if done and maze.agent_pos == maze.goal_pos:
            print(f"Blind navigator reached goal in {step + 1} steps!")
            break
    
    # Run visual navigator
    visual_nav = ExperienceMemoryNavigator(nav_config)
    visual_trajectory = []
    
    obs = maze.reset()
    visual_trajectory.append(obs.position)
    
    for step in range(max_steps):
        action = visual_nav.decide_action(obs, maze)
        obs, reward, done, info = maze.step(action)
        visual_trajectory.append(obs.position)
        
        if done and maze.agent_pos == maze.goal_pos:
            print(f"Visual navigator reached goal in {step + 1} steps!")
            break
    
    # Create frames
    frames = []
    max_traj_len = max(len(blind_trajectory), len(visual_trajectory))
    
    print(f"\nCreating {max_traj_len} frames...")
    
    for frame_idx in range(0, max_traj_len, 3):  # Every 3rd frame
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Blind navigator subplot
        ax1.set_title('Blind Navigator (Physical Only)', fontsize=14)
        draw_maze_frame(ax1, maze, blind_trajectory, frame_idx, 'red')
        
        # Visual navigator subplot
        ax2.set_title('Visual+Physical Navigator', fontsize=14)
        draw_maze_frame(ax2, maze, visual_trajectory, frame_idx, 'blue')
        
        # Overall title
        fig.suptitle(f'Step {frame_idx}', fontsize=16)
        
        # Save frame
        plt.tight_layout()
        fig.savefig(f'/tmp/frame_{frame_idx:04d}.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        # Read and append to frames
        frames.append(imageio.imread(f'/tmp/frame_{frame_idx:04d}.png'))
    
    # Save as GIF
    output_file = 'maze_navigation_comparison.gif'
    imageio.mimsave(output_file, frames, fps=5)
    print(f"\n✅ Saved GIF to {output_file}")
    
    # Also create final comparison
    create_final_frame(maze, blind_trajectory, visual_trajectory)


def draw_maze_frame(ax, maze, trajectory, frame_idx, color):
    """Draw a single frame of the maze."""
    # Draw maze walls
    for i in range(maze.size[0]):
        for j in range(maze.size[1]):
            if maze.grid[i, j] == 1:  # Wall
                ax.add_patch(Rectangle((j, maze.size[0]-1-i), 1, 1, 
                                     facecolor='black'))
    
    # Draw trajectory up to current frame
    if frame_idx < len(trajectory):
        trail_x = [pos[1] + 0.5 for pos in trajectory[:frame_idx+1]]
        trail_y = [maze.size[0] - 1 - pos[0] + 0.5 for pos in trajectory[:frame_idx+1]]
        ax.plot(trail_x, trail_y, color=color, alpha=0.5, linewidth=2)
        
        # Current position
        current_pos = trajectory[frame_idx]
        ax.add_patch(Circle((current_pos[1] + 0.5, maze.size[0] - 1 - current_pos[0] + 0.5), 
                          0.3, color=color))
    
    # Start and goal
    sx, sy = maze.start_pos
    ax.text(sy + 0.5, maze.size[0]-1-sx + 0.5, 'S', 
           ha='center', va='center', fontsize=12, color='green', weight='bold')
    
    gx, gy = maze.goal_pos
    ax.text(gy + 0.5, maze.size[0]-1-gx + 0.5, 'G', 
           ha='center', va='center', fontsize=12, color='red', weight='bold')
    
    # Setup
    ax.set_xlim(0, maze.size[1])
    ax.set_ylim(0, maze.size[0])
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Step counter
    steps = min(frame_idx, len(trajectory)-1)
    ax.text(0.5, -0.5, f'Steps: {steps}', fontsize=10)


def create_final_frame(maze, blind_traj, visual_traj):
    """Create final comparison frame."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw maze
    for i in range(maze.size[0]):
        for j in range(maze.size[1]):
            if maze.grid[i, j] == 1:
                ax.add_patch(Rectangle((j, maze.size[0]-1-i), 1, 1, 
                                     facecolor='black'))
    
    # Draw both trajectories
    # Blind
    blind_x = [pos[1] + 0.5 for pos in blind_traj]
    blind_y = [maze.size[0] - 1 - pos[0] + 0.5 for pos in blind_traj]
    ax.plot(blind_x, blind_y, 'r-', alpha=0.7, linewidth=3, 
           label=f'Blind: {len(blind_traj)-1} steps')
    
    # Visual
    visual_x = [pos[1] + 0.5 for pos in visual_traj]
    visual_y = [maze.size[0] - 1 - pos[0] + 0.5 for pos in visual_traj]
    ax.plot(visual_x, visual_y, 'b-', alpha=0.7, linewidth=3,
           label=f'Visual: {len(visual_traj)-1} steps')
    
    # Start and goal
    sx, sy = maze.start_pos
    ax.add_patch(Circle((sy + 0.5, maze.size[0]-1-sx + 0.5), 0.4, 
                       facecolor='green', alpha=0.8))
    ax.text(sy + 0.5, maze.size[0]-1-sx + 0.5, 'S', 
           ha='center', va='center', fontsize=14, color='white', weight='bold')
    
    gx, gy = maze.goal_pos
    ax.add_patch(Circle((gy + 0.5, maze.size[0]-1-gx + 0.5), 0.4,
                       facecolor='gold', alpha=0.8))
    ax.text(gy + 0.5, maze.size[0]-1-gx + 0.5, 'G', 
           ha='center', va='center', fontsize=14, color='black', weight='bold')
    
    ax.set_xlim(0, maze.size[1])
    ax.set_ylim(0, maze.size[0])
    ax.set_aspect('equal')
    ax.set_title('Navigation Comparison', fontsize=16)
    ax.legend(fontsize=12)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('maze_navigation_final.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Saved final comparison to maze_navigation_final.png")


if __name__ == "__main__":
    create_simple_gif()