#!/usr/bin/env python3
"""Create the best animated GIF of navigation comparison."""

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


def create_best_gif():
    """Create optimized navigation comparison GIF."""
    # Load config
    config_path = Path(__file__).parent / "maze_experiment_config.yaml"
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    nav_config = MazeNavigatorConfig(**config_dict['navigator'])
    nav_config.exploration_epsilon = 0.0
    
    # Create interesting maze
    np.random.seed(42)
    maze_size = (15, 15)
    maze = SimpleMaze(size=maze_size, maze_type='dfs')  # DFS maze is more solvable
    
    print(f"Creating Best Navigation GIF")
    print(f"Maze: {maze_size[0]}x{maze_size[1]} DFS maze")
    print(f"Start: {maze.start_pos}, Goal: {maze.goal_pos}")
    print("=" * 60)
    
    # Run navigators
    trajectories = {}
    
    # Blind navigator
    blind_nav = BlindExperienceNavigator(nav_config)
    blind_traj = []
    obs = maze.reset()
    blind_traj.append(obs.position)
    
    for step in range(500):
        action = blind_nav.decide_action(obs, maze)
        obs, reward, done, info = maze.step(action)
        blind_traj.append(obs.position)
        if done and maze.agent_pos == maze.goal_pos:
            print(f"Blind: {step + 1} steps, {blind_nav.wall_hits} wall hits")
            break
    
    trajectories['blind'] = {'traj': blind_traj, 'color': '#FF6B6B', 'name': 'Blind (Physical Only)'}
    
    # Visual navigator
    visual_nav = ExperienceMemoryNavigator(nav_config)
    visual_traj = []
    obs = maze.reset()
    visual_traj.append(obs.position)
    
    for step in range(500):
        action = visual_nav.decide_action(obs, maze)
        obs, reward, done, info = maze.step(action)
        visual_traj.append(obs.position)
        if done and maze.agent_pos == maze.goal_pos:
            print(f"Visual: {step + 1} steps")
            break
    
    trajectories['visual'] = {'traj': visual_traj, 'color': '#4ECDC4', 'name': 'Visual+Physical'}
    
    # Create frames
    frames = []
    max_len = max(len(t['traj']) for t in trajectories.values())
    
    # Sample frames for reasonable GIF size
    frame_indices = list(range(0, max_len, 3)) + [max_len - 1]  # Every 3rd frame + last
    
    print(f"\nCreating {len(frame_indices)} frames...")
    
    for i, frame_idx in enumerate(frame_indices):
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Dark background
        ax.set_facecolor('#2B2D42')
        fig.patch.set_facecolor('#2B2D42')
        
        # Draw maze with style
        for row in range(maze.size[0]):
            for col in range(maze.size[1]):
                if maze.grid[row, col] == 1:  # Wall
                    ax.add_patch(Rectangle((col, maze.size[0]-1-row), 1, 1, 
                                         facecolor='#8D99AE', edgecolor='#2B2D42', linewidth=0.5))
        
        # Draw trajectories
        for nav_id, nav_data in trajectories.items():
            traj = nav_data['traj']
            color = nav_data['color']
            
            if frame_idx < len(traj):
                # Trail with gradient
                trail_len = min(frame_idx + 1, len(traj))
                trail_x = [pos[1] + 0.5 for pos in traj[:trail_len]]
                trail_y = [maze.size[0] - 1 - pos[0] + 0.5 for pos in traj[:trail_len]]
                
                # Draw trail segments with fading
                for j in range(len(trail_x) - 1):
                    alpha = 0.2 + 0.6 * (j / len(trail_x))
                    ax.plot(trail_x[j:j+2], trail_y[j:j+2], 
                           color=color, alpha=alpha, linewidth=3)
                
                # Current position
                current_pos = traj[min(frame_idx, len(traj)-1)]
                circle = Circle((current_pos[1] + 0.5, maze.size[0] - 1 - current_pos[0] + 0.5), 
                              0.35, color=color, alpha=1.0, edgecolor='white', linewidth=2)
                ax.add_patch(circle)
        
        # Start and Goal with style
        sx, sy = maze.start_pos
        start_circle = Circle((sy + 0.5, maze.size[0]-1-sx + 0.5), 0.4,
                            facecolor='#06FFA5', edgecolor='white', linewidth=2)
        ax.add_patch(start_circle)
        ax.text(sy + 0.5, maze.size[0]-1-sx + 0.5, 'S', 
               ha='center', va='center', fontsize=16, color='#2B2D42', weight='bold')
        
        gx, gy = maze.goal_pos
        goal_circle = Circle((gy + 0.5, maze.size[0]-1-gx + 0.5), 0.4,
                           facecolor='#FFD60A', edgecolor='white', linewidth=2)
        ax.add_patch(goal_circle)
        ax.text(gy + 0.5, maze.size[0]-1-gx + 0.5, 'G', 
               ha='center', va='center', fontsize=16, color='#2B2D42', weight='bold')
        
        # Title and info
        ax.set_xlim(-0.5, maze.size[1] - 0.5)
        ax.set_ylim(-0.5, maze.size[0] - 0.5)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Top info panel
        info_text = f"Step {frame_idx}"
        ax.text(maze.size[1]/2, maze.size[0] + 0.5, info_text,
               ha='center', va='bottom', fontsize=20, color='white', weight='bold')
        
        # Legend
        for idx, (nav_id, nav_data) in enumerate(trajectories.items()):
            steps = min(frame_idx, len(nav_data['traj'])-1)
            ax.text(0, maze.size[0] - 1 - idx * 1.5, 
                   f"{nav_data['name']}: {steps} steps",
                   fontsize=14, color=nav_data['color'], weight='bold')
        
        # Save frame
        plt.tight_layout()
        frame_path = f'/tmp/best_frame_{i:04d}.png'
        fig.savefig(frame_path, dpi=100, bbox_inches='tight', facecolor='#2B2D42')
        plt.close()
        
        frames.append(imageio.imread(frame_path))
        
        if i % 10 == 0:
            print(f"  Frame {i+1}/{len(frame_indices)}")
    
    # Save as GIF
    output_file = 'maze_navigation_best.gif'
    imageio.mimsave(output_file, frames, fps=8, loop=0)
    print(f"\n✅ Saved animation to {output_file}")
    
    # Create summary statistics
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON:")
    print("=" * 60)
    blind_steps = len(trajectories['blind']['traj']) - 1
    visual_steps = len(trajectories['visual']['traj']) - 1
    speedup = blind_steps / visual_steps
    
    print(f"Blind Navigator:  {blind_steps} steps")
    print(f"Visual Navigator: {visual_steps} steps")
    print(f"Speedup Factor:   {speedup:.2f}x")
    print(f"\nEfficiency Gain: {(1 - visual_steps/blind_steps)*100:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    create_best_gif()