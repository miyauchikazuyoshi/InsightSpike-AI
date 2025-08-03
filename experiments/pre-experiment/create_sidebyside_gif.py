#!/usr/bin/env python3
"""Create side-by-side animated GIF comparison."""

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


def create_sidebyside_gif():
    """Create side-by-side navigation comparison GIF."""
    # Load config
    config_path = Path(__file__).parent / "maze_experiment_config.yaml"
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    nav_config = MazeNavigatorConfig(**config_dict['navigator'])
    nav_config.exploration_epsilon = 0.0
    
    # Create maze with interesting pattern
    np.random.seed(123)  # Different seed for variety
    maze_size = (12, 12)
    maze = SimpleMaze(size=maze_size, maze_type='spiral')
    
    print(f"Creating Side-by-Side Navigation GIF")
    print(f"Maze: {maze_size[0]}x{maze_size[1]} spiral")
    print(f"Start: {maze.start_pos}, Goal: {maze.goal_pos}")
    print("=" * 60)
    
    # Run both navigators
    # Blind
    blind_nav = BlindExperienceNavigator(nav_config)
    blind_traj = []
    blind_walls = []
    
    obs = maze.reset()
    blind_traj.append(obs.position)
    
    for step in range(400):
        old_pos = obs.position
        action = blind_nav.decide_action(obs, maze)
        obs, reward, done, info = maze.step(action)
        blind_traj.append(obs.position)
        
        # Track wall hits
        if old_pos == obs.position and step > 0:
            blind_walls.append((step, old_pos))
        
        if done and maze.agent_pos == maze.goal_pos:
            print(f"Blind: {step + 1} steps, {blind_nav.wall_hits} wall hits")
            break
    
    # Visual
    visual_nav = ExperienceMemoryNavigator(nav_config)
    visual_traj = []
    
    obs = maze.reset()
    visual_traj.append(obs.position)
    
    for step in range(400):
        action = visual_nav.decide_action(obs, maze)
        obs, reward, done, info = maze.step(action)
        visual_traj.append(obs.position)
        
        if done and maze.agent_pos == maze.goal_pos:
            print(f"Visual: {step + 1} steps")
            break
    
    # Create frames
    frames = []
    max_len = max(len(blind_traj), len(visual_traj))
    
    # Sample frames
    frame_indices = list(range(0, max_len, 2)) + [max_len - 1]
    
    print(f"\nCreating {len(frame_indices)} frames...")
    
    for i, frame_idx in enumerate(frame_indices):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        
        # Style
        for ax in [ax1, ax2]:
            ax.set_facecolor('#1a1a2e')
        fig.patch.set_facecolor('#0f0f1e')
        
        # Draw maze for both
        for ax in [ax1, ax2]:
            for row in range(maze.size[0]):
                for col in range(maze.size[1]):
                    if maze.grid[row, col] == 1:  # Wall
                        ax.add_patch(Rectangle((col, maze.size[0]-1-row), 1, 1, 
                                             facecolor='#16213e', edgecolor='#0f3460', linewidth=0.5))
        
        # Blind navigator (left)
        ax1.set_title('Blind Navigator (Physical Only)', fontsize=16, color='#e94560', pad=20)
        
        if frame_idx < len(blind_traj):
            # Trail
            trail_x = [pos[1] + 0.5 for pos in blind_traj[:frame_idx+1]]
            trail_y = [maze.size[0] - 1 - pos[0] + 0.5 for pos in blind_traj[:frame_idx+1]]
            
            for j in range(len(trail_x) - 1):
                alpha = 0.3 + 0.5 * (j / len(trail_x))
                ax1.plot(trail_x[j:j+2], trail_y[j:j+2], 
                       color='#e94560', alpha=alpha, linewidth=2.5)
            
            # Wall hits
            for hit_step, hit_pos in blind_walls:
                if hit_step <= frame_idx:
                    ax1.add_patch(Circle((hit_pos[1] + 0.5, maze.size[0] - 1 - hit_pos[0] + 0.5),
                                       0.15, color='#ff6b6b', alpha=0.6))
            
            # Current position
            current = blind_traj[min(frame_idx, len(blind_traj)-1)]
            ax1.add_patch(Circle((current[1] + 0.5, maze.size[0] - 1 - current[0] + 0.5),
                               0.3, color='#e94560', edgecolor='white', linewidth=2))
            
            ax1.text(0.5, -0.5, f'Steps: {min(frame_idx, len(blind_traj)-1)}', 
                    transform=ax1.transAxes, fontsize=12, color='#e94560')
        
        # Visual navigator (right)
        ax2.set_title('Visual+Physical Navigator', fontsize=16, color='#00d9ff', pad=20)
        
        if frame_idx < len(visual_traj):
            # Trail
            trail_x = [pos[1] + 0.5 for pos in visual_traj[:frame_idx+1]]
            trail_y = [maze.size[0] - 1 - pos[0] + 0.5 for pos in visual_traj[:frame_idx+1]]
            
            for j in range(len(trail_x) - 1):
                alpha = 0.3 + 0.5 * (j / len(trail_x))
                ax2.plot(trail_x[j:j+2], trail_y[j:j+2], 
                       color='#00d9ff', alpha=alpha, linewidth=2.5)
            
            # Current position
            current = visual_traj[min(frame_idx, len(visual_traj)-1)]
            ax2.add_patch(Circle((current[1] + 0.5, maze.size[0] - 1 - current[0] + 0.5),
                               0.3, color='#00d9ff', edgecolor='white', linewidth=2))
            
            ax2.text(0.5, -0.5, f'Steps: {min(frame_idx, len(visual_traj)-1)}', 
                    transform=ax2.transAxes, fontsize=12, color='#00d9ff')
        
        # Start and goal for both
        for ax in [ax1, ax2]:
            sx, sy = maze.start_pos
            ax.add_patch(Circle((sy + 0.5, maze.size[0]-1-sx + 0.5), 0.35,
                              facecolor='#2ecc71', edgecolor='white', linewidth=2))
            ax.text(sy + 0.5, maze.size[0]-1-sx + 0.5, 'S', 
                   ha='center', va='center', fontsize=14, color='white', weight='bold')
            
            gx, gy = maze.goal_pos
            ax.add_patch(Circle((gy + 0.5, maze.size[0]-1-gx + 0.5), 0.35,
                              facecolor='#f39c12', edgecolor='white', linewidth=2))
            ax.text(gy + 0.5, maze.size[0]-1-gx + 0.5, 'G', 
                   ha='center', va='center', fontsize=14, color='black', weight='bold')
            
            ax.set_xlim(-0.5, maze.size[1] - 0.5)
            ax.set_ylim(-0.5, maze.size[0] - 0.5)
            ax.set_aspect('equal')
            ax.axis('off')
        
        # Main title
        fig.suptitle(f'Navigation Comparison - Frame {frame_idx}', 
                    fontsize=20, color='white', y=0.98)
        
        plt.tight_layout()
        
        # Save frame
        frame_path = f'/tmp/sidebyside_{i:04d}.png'
        fig.savefig(frame_path, dpi=100, bbox_inches='tight', facecolor='#0f0f1e')
        plt.close()
        
        frames.append(imageio.imread(frame_path))
        
        if i % 20 == 0:
            print(f"  Frame {i+1}/{len(frame_indices)}")
    
    # Save as GIF
    output_file = 'maze_navigation_sidebyside.gif'
    imageio.mimsave(output_file, frames, fps=10, loop=0)
    print(f"\n✅ Saved animation to {output_file}")
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)
    blind_steps = len(blind_traj) - 1
    visual_steps = len(visual_traj) - 1
    
    if blind_steps > 0 and visual_steps > 0:
        print(f"Blind:  {blind_steps} steps ({blind_nav.wall_hits} wall hits)")
        print(f"Visual: {visual_steps} steps")
        print(f"Visual is {blind_steps/visual_steps:.1f}x faster")
        print(f"Visual saves {blind_steps - visual_steps} steps ({(1 - visual_steps/blind_steps)*100:.0f}%)")
    print("=" * 60)


if __name__ == "__main__":
    create_sidebyside_gif()