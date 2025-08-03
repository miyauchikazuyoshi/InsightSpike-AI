#!/usr/bin/env python3
"""Create animated GIF of navigation on complex mazes."""

import sys
from pathlib import Path
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
import imageio
from PIL import Image

sys.path.append(str(Path(__file__).parent.parent.parent))

from insightspike.environments.maze import SimpleMaze
from insightspike.navigators.blind_experience_navigator import BlindExperienceNavigator
from insightspike.navigators.experience_memory_navigator import ExperienceMemoryNavigator
from insightspike.config.maze_config import MazeNavigatorConfig

# Suppress verbose output
import logging
logging.getLogger().setLevel(logging.WARNING)


def create_navigation_gif(navigator_type='both', maze_size=(20, 20), maze_type='rooms'):
    """Create animated GIF of maze navigation."""
    # Load config
    config_path = Path(__file__).parent / "maze_experiment_config.yaml"
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    nav_config = MazeNavigatorConfig(**config_dict['navigator'])
    nav_config.exploration_epsilon = 0.0
    
    print(f"Creating Navigation GIF")
    print(f"Maze: {maze_size[0]}x{maze_size[1]} {maze_type}")
    print("=" * 60)
    
    # Create maze
    np.random.seed(42)
    maze = SimpleMaze(size=maze_size, maze_type=maze_type)
    
    print(f"Start: {maze.start_pos}, Goal: {maze.goal_pos}")
    
    # Run both navigators
    navigators = {}
    
    if navigator_type in ['blind', 'both']:
        # Run blind navigator
        blind_nav = BlindExperienceNavigator(nav_config)
        blind_trajectory = []
        blind_walls = []
        
        obs = maze.reset()
        blind_trajectory.append(obs.position)
        
        for step in range(1000):
            action = blind_nav.decide_action(obs, maze)
            old_pos = obs.position
            obs, reward, done, info = maze.step(action)
            blind_trajectory.append(obs.position)
            
            # Record wall hits
            if old_pos == obs.position:
                blind_walls.append((old_pos, action))
            
            if done and maze.agent_pos == maze.goal_pos:
                print(f"Blind navigator: {step + 1} steps, {blind_nav.wall_hits} wall hits")
                break
        else:
            print(f"Blind navigator: Failed")
        
        navigators['blind'] = {
            'trajectory': blind_trajectory,
            'wall_hits': blind_walls,
            'color': 'red',
            'name': 'Blind (Physical Only)'
        }
    
    if navigator_type in ['visual', 'both']:
        # Run visual navigator
        visual_nav = ExperienceMemoryNavigator(nav_config)
        visual_trajectory = []
        
        obs = maze.reset()
        visual_trajectory.append(obs.position)
        
        for step in range(1000):
            action = visual_nav.decide_action(obs, maze)
            obs, reward, done, info = maze.step(action)
            visual_trajectory.append(obs.position)
            
            if done and maze.agent_pos == maze.goal_pos:
                print(f"Visual navigator: {step + 1} steps")
                break
        else:
            print(f"Visual navigator: Failed")
        
        navigators['visual'] = {
            'trajectory': visual_trajectory,
            'wall_hits': [],
            'color': 'blue',
            'name': 'Visual+Physical'
        }
    
    # Create animation frames
    frames = []
    max_steps = max(len(nav['trajectory']) for nav in navigators.values())
    
    print(f"\nCreating {max_steps} frames...")
    
    for frame_idx in range(0, max_steps, 2):  # Skip every other frame for smaller file
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Draw maze
        for i in range(maze.size[0]):
            for j in range(maze.size[1]):
                if maze.grid[i, j] == 1:  # Wall
                    ax.add_patch(Rectangle((j, maze.size[0]-1-i), 1, 1, 
                                         facecolor='black'))
        
        # Draw trails and current positions
        for nav_name, nav_data in navigators.items():
            trajectory = nav_data['trajectory']
            color = nav_data['color']
            
            # Draw trail up to current frame
            if frame_idx < len(trajectory):
                trail_x = [pos[1] + 0.5 for pos in trajectory[:frame_idx+1]]
                trail_y = [maze.size[0] - 1 - pos[0] + 0.5 for pos in trajectory[:frame_idx+1]]
                
                # Draw trail with fading effect
                for i in range(len(trail_x) - 1):
                    alpha = 0.3 + 0.5 * (i / len(trail_x))  # Fade in
                    ax.plot(trail_x[i:i+2], trail_y[i:i+2], 
                           color=color, alpha=alpha, linewidth=2)
                
                # Draw current position
                current_pos = trajectory[min(frame_idx, len(trajectory)-1)]
                circle = Circle((current_pos[1] + 0.5, maze.size[0] - 1 - current_pos[0] + 0.5), 
                              0.3, color=color, alpha=0.8)
                ax.add_patch(circle)
                
                # Add label
                ax.text(current_pos[1] + 0.5, maze.size[0] - 1 - current_pos[0] + 0.3,
                       nav_name[0].upper(), ha='center', va='center', 
                       color='white', fontsize=12, weight='bold')
        
        # Draw start and goal
        sx, sy = maze.start_pos
        ax.add_patch(Rectangle((sy + 0.3, maze.size[0]-1-sx + 0.3), 0.4, 0.4,
                             facecolor='green', alpha=0.7))
        ax.text(sy + 0.5, maze.size[0]-1-sx + 0.5, 'S', 
               ha='center', va='center', fontsize=14, color='white', weight='bold')
        
        gx, gy = maze.goal_pos
        ax.add_patch(Rectangle((gy + 0.3, maze.size[0]-1-gx + 0.3), 0.4, 0.4,
                             facecolor='gold', alpha=0.7))
        ax.text(gy + 0.5, maze.size[0]-1-gx + 0.5, 'G', 
               ha='center', va='center', fontsize=14, color='black', weight='bold')
        
        # Legend
        legend_elements = []
        for nav_name, nav_data in navigators.items():
            steps = min(frame_idx, len(nav_data['trajectory'])-1)
            legend_elements.append(
                mpatches.Patch(color=nav_data['color'], 
                             label=f"{nav_data['name']}: Step {steps}")
            )
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        # Setup
        ax.set_xlim(0, maze.size[1])
        ax.set_ylim(0, maze.size[0])
        ax.set_aspect('equal')
        ax.set_title(f'Maze Navigation Comparison - Frame {frame_idx//2 + 1}', fontsize=16)
        ax.axis('off')
        
        # Save frame to file first
        frame_path = f'/tmp/maze_frame_{frame_idx:04d}.png'
        fig.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        # Read back and append
        frames.append(imageio.imread(frame_path))
        
        if frame_idx % 20 == 0:
            print(f"  Frame {frame_idx//2 + 1}/{max_steps//2}")
    
    # Save as GIF
    output_file = f'maze_navigation_{maze_type}_{maze_size[0]}x{maze_size[1]}.gif'
    imageio.mimsave(output_file, frames, fps=10)
    print(f"\n✅ Saved animation to {output_file}")
    
    # Also create a final comparison image
    create_final_comparison(maze, navigators, maze_type, maze_size)


def create_final_comparison(maze, navigators, maze_type, maze_size):
    """Create final comparison image."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Draw maze
    for i in range(maze.size[0]):
        for j in range(maze.size[1]):
            if maze.grid[i, j] == 1:  # Wall
                ax.add_patch(Rectangle((j, maze.size[0]-1-i), 1, 1, 
                                     facecolor='black'))
    
    # Draw complete trajectories
    for nav_name, nav_data in navigators.items():
        trajectory = nav_data['trajectory']
        color = nav_data['color']
        
        trail_x = [pos[1] + 0.5 for pos in trajectory]
        trail_y = [maze.size[0] - 1 - pos[0] + 0.5 for pos in trajectory]
        
        ax.plot(trail_x, trail_y, color=color, alpha=0.7, linewidth=3, 
               label=f"{nav_data['name']}: {len(trajectory)-1} steps")
    
    # Draw start and goal
    sx, sy = maze.start_pos
    ax.add_patch(Circle((sy + 0.5, maze.size[0]-1-sx + 0.5), 0.4,
                       facecolor='green', edgecolor='darkgreen', linewidth=2))
    ax.text(sy + 0.5, maze.size[0]-1-sx + 0.5, 'S', 
           ha='center', va='center', fontsize=16, color='white', weight='bold')
    
    gx, gy = maze.goal_pos
    ax.add_patch(Circle((gy + 0.5, maze.size[0]-1-gx + 0.5), 0.4,
                       facecolor='gold', edgecolor='orange', linewidth=2))
    ax.text(gy + 0.5, maze.size[0]-1-gx + 0.5, 'G', 
           ha='center', va='center', fontsize=16, color='black', weight='bold')
    
    # Setup
    ax.set_xlim(0, maze.size[1])
    ax.set_ylim(0, maze.size[0])
    ax.set_aspect('equal')
    ax.set_title(f'{maze_type.capitalize()} Maze ({maze_size[0]}x{maze_size[1]}) - Navigation Comparison', 
                fontsize=18)
    ax.legend(loc='upper right', fontsize=14)
    ax.axis('off')
    
    plt.tight_layout()
    output_file = f'maze_comparison_{maze_type}_{maze_size[0]}x{maze_size[1]}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved comparison to {output_file}")


def create_multiple_gifs():
    """Create GIFs for different maze types."""
    maze_configs = [
        {'size': (20, 20), 'type': 'rooms', 'nav': 'both'},
        {'size': (25, 25), 'type': 'spiral', 'nav': 'both'},
        {'size': (30, 30), 'type': 'dfs', 'nav': 'visual'},  # Larger maze, visual only
        {'size': (20, 20), 'type': 'prim', 'nav': 'blind'},   # Blind only
    ]
    
    for config in maze_configs:
        print(f"\n{'='*60}")
        print(f"Creating GIF for {config['type']} maze {config['size'][0]}x{config['size'][1]}")
        print(f"Navigator: {config['nav']}")
        print('='*60)
        
        create_navigation_gif(
            navigator_type=config['nav'],
            maze_size=config['size'],
            maze_type=config['type']
        )


if __name__ == "__main__":
    # Create single GIF with smaller maze first
    create_navigation_gif(navigator_type='both', maze_size=(15, 15), maze_type='dfs')
    
    # Create multiple GIFs
    create_multiple_gifs()