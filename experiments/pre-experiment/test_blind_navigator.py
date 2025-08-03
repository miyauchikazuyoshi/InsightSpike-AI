#!/usr/bin/env python3
"""Test blind navigator - no visual cheating!"""

import sys
from pathlib import Path
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.animation as animation

sys.path.append(str(Path(__file__).parent.parent.parent))

from insightspike.environments.maze import SimpleMaze
from insightspike.navigators.blind_experience_navigator import BlindExperienceNavigator, ExperienceType
from insightspike.navigators.experience_memory_navigator import ExperienceMemoryNavigator
from insightspike.config.maze_config import MazeNavigatorConfig


def run_blind_comparison():
    """Compare blind navigator (no cheating) vs visual navigator."""
    # Load config
    config_path = Path(__file__).parent / "maze_experiment_config.yaml"
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    nav_config = MazeNavigatorConfig(**config_dict['navigator'])
    
    # Test on multiple maze types
    maze_types = ['dfs', 'prim', 'kruskal']
    results = {}
    
    for maze_type in maze_types:
        print(f"\n{'='*60}")
        print(f"Testing on {maze_type.upper()} maze (15x15)")
        print(f"{'='*60}")
        
        results[maze_type] = {
            'blind': [],
            'visual': []
        }
        
        # Run multiple trials
        for trial in range(5):
            print(f"\n--- Trial {trial + 1} ---")
            
            # Create same maze for both navigators
            np.random.seed(42 + trial)
            maze = SimpleMaze(size=(15, 15), maze_type=maze_type)
            
            # Test blind navigator
            print("\n🦯 BLIND Navigator (no visual cheating):")
            blind_nav = BlindExperienceNavigator(nav_config)
            obs = maze.reset()
            
            for step in range(500):  # Allow more steps since it's harder
                action = blind_nav.decide_action(obs, maze)
                obs, reward, done, info = maze.step(action)
                
                if done and maze.agent_pos == maze.goal_pos:
                    print(f"\n✅ Goal reached in {step + 1} steps!")
                    print(f"   Wall hits: {blind_nav.wall_hits}")
                    results[maze_type]['blind'].append(step + 1)
                    break
            else:
                print(f"\n❌ Failed to reach goal in 500 steps")
                results[maze_type]['blind'].append(500)
            
            # Test visual navigator
            print("\n👁️  VISUAL Navigator (with visual info):")
            visual_nav = ExperienceMemoryNavigator(nav_config)
            obs = maze.reset()
            
            for step in range(500):
                action = visual_nav.decide_action(obs, maze)
                obs, reward, done, info = maze.step(action)
                
                if done and maze.agent_pos == maze.goal_pos:
                    print(f"\n✅ Goal reached in {step + 1} steps!")
                    results[maze_type]['visual'].append(step + 1)
                    break
            else:
                print(f"\n❌ Failed to reach goal in 500 steps")
                results[maze_type]['visual'].append(500)
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY RESULTS")
    print(f"{'='*60}")
    
    for maze_type in maze_types:
        blind_avg = np.mean(results[maze_type]['blind'])
        visual_avg = np.mean(results[maze_type]['visual'])
        overhead = (blind_avg / visual_avg - 1) * 100
        
        print(f"\n{maze_type.upper()} Maze:")
        print(f"  Blind (no cheat):  {blind_avg:.1f} steps")
        print(f"  Visual (w/ cheat): {visual_avg:.1f} steps")
        print(f"  Overhead: +{overhead:.1f}%")


def visualize_blind_exploration():
    """Visualize how blind navigator explores the maze."""
    # Load config
    config_path = Path(__file__).parent / "maze_experiment_config.yaml"
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    nav_config = MazeNavigatorConfig(**config_dict['navigator'])
    
    # Create maze
    np.random.seed(42)
    maze = SimpleMaze(size=(10, 10), maze_type='dfs')  # Smaller for clearer visualization
    
    # Create navigator
    navigator = BlindExperienceNavigator(nav_config)
    
    # Run navigation and collect frames
    obs = maze.reset()
    frames = []
    
    for step in range(200):
        # Create frame
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Draw maze
        for i in range(maze.size[0]):
            for j in range(maze.size[1]):
                if maze.grid[i, j] == 1:  # Wall
                    ax.add_patch(Rectangle((j, maze.size[0]-1-i), 1, 1, 
                                         facecolor='black'))
                elif (i, j) in navigator.memory_nodes:
                    node = navigator.memory_nodes[(i, j)]
                    # Color based on knowledge level
                    known_dirs = sum(1 for exp in node.experiences.values() 
                                   if exp.physical != ExperienceType.UNKNOWN)
                    alpha = known_dirs / 4.0
                    ax.add_patch(Rectangle((j, maze.size[0]-1-i), 1, 1, 
                                         facecolor='lightblue', alpha=alpha))
        
        # Mark physical experiences
        for pos, node in navigator.memory_nodes.items():
            y, x = pos
            for direction, exp in node.experiences.items():
                if exp.physical == ExperienceType.PHYSICAL_BLOCKED:
                    # Draw red line for known walls
                    if direction == 0:  # Up
                        ax.plot([x, x+1], [maze.size[0]-y, maze.size[0]-y], 'r-', linewidth=3)
                    elif direction == 1:  # Right
                        ax.plot([x+1, x+1], [maze.size[0]-1-y, maze.size[0]-y], 'r-', linewidth=3)
                    elif direction == 2:  # Down
                        ax.plot([x, x+1], [maze.size[0]-1-y, maze.size[0]-1-y], 'r-', linewidth=3)
                    elif direction == 3:  # Left
                        ax.plot([x, x], [maze.size[0]-1-y, maze.size[0]-y], 'r-', linewidth=3)
        
        # Draw start and goal
        sx, sy = maze.start_pos
        ax.text(sy + 0.5, maze.size[0]-1-sx + 0.5, 'S', 
               ha='center', va='center', fontsize=16, color='green', weight='bold')
        
        gx, gy = maze.goal_pos
        ax.text(gy + 0.5, maze.size[0]-1-gx + 0.5, 'G', 
               ha='center', va='center', fontsize=16, color='red', weight='bold')
        
        # Draw agent
        agent_y, agent_x = maze.agent_pos
        circle = plt.Circle((agent_x + 0.5, maze.size[0]-1-agent_y + 0.5), 0.3, 
                          color='blue', zorder=10)
        ax.add_patch(circle)
        
        # Setup
        ax.set_xlim(0, maze.size[1])
        ax.set_ylim(0, maze.size[0])
        ax.set_aspect('equal')
        ax.set_title(f'Blind Navigation - Step {step}\nWall hits: {navigator.wall_hits}')
        ax.axis('off')
        
        # Convert to image
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        frames.append(img)
        plt.close(fig)
        
        # Make decision
        action = navigator.decide_action(obs, maze)
        obs, reward, done, info = maze.step(action)
        
        if done and maze.agent_pos == maze.goal_pos:
            print(f"Goal reached in {step + 1} steps with {navigator.wall_hits} wall hits!")
            break
    
    # Save as GIF
    import imageio
    imageio.mimsave('blind_navigation.gif', frames[::2], fps=5)  # Skip every other frame for smaller file
    print("Saved animation to blind_navigation.gif")
    
    # Also save key metrics plot
    fig, ax = plt.subplots(figsize=(10, 6))
    steps = list(range(len(frames)))
    wall_hits = []
    positions_known = []
    
    # Replay to collect metrics
    np.random.seed(42)
    maze = SimpleMaze(size=(10, 10), maze_type='dfs')
    navigator = BlindExperienceNavigator(nav_config)
    obs = maze.reset()
    
    for step in range(len(frames)):
        wall_hits.append(navigator.wall_hits)
        positions_known.append(len(navigator.memory_nodes))
        
        action = navigator.decide_action(obs, maze)
        obs, reward, done, info = maze.step(action)
        
        if done:
            break
    
    ax.plot(steps[:len(wall_hits)], wall_hits, 'r-', label='Wall Hits', linewidth=2)
    ax.plot(steps[:len(positions_known)], positions_known, 'b-', label='Positions Known', linewidth=2)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Count')
    ax.set_title('Blind Navigation Learning Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('blind_navigation_metrics.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    # Run comparison
    run_blind_comparison()
    
    # Create visualization
    print("\nCreating visualization...")
    visualize_blind_exploration()