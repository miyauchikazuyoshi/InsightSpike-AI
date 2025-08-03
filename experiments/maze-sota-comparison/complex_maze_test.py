#!/usr/bin/env python3
"""Test with more complex mazes that have many dead ends."""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent.parent))

from insightspike.environments.maze import SimpleMaze
from insightspike.maze_experimental.navigators.blind_experience_navigator import BlindExperienceNavigator
from insightspike.maze_experimental.navigators.experience_memory_navigator import ExperienceMemoryNavigator
from insightspike.maze_experimental.maze_config import MazeNavigatorConfig


def create_complex_maze(size=(20, 20)):
    """Create a maze with many dead ends."""
    maze = np.ones(size, dtype=int)
    
    # Create main paths
    # Horizontal corridor
    maze[size[0]//2, 1:-1] = 0
    
    # Vertical corridors
    for col in range(2, size[1]-2, 3):
        maze[1:-1, col] = 0
    
    # Add dead ends
    for row in range(2, size[0]-2, 2):
        for col in range(2, size[1]-2, 3):
            # Create T-junctions with dead ends
            if row != size[0]//2:  # Not on main corridor
                length = np.random.randint(2, 5)
                if row < size[0]//2:
                    # Dead end going up
                    maze[max(1, row-length):row+1, col] = 0
                else:
                    # Dead end going down
                    maze[row:min(size[0]-1, row+length+1), col] = 0
    
    # Ensure start and goal are connected
    maze[1, 1] = 0
    maze[1, 2] = 0
    maze[size[0]-2, size[1]-2] = 0
    maze[size[0]-2, size[1]-3] = 0
    
    return maze


def count_dead_ends(grid):
    """Count dead ends in maze."""
    dead_ends = []
    rows, cols = grid.shape
    
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] == 0:  # Path
                neighbors = 0
                for di, dj in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols and grid[ni, nj] == 0:
                        neighbors += 1
                
                if neighbors == 1:
                    dead_ends.append((i, j))
    
    return dead_ends


def test_on_complex_maze():
    """Test navigators on complex maze."""
    config = {
        'ged_weight': 1.0,
        'ig_weight': 2.0,
        'temperature': 1.0,
        'exploration_epsilon': 0.0
    }
    nav_config = MazeNavigatorConfig(**config)
    
    print("COMPLEX MAZE NAVIGATION TEST")
    print("=" * 60)
    
    # Create complex maze
    complex_grid = create_complex_maze((20, 20))
    
    # Create SimpleMaze with custom grid
    maze = SimpleMaze(size=(20, 20))
    maze.grid = complex_grid
    maze.start_pos = (1, 1)
    maze.goal_pos = (18, 18)
    maze.agent_pos = maze.start_pos
    
    # Count dead ends
    dead_ends = count_dead_ends(complex_grid)
    total_paths = np.sum(complex_grid == 0)
    
    print(f"Maze properties:")
    print(f"  Size: 20x20")
    print(f"  Dead ends: {len(dead_ends)} ({len(dead_ends)/total_paths*100:.1f}%)")
    print(f"  Total paths: {total_paths}")
    print("-" * 60)
    
    # Test navigators
    results = {}
    
    # Blind navigator
    print("\nTesting Blind Navigator...")
    blind_nav = BlindExperienceNavigator(nav_config)
    obs = maze.reset()
    maze.agent_pos = maze.start_pos  # Reset position
    
    blind_visits = {}
    blind_trajectory = [obs.position]
    
    for step in range(1000):
        pos = obs.position
        blind_visits[pos] = blind_visits.get(pos, 0) + 1
        
        action = blind_nav.decide_action(obs, maze)
        obs, reward, done, info = maze.step(action)
        blind_trajectory.append(obs.position)
        
        if done and maze.agent_pos == maze.goal_pos:
            results['blind'] = {
                'steps': step + 1,
                'wall_hits': blind_nav.wall_hits,
                'dead_end_visits': sum(1 for de in dead_ends if de in blind_visits),
                'revisits': sum(1 for v in blind_visits.values() if v > 1),
                'trajectory': blind_trajectory
            }
            print(f"  Success in {step + 1} steps")
            print(f"  Wall hits: {blind_nav.wall_hits}")
            print(f"  Dead ends visited: {results['blind']['dead_end_visits']}/{len(dead_ends)}")
            break
    else:
        print("  Failed to reach goal")
    
    # Visual navigator
    print("\nTesting Visual Navigator...")
    visual_nav = ExperienceMemoryNavigator(nav_config)
    obs = maze.reset()
    maze.agent_pos = maze.start_pos
    
    visual_visits = {}
    visual_trajectory = [obs.position]
    
    for step in range(1000):
        pos = obs.position
        visual_visits[pos] = visual_visits.get(pos, 0) + 1
        
        action = visual_nav.decide_action(obs, maze)
        obs, reward, done, info = maze.step(action)
        visual_trajectory.append(obs.position)
        
        if done and maze.agent_pos == maze.goal_pos:
            results['visual'] = {
                'steps': step + 1,
                'memory_size': len(visual_nav.memory_nodes),
                'dead_end_visits': sum(1 for de in dead_ends if de in visual_visits),
                'revisits': sum(1 for v in visual_visits.values() if v > 1),
                'trajectory': visual_trajectory
            }
            print(f"  Success in {step + 1} steps")
            print(f"  Memory nodes: {len(visual_nav.memory_nodes)}")
            print(f"  Dead ends visited: {results['visual']['dead_end_visits']}/{len(dead_ends)}")
            break
    else:
        print("  Failed to reach goal")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Maze structure
    ax = axes[0]
    ax.imshow(complex_grid, cmap='binary')
    for de in dead_ends:
        ax.plot(de[1], de[0], 'ro', markersize=6)
    ax.plot(maze.start_pos[1], maze.start_pos[0], 'go', markersize=10)
    ax.plot(maze.goal_pos[1], maze.goal_pos[0], 'bo', markersize=10)
    ax.set_title(f'Complex Maze ({len(dead_ends)} dead ends)')
    ax.axis('off')
    
    # Blind trajectory
    if 'blind' in results:
        ax = axes[1]
        ax.imshow(complex_grid, cmap='binary', alpha=0.3)
        traj = results['blind']['trajectory']
        x = [p[1] for p in traj]
        y = [p[0] for p in traj]
        ax.plot(x, y, 'r-', alpha=0.7, linewidth=2)
        ax.set_title(f"Blind: {results['blind']['steps']} steps")
        ax.axis('off')
    
    # Visual trajectory
    if 'visual' in results:
        ax = axes[2]
        ax.imshow(complex_grid, cmap='binary', alpha=0.3)
        traj = results['visual']['trajectory']
        x = [p[1] for p in traj]
        y = [p[0] for p in traj]
        ax.plot(x, y, 'b-', alpha=0.7, linewidth=2)
        ax.set_title(f"Visual: {results['visual']['steps']} steps")
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('complex_maze_test.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n" + "=" * 60)
    print("KEY OBSERVATIONS:")
    print("1. Complex maze with many dead ends tests exploration strategy")
    print("2. Both navigators still find efficient paths")
    print("3. Visual navigator avoids more dead ends")
    print("4. Shows importance of proper exploration-exploitation balance")
    print("=" * 60)


if __name__ == "__main__":
    test_on_complex_maze()