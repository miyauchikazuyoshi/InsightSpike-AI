#!/usr/bin/env python3
"""
Test pure memory-based navigation
- No bonuses, no penalties
- Decision making purely from episodic memory
"""

import numpy as np
import matplotlib.pyplot as plt
from pure_episodic_movement_memory import PureEpisodicMovementMemory
import time


def generate_maze(size):
    """Generate a random maze with guaranteed path"""
    maze = np.ones((size, size), dtype=int)
    
    # Create a simple path (can be improved with proper maze generation)
    # Start with edges
    maze[0, :] = 0
    maze[-1, :] = 0
    maze[:, 0] = 0
    maze[:, -1] = 0
    
    # Add some random paths
    for _ in range(size * 2):
        x = np.random.randint(1, size-1)
        y = np.random.randint(1, size-1)
        maze[x, y] = 0
    
    # Ensure start and goal are clear
    maze[0, 0] = 0
    maze[size-1, size-1] = 0
    
    # Add some guaranteed connections
    for i in range(0, size, 2):
        if i < size:
            maze[i, :] = 0
    for j in range(1, size, 2):
        if j < size:
            maze[:, j] = 0
    
    # Add some walls back for complexity
    for _ in range(size):
        x = np.random.randint(2, size-2)
        y = np.random.randint(2, size-2)
        if (x, y) != (0, 0) and (x, y) != (size-1, size-1):
            maze[x, y] = 1
    
    return maze


def visualize_result(maze, path, visit_counts, title="Pure Memory Navigation"):
    """Visualize maze with path and visit counts"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Maze with path
    maze_vis = maze.copy().astype(float)
    maze_vis[maze == 1] = -1  # Walls
    
    for i, (x, y) in enumerate(path):
        maze_vis[x, y] = 2 + i / len(path)  # Gradient for path
    
    im1 = ax1.imshow(maze_vis, cmap='coolwarm', vmin=-1, vmax=3)
    ax1.set_title(f"{title} - Path")
    ax1.plot([p[1] for p in path], [p[0] for p in path], 'g-', alpha=0.5, linewidth=2)
    plt.colorbar(im1, ax=ax1)
    
    # Visit counts
    visit_map = np.zeros_like(maze, dtype=float)
    for (x, y), count in visit_counts.items():
        visit_map[x, y] = count
    
    im2 = ax2.imshow(visit_map, cmap='YlOrRd', vmin=0)
    ax2.set_title("Visit Counts")
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    return fig


def run_experiment(maze_sizes=[10, 15, 20], num_trials=3):
    """Run pure memory navigation experiment"""
    results = []
    
    for size in maze_sizes:
        print(f"\n{'='*50}")
        print(f"Testing {size}x{size} maze")
        print(f"{'='*50}")
        
        size_results = []
        
        for trial in range(num_trials):
            print(f"\nTrial {trial+1}/{num_trials}")
            
            # Generate maze
            maze = generate_maze(size)
            
            # Create navigator
            navigator = PureEpisodicMovementMemory(maze)
            
            # Navigate
            max_steps = size * size * 10
            result = navigator.navigate(max_steps=max_steps)
            
            # Store result
            result['maze_size'] = size
            result['trial'] = trial
            size_results.append(result)
            
            # Print summary
            if result['success']:
                print(f"✅ Success in {result['steps']} steps")
                print(f"   Episodes: {result['total_episodes']}")
                print(f"   Wall hits: {result['wall_hits']}")
                print(f"   Avg search time: {result['avg_search_time']:.2f}ms")
            else:
                print(f"❌ Failed after {max_steps} steps")
            
            # Visualize first trial
            if trial == 0:
                fig = visualize_result(
                    maze, 
                    result['path'], 
                    result['visit_counts'],
                    f"{size}x{size} Maze - Pure Memory"
                )
                plt.savefig(f"pure_memory_{size}x{size}.png", dpi=100, bbox_inches='tight')
                plt.close()
        
        results.append(size_results)
    
    # Summary statistics
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    
    for i, size in enumerate(maze_sizes):
        size_results = results[i]
        successes = [r for r in size_results if r['success']]
        
        if successes:
            avg_steps = np.mean([r['steps'] for r in successes])
            avg_episodes = np.mean([r['total_episodes'] for r in successes])
            avg_wall_hits = np.mean([r['wall_hits'] for r in successes])
            success_rate = len(successes) / len(size_results) * 100
            
            print(f"\n{size}x{size} Maze:")
            print(f"  Success rate: {success_rate:.0f}%")
            print(f"  Avg steps: {avg_steps:.0f}")
            print(f"  Avg episodes: {avg_episodes:.0f}")
            print(f"  Avg wall hits: {avg_wall_hits:.0f}")
        else:
            print(f"\n{size}x{size} Maze: No successes")
    
    return results


if __name__ == "__main__":
    print("Pure Memory-Based Navigation Experiment")
    print("=" * 50)
    print("No bonuses, no penalties - pure episodic memory")
    print("=" * 50)
    
    # Run experiment
    results = run_experiment(
        maze_sizes=[10, 15, 20],
        num_trials=3
    )
    
    print("\n✅ Experiment complete!")
    print("Results saved as pure_memory_*.png")