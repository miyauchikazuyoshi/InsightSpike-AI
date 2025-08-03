#!/usr/bin/env python3
"""Analyze maze complexity and dead-end frequency."""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

sys.path.append(str(Path(__file__).parent.parent.parent))
from insightspike.environments.maze import SimpleMaze


def analyze_maze(maze):
    """Analyze maze properties."""
    grid = maze.grid
    rows, cols = grid.shape
    
    # Count dead ends
    dead_ends = 0
    junctions = 0
    corridors = 0
    
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] == 0:  # Path
                # Count neighbors
                neighbors = 0
                for di, dj in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols and grid[ni, nj] == 0:
                        neighbors += 1
                
                if neighbors == 1:
                    dead_ends += 1
                elif neighbors == 2:
                    corridors += 1
                elif neighbors > 2:
                    junctions += 1
    
    # Calculate shortest path
    shortest_path = bfs_shortest_path(maze)
    
    # Calculate total path cells
    total_paths = np.sum(grid == 0)
    
    return {
        'dead_ends': dead_ends,
        'junctions': junctions,
        'corridors': corridors,
        'total_paths': total_paths,
        'shortest_path': shortest_path,
        'dead_end_ratio': dead_ends / total_paths if total_paths > 0 else 0,
        'junction_ratio': junctions / total_paths if total_paths > 0 else 0,
        'path_efficiency': shortest_path / total_paths if total_paths > 0 else 0
    }


def bfs_shortest_path(maze):
    """Find shortest path length."""
    start = maze.start_pos
    goal = maze.goal_pos
    grid = maze.grid
    
    queue = deque([(start, 0)])
    visited = set()
    
    while queue:
        (x, y), dist = queue.popleft()
        
        if (x, y) == goal:
            return dist
        
        if (x, y) in visited:
            continue
            
        visited.add((x, y))
        
        for dx, dy in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
            nx, ny = x + dx, y + dy
            if (0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1] and
                grid[nx, ny] == 0 and (nx, ny) not in visited):
                queue.append(((nx, ny), dist + 1))
    
    return float('inf')


def main():
    """Analyze different maze types."""
    maze_types = ['dfs', 'prim', 'kruskal', 'recursive_division']
    sizes = [(10, 10), (15, 15), (20, 20)]
    
    print("MAZE COMPLEXITY ANALYSIS")
    print("=" * 80)
    
    results = []
    
    for maze_type in maze_types:
        print(f"\n{maze_type.upper()} Maze Type:")
        print("-" * 40)
        
        for size in sizes:
            metrics_list = []
            
            # Analyze multiple instances
            for seed in range(10):
                try:
                    np.random.seed(seed)
                    maze = SimpleMaze(size=size, maze_type=maze_type)
                    metrics = analyze_maze(maze)
                    metrics_list.append(metrics)
                except:
                    continue
            
            if metrics_list:
                # Average metrics
                avg_metrics = {}
                for key in metrics_list[0]:
                    avg_metrics[key] = np.mean([m[key] for m in metrics_list])
                
                print(f"\n{size[0]}x{size[1]}:")
                print(f"  Dead ends: {avg_metrics['dead_ends']:.1f} ({avg_metrics['dead_end_ratio']*100:.1f}%)")
                print(f"  Junctions: {avg_metrics['junctions']:.1f} ({avg_metrics['junction_ratio']*100:.1f}%)")
                print(f"  Corridors: {avg_metrics['corridors']:.1f}")
                print(f"  Shortest path: {avg_metrics['shortest_path']:.1f}")
                print(f"  Path efficiency: {avg_metrics['path_efficiency']*100:.1f}%")
                
                results.append({
                    'type': maze_type,
                    'size': size,
                    'metrics': avg_metrics
                })
    
    # Visualize sample mazes
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for idx, maze_type in enumerate(maze_types):
        np.random.seed(42)
        
        # Small maze
        small_maze = SimpleMaze(size=(10, 10), maze_type=maze_type)
        ax = axes[0, idx]
        ax.imshow(small_maze.grid, cmap='binary')
        ax.set_title(f'{maze_type} (10x10)')
        ax.axis('off')
        
        # Mark dead ends in red
        for i in range(10):
            for j in range(10):
                if small_maze.grid[i, j] == 0:
                    neighbors = 0
                    for di, dj in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
                        ni, nj = i + di, j + dj
                        if (0 <= ni < 10 and 0 <= nj < 10 and 
                            small_maze.grid[ni, nj] == 0):
                            neighbors += 1
                    
                    if neighbors == 1:  # Dead end
                        ax.plot(j, i, 'ro', markersize=8)
                    elif neighbors > 2:  # Junction
                        ax.plot(j, i, 'go', markersize=8)
        
        # Large maze
        large_maze = SimpleMaze(size=(20, 20), maze_type=maze_type)
        ax = axes[1, idx]
        ax.imshow(large_maze.grid, cmap='binary')
        ax.set_title(f'{maze_type} (20x20)')
        ax.axis('off')
    
    plt.suptitle('Maze Types Comparison (Red=Dead ends, Green=Junctions)', fontsize=16)
    plt.tight_layout()
    plt.savefig('maze_complexity_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n" + "=" * 80)
    print("KEY FINDINGS:")
    print("1. DFS mazes have very few dead ends (mostly single path)")
    print("2. Prim's algorithm creates more branching")
    print("3. Current mazes are too simple for proper evaluation")
    print("4. Need mazes with more dead ends for realistic navigation challenges")
    print("=" * 80)


if __name__ == "__main__":
    main()