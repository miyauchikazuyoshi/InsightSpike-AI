#!/usr/bin/env python3
"""Test on progressively larger mazes"""

from pure_episodic_donut import PureEpisodicDonutNavigator
from pure_episodic_navigator import create_complex_maze, visualize_maze_with_path
import time

sizes = [5, 10, 15]
results = []

for size in sizes:
    print(f"\n{'='*50}")
    print(f"Testing {size}x{size} maze")
    print(f"{'='*50}")
    
    maze = create_complex_maze(size, seed=42)
    nav = PureEpisodicDonutNavigator(maze)
    
    max_steps = size * size * 3
    start_time = time.time()
    
    result = nav.navigate(max_steps=max_steps)
    result['size'] = size
    result['graph_nodes'] = nav.episode_graph.number_of_nodes()
    result['graph_edges'] = nav.episode_graph.number_of_edges()
    
    print(f"\n✓ Success: {result['success']}")
    print(f"- Steps: {result['steps']}")
    print(f"- Episodes: {result['episodes']}")
    print(f"- Time: {result['time']:.1f}s")
    print(f"- Graph: {result['graph_nodes']} nodes, {result['graph_edges']} edges")
    
    if result['success']:
        visualize_maze_with_path(maze, result['path'], f'pure_gedig_success_{size}x{size}.png')
        print(f"- Path efficiency: {(size-2)*2 / len(result['path']) * 100:.1f}%")
    else:
        print(f"- Final distance to goal: {abs(nav.position[0] - nav.goal_pos[0]) + abs(nav.position[1] - nav.goal_pos[1])}")
    
    results.append(result)
    
    # Stop if failed on smaller maze
    if not result['success'] and size < 15:
        print(f"\nStopping - failed on {size}x{size}")
        break

print(f"\n\n{'='*50}")
print("SUMMARY")
print(f"{'='*50}")
print(f"Size | Success | Steps | Episodes | Graph Edges | Time")
print(f"-"*50)
for r in results:
    print(f"{r['size']:4d} | {'Yes' if r['success'] else 'No':7s} | {r['steps']:5d} | {r['episodes']:8d} | "
          f"{r['graph_edges']:11d} | {r['time']:5.1f}s")