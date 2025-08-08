#!/usr/bin/env python3
"""Test 15x15 maze with vector search"""

from donut_gedig_navigator_simple import DonutGeDIGNavigator
from pure_episodic_navigator import create_complex_maze, visualize_maze_with_path
import time

size = 15
maze = create_complex_maze(size, seed=42)

nav = DonutGeDIGNavigator(maze, inner_radius=0.1, outer_radius=0.6)

print(f"Testing {size}×{size} maze")
print(f"Start: {nav.position}, Goal: {nav.goal}")

steps = 0
max_steps = 2000
start_time = time.time()

while nav.position != nav.goal and steps < max_steps:
    if steps % 100 == 0:
        dist = abs(nav.position[0] - nav.goal[0]) + abs(nav.position[1] - nav.goal[1])
        coverage = len(nav.visited) / (size * size) * 100
        donut_active = len(nav.episodes) > 100
        
        print(f"Step {steps}: pos={nav.position}, dist={dist}, "
              f"coverage={coverage:.1f}%, episodes={len(nav.episodes)}, "
              f"donut={'ON' if donut_active else 'OFF'}")
    
    # Navigate
    action = nav.decide_action()
    
    old_pos = nav.position
    dx, dy = {'up': (0, -1), 'right': (1, 0), 
             'down': (0, 1), 'left': (-1, 0)}[action]
    new_pos = (nav.position[0] + dx, nav.position[1] + dy)
    
    result = 'wall'
    reached_goal = False
    
    if (0 <= new_pos[0] < nav.width and 
        0 <= new_pos[1] < nav.height and
        nav.maze[new_pos[1], new_pos[0]] == 0):
        
        if new_pos in nav.visited:
            result = 'visited'
        else:
            result = 'success'
        
        nav.position = new_pos
        nav.visited.add(new_pos)
        nav.path.append(new_pos)
        nav._update_visual_memory(new_pos[0], new_pos[1])
        
        if new_pos == nav.goal:
            reached_goal = True
    else:
        nav.wall_hits += 1
    
    nav.add_episode(old_pos, action, result, reached_goal)
    steps += 1
    
    if reached_goal:
        break

elapsed = time.time() - start_time

if nav.position == nav.goal:
    efficiency = steps / (2 * (size - 2))
    print(f"\n✓ SUCCESS in {steps} steps!")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Efficiency: {efficiency:.2f}x optimal")
    print(f"  Wall hits: {nav.wall_hits}")
    print(f"  Episodes: {len(nav.episodes)}")
    print(f"  Graph edges: {sum(len(n) for n in nav.graph.values()) // 2}")
    
    # Hop distribution
    total_hops = sum(nav.hop_selections.values())
    if total_hops > 0:
        print("  Hop usage:", end=" ")
        for hop, count in nav.hop_selections.items():
            print(f"{hop}: {count/total_hops*100:.1f}%", end=" ")
        print()
    
    visualize_maze_with_path(maze, nav.path, 'vector_15x15_success.png')
    print("\nSaved visualization to vector_15x15_success.png")
else:
    print(f"\n✗ Not completed in {steps} steps")
    print(f"  Distance to goal: {abs(nav.position[0] - nav.goal[0]) + abs(nav.position[1] - nav.goal[1])}")
    print(f"  Coverage: {len(nav.visited)/(size*size)*100:.1f}%")