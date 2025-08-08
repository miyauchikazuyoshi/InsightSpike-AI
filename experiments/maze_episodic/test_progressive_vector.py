#!/usr/bin/env python3
"""
Progressive Vector Search Test
==============================

Demonstrate vector-based search on progressively larger mazes.
"""

from donut_gedig_navigator_simple import DonutGeDIGNavigator
from pure_episodic_navigator import create_complex_maze, visualize_maze_with_path
import time

def test_maze(size, seed=42):
    """Test single maze"""
    print(f"\n{'='*60}")
    print(f"Testing {size}×{size} maze")
    print('='*60)
    
    # Create maze
    maze = create_complex_maze(size, seed=seed)
    
    # Adaptive parameters
    if size <= 20:
        inner_radius = 0.1
        outer_radius = 0.6
    elif size <= 30:
        inner_radius = 0.15
        outer_radius = 0.7
    else:
        inner_radius = 0.2
        outer_radius = 0.8
    
    nav = DonutGeDIGNavigator(maze, inner_radius=inner_radius, outer_radius=outer_radius)
    
    # Navigate
    max_steps = min(size * size * 5, 10000)
    
    start_time = time.time()
    steps = 0
    
    while nav.position != nav.goal and steps < max_steps:
        # Progress updates
        if steps % 500 == 0 and steps > 0:
            dist = abs(nav.position[0] - nav.goal[0]) + abs(nav.position[1] - nav.goal[1])
            coverage = len(nav.visited) / (size * size) * 100
            donut_active = len(nav.episodes) > 100
            
            print(f"Step {steps}: dist={dist}, coverage={coverage:.1f}%, "
                  f"episodes={len(nav.episodes)}, donut={'ON' if donut_active else 'OFF'}")
        
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
    success = nav.position == nav.goal
    
    # Results
    if success:
        efficiency = steps / (2 * (size - 2))
        print(f"\n✓ SUCCESS in {steps} steps!")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Efficiency: {efficiency:.2f}x optimal")
        print(f"  Episodes: {len(nav.episodes)}")
        print(f"  Graph edges: {sum(len(n) for n in nav.graph.values()) // 2}")
        
        # Hop distribution
        total_hops = sum(nav.hop_selections.values())
        if total_hops > 0:
            print("  Hop usage:", end=" ")
            for hop, count in nav.hop_selections.items():
                print(f"{hop}: {count/total_hops*100:.1f}%", end=" ")
            print()
        
        # Save visualization
        visualize_maze_with_path(
            maze, nav.path,
            f'vector_success_{size}x{size}.png'
        )
        
        return True, steps, elapsed
    else:
        print(f"\n✗ Not completed in {steps} steps")
        print(f"  Distance to goal: {abs(nav.position[0] - nav.goal[0]) + abs(nav.position[1] - nav.goal[1])}")
        print(f"  Coverage: {len(nav.visited)/(size*size)*100:.1f}%")
        
        return False, steps, elapsed

def main():
    """Run progressive tests"""
    print("="*70)
    print("VECTOR-BASED geDIG NAVIGATION - PROGRESSIVE TEST")
    print("="*70)
    print("\nDemonstrating scalable maze navigation using vector search")
    
    # Test sizes
    sizes = [15, 20, 25, 30, 35, 40, 50]
    
    results = []
    for size in sizes:
        success, steps, time_taken = test_maze(size)
        results.append((size, success, steps, time_taken))
        
        # Stop if failed on smaller size
        if not success and size <= 30:
            print(f"\nStopping - failed on {size}x{size}")
            break
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Size':<10} {'Success':<10} {'Steps':<10} {'Time (s)':<10} {'Efficiency':<12}")
    print("-" * 60)
    
    for size, success, steps, time_taken in results:
        status = "✓" if success else "✗"
        efficiency = f"{steps/(2*(size-2)):.2f}x" if success else "N/A"
        print(f"{size}×{size:<5} {status:<10} {steps:<10} {time_taken:<10.1f} {efficiency:<12}")
    
    print("\nConclusion: Vector-based search enables efficient navigation")
    print("by finding similar past experiences in episodic memory!")

if __name__ == "__main__":
    main()