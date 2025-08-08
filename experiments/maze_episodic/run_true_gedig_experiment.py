#!/usr/bin/env python3
"""
True geDIG Progressive Test
===========================

Test true geDIG (GED - IG) on progressively larger mazes.
"""

from true_pure_gedig_navigator import TruePureGeDIGNavigator
from pure_episodic_navigator import create_complex_maze, visualize_maze_with_path
import time
import numpy as np

def test_maze(size, seed=42, max_steps=None):
    """Test single maze with true geDIG"""
    print(f"\n{'='*60}")
    print(f"Testing {size}×{size} maze with TRUE geDIG")
    print('='*60)
    
    # Create maze
    maze = create_complex_maze(size, seed=seed)
    
    # Create navigator
    nav = TruePureGeDIGNavigator(maze)
    
    # Set max steps
    if max_steps is None:
        max_steps = min(size * size * 5, 5000)
    
    # Navigate
    start_time = time.time()
    steps = 0
    
    # Track geDIG values for analysis
    gedig_history = []
    
    while nav.position != nav.goal and steps < max_steps:
        # Progress update
        if steps % 200 == 0 and steps > 0:
            dist = abs(nav.position[0] - nav.goal[0]) + abs(nav.position[1] - nav.goal[1])
            coverage = len(nav.visited) / (size * size) * 100
            avg_gedig = np.mean(gedig_history[-100:]) if gedig_history else 0
            
            print(f"Step {steps}: pos={nav.position}, dist={dist}, "
                  f"coverage={coverage:.1f}%, avg_geDIG={avg_gedig:.3f}")
        
        # Get action
        visual = nav.visual_memory.get(nav.position, {})
        action_gedigs = {}
        
        for action in ['up', 'right', 'down', 'left']:
            if visual.get(action) != 'wall':
                gedig = nav.evaluate_action_gedig(nav.position, action)
                action_gedigs[action] = gedig
        
        if not action_gedigs:
            break
        
        # Choose minimum geDIG with small noise
        best_action = min(action_gedigs.items(), 
                         key=lambda x: x[1] + np.random.normal(0, 0.01))[0]
        best_gedig = action_gedigs[best_action]
        gedig_history.append(best_gedig)
        
        # Execute action
        old_pos = nav.position
        dx, dy = {'up': (0, -1), 'right': (1, 0), 
                 'down': (0, 1), 'left': (-1, 0)}[best_action]
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
        
        nav.add_episode(old_pos, best_action, result, reached_goal)
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
        print(f"  Graph: {nav.episode_graph.number_of_nodes()} nodes, "
              f"{nav.episode_graph.number_of_edges()} edges")
        print(f"  geDIG calculations: {nav.gedig_calculations}")
        
        # Analyze geDIG values
        if gedig_history:
            negative_gedigs = sum(1 for g in gedig_history if g < 0)
            print(f"  Negative geDIG actions: {negative_gedigs} "
                  f"({negative_gedigs/len(gedig_history)*100:.1f}%)")
        
        # Save visualization
        visualize_maze_with_path(
            maze, nav.path,
            f'true_gedig_{size}x{size}.png'
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
    print("TRUE geDIG (GED - IG) NAVIGATION TEST")
    print("="*70)
    print("\nUsing the correct theoretical definition:")
    print("- geDIG = Graph Edit Distance - Information Gain")
    print("- Minimize geDIG to find informative actions")
    print("- No exploration bonus, no visit penalty")
    print("- Pure information-theoretic navigation")
    
    # Test sizes
    sizes = [5, 10, 15, 20, 25]
    
    results = []
    for size in sizes:
        # Adjust max steps for computational efficiency
        max_steps = min(size * size * 3, 2000)
        
        success, steps, time_taken = test_maze(size, max_steps=max_steps)
        results.append((size, success, steps, time_taken))
        
        # Continue even if failed to see the pattern
    
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
    
    print("\nConclusion:")
    print("True geDIG = GED - IG provides information-theoretic navigation")
    print("without any heuristics or exploration bonuses!")

if __name__ == "__main__":
    main()