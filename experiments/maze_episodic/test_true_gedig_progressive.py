#!/usr/bin/env python3
"""
Progressive test of True geDIG Navigator
"""

from true_gedig_navigator import TrueGeDIGNavigator
from pure_episodic_navigator import create_complex_maze, visualize_maze_with_path
import os
import time

def test_progressive():
    """Test progressively larger mazes"""
    print("="*70)
    print("TRUE geDIG PROGRESSIVE TEST")
    print("="*70)
    
    sizes = [15, 20, 25, 30, 35, 40, 50]
    results = {}
    
    for size in sizes:
        print(f"\n{'='*50}")
        print(f"Testing {size}×{size} maze")
        print('='*50)
        
        # Clean database
        db_path = f"maze_graph_{size}x{size}.db"
        if os.path.exists(db_path):
            os.remove(db_path)
        
        maze = create_complex_maze(size, seed=42)
        nav = TrueGeDIGNavigator(maze, db_path=db_path)
        
        # Adaptive max steps
        max_steps = min(size * size * 3, 5000)
        
        start_time = time.time()
        result = nav.navigate(max_steps=max_steps)
        
        results[size] = result
        
        if result['success']:
            efficiency = result['steps'] / (2 * (size - 2))
            print(f"\n✓ SUCCESS!")
            print(f"  Efficiency: {efficiency:.2f}x optimal")
            print(f"  Graph: {result['total_edges']} edges, {result['total_episodes']} episodes")
            print(f"  Density: {result['total_edges']/result['total_episodes']:.1f} edges/episode")
            
            # Save visualization
            os.makedirs('visualizations', exist_ok=True)
            visualize_maze_with_path(
                maze, nav.path,
                f'visualizations/true_gedig_{size}x{size}.png'
            )
        else:
            print(f"\n✗ Failed after {result['steps']} steps")
            print(f"  Explored {len(nav.visited)} cells ({len(nav.visited)/(size*size)*100:.1f}%)")
            # Stop testing larger sizes if failed
            if size >= 25:
                break
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\n{'Size':<10} {'Success':<10} {'Steps':<10} {'Efficiency':<12} {'Edges':<10} {'Time (s)':<10}")
    print("-" * 70)
    
    for size, result in results.items():
        success = "✓" if result['success'] else "✗"
        steps = result['steps']
        efficiency = f"{steps/(2*(size-2)):.2f}x" if result['success'] else "N/A"
        edges = result.get('total_edges', 0)
        time_taken = f"{result['time']:.1f}"
        
        print(f"{size}×{size:<5} {success:<10} {steps:<10} {efficiency:<12} {edges:<10} {time_taken:<10}")
    
    # Hop analysis
    print("\nHop Distribution Analysis:")
    for size, result in results.items():
        if result['success']:
            print(f"\n{size}×{size}:")
            total = sum(result['hop_selections'].values())
            for hop, count in result['hop_selections'].items():
                print(f"  {hop}: {count/total*100:.1f}%")


if __name__ == "__main__":
    test_progressive()