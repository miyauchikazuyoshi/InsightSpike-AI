#!/usr/bin/env python
"""Test multi-hop exploration configurations"""

import numpy as np
from navigation.maze_navigator import MazeNavigator
from experiments.maze_layouts import create_complex_maze, COMPLEX_DEFAULT_START, COMPLEX_DEFAULT_GOAL
from experiments.generate_complex_maze_report import loop_erased_redundancy

def test_configuration(seed, config_name, force_multihop=False, max_hops=2):
    """Test a specific configuration"""
    np.random.seed(seed)
    maze = create_complex_maze()
    start, goal = COMPLEX_DEFAULT_START, COMPLEX_DEFAULT_GOAL
    
    # Standard weights
    weights = np.array([1.0, 1.0, 0.0, 0.0, 3.0, 2.0, 0.1, 0.0])
    
    nav = MazeNavigator(
        maze, start, goal, 
        weights=weights, 
        temperature=0.1,
        gedig_threshold=0.3, 
        backtrack_threshold=-0.2,
        wiring_strategy='simple', 
        simple_mode=True, 
        backtrack_debounce=True,
        backtrack_target_strategy='semantic',
        force_multihop=force_multihop,
        max_hops=max_hops,
        verbosity=0  # Silent
    )
    
    # Run navigation
    nav.navigate(max_steps=200)
    
    # Calculate metrics
    path = nav.path
    loop_red = loop_erased_redundancy(path)
    unique_coverage = len(set(path)) / np.sum(maze == 0)
    
    return {
        'config': config_name,
        'seed': seed,
        'path_len': len(path),
        'unique': len(set(path)),
        'loop_red': loop_red,
        'coverage': unique_coverage,
        'goal_reached': path[-1] == goal if path else False
    }

# Test configurations
configs = [
    ('baseline', False, 0),
    ('2-hop', True, 2),
    ('3-hop', True, 3),
    ('5-hop', True, 5),
]

print("Testing multi-hop configurations...")
print("-" * 60)

for config_name, force_multihop, max_hops in configs:
    results = []
    for seed in [1501, 1502, 1503]:
        result = test_configuration(seed, config_name, force_multihop, max_hops)
        results.append(result)
    
    # Average metrics
    avg_loop_red = np.mean([r['loop_red'] for r in results])
    avg_coverage = np.mean([r['coverage'] for r in results])
    
    print(f"\n{config_name:10} (multihop={force_multihop}, hops={max_hops})")
    print(f"  Loop redundancy: {avg_loop_red:.3f}")
    print(f"  Coverage: {avg_coverage:.3%}")
    print(f"  Goal reached: {sum(r['goal_reached'] for r in results)}/3")