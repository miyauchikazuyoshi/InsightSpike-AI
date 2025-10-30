#!/usr/bin/env python3
"""Profile geDIG computation to find bottlenecks."""

import os
import sys
import time
import numpy as np
import cProfile
import pstats
from io import StringIO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from navigation.maze_navigator import MazeNavigator
from core.graph_manager import GraphManager
from core.gedig_evaluator import GeDIGEvaluator
from core.episode_manager import Episode
import networkx as nx


def profile_gedig_calculation():
    """Profile the geDIG calculation specifically."""
    
    print("=" * 60)
    print("Profiling geDIG Calculation")
    print("=" * 60)
    
    # Create evaluator and graphs
    evaluator = GeDIGEvaluator()
    
    # Test with different graph sizes
    sizes = [10, 20, 50, 100]
    
    for n in sizes:
        # Create random graphs
        g1 = nx.erdos_renyi_graph(n, 0.1)
        g2 = g1.copy()
        g2.add_edge(0, min(n-1, 5))  # Add one edge
        
        # Time the calculation
        start = time.perf_counter()
        for _ in range(10):  # Run 10 times for average
            result = evaluator.calculate(g1, g2)
        elapsed = (time.perf_counter() - start) / 10
        
        print(f"Graph size {n:3d}: {elapsed*1000:.2f} ms/calculation")
    
    print("\n" + "=" * 60)
    print("Profiling geDIG Wiring Strategy")
    print("=" * 60)
    
    # Profile the wiring strategy
    graph_mgr = GraphManager(evaluator)
    
    # Create episodes
    episodes = []
    for i in range(50):
        ep = Episode(
            episode_id=i,
            position=(i % 10, i // 10),
            direction='N',
            vector=np.random.randn(128),
            is_wall=False,
            timestamp=float(i)
        )
        graph_mgr.add_episode_node(ep)
        episodes.append(ep)
    
    print(f"Testing with {len(episodes)} episodes...")
    
    # Profile simple wiring
    graph_mgr.graph.clear()
    for ep in episodes:
        graph_mgr.graph.add_node(ep.episode_id)
    
    start = time.perf_counter()
    graph_mgr._wire_simple(episodes)
    simple_time = time.perf_counter() - start
    simple_edges = graph_mgr.graph.number_of_edges()
    
    # Profile gedig wiring
    graph_mgr.graph.clear()
    for ep in episodes:
        graph_mgr.graph.add_node(ep.episode_id)
    
    start = time.perf_counter()
    graph_mgr._wire_with_gedig(episodes, threshold=-0.05)
    gedig_time = time.perf_counter() - start
    gedig_edges = graph_mgr.graph.number_of_edges()
    
    print(f"Simple wiring: {simple_time*1000:.2f} ms, {simple_edges} edges")
    print(f"geDIG wiring:  {gedig_time*1000:.2f} ms, {gedig_edges} edges")
    print(f"Slowdown: {gedig_time/simple_time:.1f}x")
    
    # Detailed profiling with cProfile
    print("\n" + "=" * 60)
    print("Detailed Profile of geDIG Wiring")
    print("=" * 60)
    
    profiler = cProfile.Profile()
    
    # Reset graph
    graph_mgr.graph.clear()
    for ep in episodes:
        graph_mgr.graph.add_node(ep.episode_id)
    
    # Profile the wiring
    profiler.enable()
    graph_mgr._wire_with_gedig(episodes, threshold=-0.05)
    profiler.disable()
    
    # Print stats
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(15)  # Top 15 functions
    print(s.getvalue())


def analyze_navigation_bottleneck():
    """Analyze where navigation gets stuck."""
    
    print("\n" + "=" * 60)
    print("Analyzing Navigation Bottleneck")
    print("=" * 60)
    
    # Create a simple maze
    maze = np.ones((10, 10), dtype=int)
    for i in range(1, 9):
        maze[5, i] = 0
        maze[i, 5] = 0
    
    start = (1, 5)
    goal = (8, 5)
    maze[start[0], start[1]] = 0
    maze[goal[0], goal[1]] = 0
    
    # Test both strategies
    for strategy in ['simple', 'gedig']:
        print(f"\nTesting '{strategy}' strategy...")
        
        nav = MazeNavigator(
            maze=maze,
            start_pos=start,
            goal_pos=goal,
            wiring_strategy=strategy,
            gedig_threshold=-0.05,
            simple_mode=True
        )
        
        step_times = []
        
        for step in range(20):
            start_time = time.perf_counter()
            nav.step()
            step_time = time.perf_counter() - start_time
            step_times.append(step_time * 1000)  # Convert to ms
            
            if nav.current_pos == goal:
                break
        
        print(f"  Average step time: {np.mean(step_times):.2f} ms")
        print(f"  Max step time: {np.max(step_times):.2f} ms")
        print(f"  Steps taken: {len(step_times)}")
        
        # Check which steps were slow
        slow_steps = [(i, t) for i, t in enumerate(step_times) if t > np.mean(step_times) * 2]
        if slow_steps:
            print(f"  Slow steps: {slow_steps}")


if __name__ == '__main__':
    profile_gedig_calculation()
    analyze_navigation_bottleneck()