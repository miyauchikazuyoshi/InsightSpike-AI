#!/usr/bin/env python3
"""Test optimized geDIG implementation."""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.graph_manager import GraphManager
from core.graph_manager_optimized import OptimizedGraphManager
from core.episode_manager import Episode
from core.gedig_evaluator import GeDIGEvaluator


def compare_implementations():
    """Compare original vs optimized implementations."""
    
    print("=" * 60)
    print("Comparing Original vs Optimized geDIG")
    print("=" * 60)
    
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
        episodes.append(ep)
    
    print(f"Testing with {len(episodes)} episodes...")
    print()
    
    # Test original implementation
    print("Original GraphManager:")
    print("-" * 40)
    
    evaluator = GeDIGEvaluator()
    original_mgr = GraphManager(evaluator)
    
    # Add nodes
    for ep in episodes:
        original_mgr.add_episode_node(ep)
    
    # Time wiring
    start = time.perf_counter()
    original_mgr._wire_with_gedig(episodes, threshold=-0.05)
    original_time = time.perf_counter() - start
    original_edges = original_mgr.graph.number_of_edges()
    
    print(f"  Time: {original_time*1000:.2f} ms")
    print(f"  Edges created: {original_edges}")
    print()
    
    # Test optimized implementation
    print("Optimized GraphManager:")
    print("-" * 40)
    
    optimized_mgr = OptimizedGraphManager(evaluator)
    
    # Add nodes
    for ep in episodes:
        optimized_mgr.add_episode_node(ep)
    
    # Time wiring
    start = time.perf_counter()
    optimized_mgr._wire_with_gedig_optimized(episodes, threshold=-0.1, adaptive=True)
    optimized_time = time.perf_counter() - start
    optimized_edges = optimized_mgr.graph.number_of_edges()
    
    print(f"  Time: {optimized_time*1000:.2f} ms")
    print(f"  Edges created: {optimized_edges}")
    print(f"  Cache size: {len(optimized_mgr._gedig_cache)}")
    
    if optimized_mgr.edge_logs:
        gedig_values = [log['gedig'] for log in optimized_mgr.edge_logs]
        print(f"  geDIG values: min={min(gedig_values):.3f}, max={max(gedig_values):.3f}")
    
    print()
    print("=" * 60)
    print("Performance Comparison:")
    print(f"  Speedup: {original_time/optimized_time:.1f}x")
    print(f"  Edge difference: {optimized_edges - original_edges}")
    print("=" * 60)


def test_scaling():
    """Test how optimized version scales."""
    
    print("\n" + "=" * 60)
    print("Scaling Test")
    print("=" * 60)
    
    sizes = [10, 20, 50, 100, 200]
    
    print(f"{'Size':<8} {'Original (ms)':<15} {'Optimized (ms)':<15} {'Speedup':<10}")
    print("-" * 60)
    
    for n in sizes:
        # Create episodes
        episodes = []
        for i in range(n):
            ep = Episode(
                episode_id=i,
                position=(i % 10, i // 10),
                direction='N',
                vector=np.random.randn(128),
                is_wall=False,
                timestamp=float(i)
            )
            episodes.append(ep)
        
        # Original
        evaluator = GeDIGEvaluator()
        original_mgr = GraphManager(evaluator)
        for ep in episodes:
            original_mgr.add_episode_node(ep)
        
        start = time.perf_counter()
        original_mgr._wire_with_gedig(episodes, threshold=-0.05)
        original_time = (time.perf_counter() - start) * 1000
        
        # Optimized
        optimized_mgr = OptimizedGraphManager(evaluator)
        for ep in episodes:
            optimized_mgr.add_episode_node(ep)
        
        start = time.perf_counter()
        optimized_mgr._wire_with_gedig_optimized(episodes, threshold=-0.1, adaptive=True)
        optimized_time = (time.perf_counter() - start) * 1000
        
        speedup = original_time / optimized_time if optimized_time > 0 else float('inf')
        
        print(f"{n:<8} {original_time:<15.2f} {optimized_time:<15.2f} {speedup:<10.1f}x")


if __name__ == '__main__':
    compare_implementations()
    test_scaling()