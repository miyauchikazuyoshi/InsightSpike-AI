#!/usr/bin/env python3
"""Debug geDIG values to understand thresholds."""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.graph_manager_optimized import OptimizedGraphManager
from core.episode_manager import Episode
from core.gedig_evaluator import GeDIGEvaluator


def test_gedig_values():
    """Test what geDIG values we get in practice."""
    
    print("Testing geDIG value ranges...")
    print("=" * 60)
    
    # Create evaluator and manager
    evaluator = GeDIGEvaluator()
    mgr = OptimizedGraphManager(evaluator)
    
    # Create test episodes
    episodes = []
    for i in range(10):
        ep = Episode(
            episode_id=i,
            position=(i % 5, i // 5),
            direction='N',
            vector=np.random.randn(128),
            is_wall=False,
            timestamp=float(i)
        )
        episodes.append(ep)
        mgr.add_episode_node(ep)
    
    # Test different thresholds
    thresholds = [0.5, 0.3, 0.1, 0.0, -0.05, -0.1, -0.2, -0.3]
    
    for threshold in thresholds:
        # Reset graph edges
        mgr.graph.clear_edges()
        mgr.edge_logs = []
        
        # Wire with threshold
        mgr._wire_with_gedig_optimized(episodes, threshold=threshold)
        
        edges = mgr.graph.number_of_edges()
        
        if mgr.edge_logs:
            gedig_values = [log['gedig'] for log in mgr.edge_logs]
            min_val = min(gedig_values)
            max_val = max(gedig_values)
            avg_val = sum(gedig_values) / len(gedig_values)
            print(f"Threshold {threshold:>6.2f}: {edges:>3} edges, "
                  f"geDIG range [{min_val:>6.3f}, {max_val:>6.3f}], avg={avg_val:>6.3f}")
        else:
            print(f"Threshold {threshold:>6.2f}: {edges:>3} edges (no edges created)")
    
    print("\n" + "=" * 60)
    print("Recommendation: Use threshold around -0.1 to -0.2")
    print("=" * 60)


if __name__ == '__main__':
    test_gedig_values()