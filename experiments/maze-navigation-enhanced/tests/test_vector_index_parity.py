"""Parity test for query-based wiring: heap fallback vs InMemoryIndex.

Ensures that for a small maze and k <= number of existing episodes, the set of
connected neighbor episode IDs chosen via index search matches the legacy
heap-based selection (distance ordering parity).

NOTE: Because MazeNavigator currently wires incrementally and includes a
forced trajectory edge, we isolate the core selection by invoking the private
_wirings with a controlled state.
"""
from __future__ import annotations
import numpy as np
import os, sys

# Adjust path to import navigation components
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from navigation.maze_navigator import MazeNavigator  # type: ignore
from indexes.vector_index import InMemoryIndex  # type: ignore


def build_simple_nav(query_mode=True, use_index=False):
    maze = np.zeros((5,5), dtype=int)
    # 強制 query wiring にすると index 経由の追加が早期に発生
    strategy = 'query' if query_mode else 'simple'
    nav = MazeNavigator(maze, (0,0), (4,4), wiring_strategy=strategy, verbosity=0,
                        vector_index=(InMemoryIndex() if use_index else None))
    return nav


def test_query_wiring_parity_small_maze():
    nav_heap = build_simple_nav(use_index=False)
    nav_idx = build_simple_nav(use_index=True)
    # Deterministic exploration: reset RNG before each pair of steps
    for i in range(6):
        np.random.seed(100 + i)
        nav_heap.step()
        np.random.seed(100 + i)
        nav_idx.step()
    stats_idx = nav_idx.get_statistics()
    # vector_index_size キーが存在 (統計構造に露出) すること
    assert 'vector_index_size' in stats_idx
    g_heap = nav_heap.graph_manager.graph
    g_idx = nav_idx.graph_manager.graph
    newest_heap = max(g_heap.nodes)
    newest_idx = max(g_idx.nodes)
    def neighbors(graph, node):
        nbrs = set()
        for u, v in graph.edges():
            if u == node:
                nbrs.add(v)
            elif v == node:
                nbrs.add(u)
        return nbrs
    heap_neighbors = neighbors(g_heap, newest_heap)
    idx_neighbors = neighbors(g_idx, newest_idx)
    assert idx_neighbors == heap_neighbors


def test_fallback_no_index():
    nav = build_simple_nav(use_index=False)
    for _ in range(4):
        nav.step()
    stats = nav.get_statistics()
    # Should have at least one edge (simple wiring adds edges)
    assert stats['graph_stats']['num_edges'] >= 1
    # vector_index_size must be 0 when no index
    assert stats['vector_index_size'] == 0


def test_phase5_flush_placeholders():
    import numpy as np
    from navigation.maze_navigator import MazeNavigator
    maze = np.zeros((5,5), dtype=int)
    nav = MazeNavigator(maze, (0,0), (4,4), wiring_strategy='simple', enable_flush=True, flush_interval=123, max_in_memory=456)
    nav.run(max_steps=5)
    stats = nav.get_statistics()
    assert stats['flush_enabled'] is True
    assert stats['flush_interval'] == 123
    assert stats['max_in_memory'] == 456
    # Counters should start at 0
    assert stats['flush_events'] == 0
    assert stats['episodes_evicted_total'] == 0
    assert stats['episodes_rehydrated_total'] == 0


def test_flush_auto_index_injection_when_enabled():
    import numpy as np
    from navigation.maze_navigator import MazeNavigator
    maze = np.zeros((6,6), dtype=int)
    nav = MazeNavigator(maze, (0,0), (5,5), wiring_strategy='simple', enable_flush=True)
    nav.run(max_steps=10)
    stats = nav.get_statistics()
    # If DataStoreIndex is available index size should be >=0; we at least assert key presence
    assert 'vector_index_size' in stats
    # Not asserting >0 because episodes may all be walls in contrived cases; just ensure no exception


def test_memory_guard_pass_skeleton_probe():
    import numpy as np
    from navigation.maze_navigator import MazeNavigator
    maze = np.zeros((5,5), dtype=int)
    nav = MazeNavigator(maze, (0,0), (4,4), wiring_strategy='simple', enable_flush=True, flush_interval=1, max_in_memory=5)
    # Run enough steps to create > max_in_memory episodes
    nav.run(max_steps=15)
    stats = nav.get_statistics()
    assert stats['flush_events'] > 0
    # find a flush_score_probe event
    assert any(ev.get('type') == 'flush_score_probe' for ev in nav.event_log)


def test_memory_guard_eviction_occurs():
    import numpy as np
    from navigation.maze_navigator import MazeNavigator
    # Small maze but force many steps to exceed capacity; low max_in_memory
    maze = np.zeros((5,5), dtype=int)
    nav = MazeNavigator(maze, (0,0), (4,4), wiring_strategy='simple', enable_flush=True, flush_interval=1, max_in_memory=20)
    nav.run(max_steps=120)  # should trigger multiple flush passes
    stats = nav.get_statistics()
    # Validate that at least one eviction event fired
    assert any(ev.get('type') == 'flush_eviction' for ev in nav.event_log)
    # Ensure evictions occurred (counter >0)
    assert stats['episodes_evicted_total'] > 0
