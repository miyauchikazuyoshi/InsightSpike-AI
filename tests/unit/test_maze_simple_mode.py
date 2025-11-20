import numpy as np
import os
import sys
from pathlib import Path
import pytest

maze_nav_src = Path("experiments/maze-navigation-enhanced/src")
if not maze_nav_src.is_dir():
    pytest.skip("maze-navigation-enhanced prototype not present; skipping simple-mode tests", allow_module_level=True)

sys.path.insert(0, os.path.abspath(str(maze_nav_src)))
from navigation.maze_navigator import MazeNavigator  # type: ignore


def small_maze():
    # Simple 5x5 maze with open paths (0) and walls (1); goal at bottom-right
    grid = np.zeros((5,5), dtype=int)
    # add a couple walls
    grid[1,2] = 1
    grid[3,1] = 1
    return grid


def test_query_single_generation():
    maze = small_maze()
    nav = MazeNavigator(maze, (0,0), (4,4), simple_mode=True)
    nav.run(max_steps=30)
    stats = nav.get_statistics()
    assert 'simple_mode' in stats
    sm = stats['simple_mode']
    # Expect close to 1.0
    assert sm['queries_per_step'] is not None
    assert abs(sm['queries_per_step'] - 1.0) < 0.05


def test_backtrack_simple_static_only():
    maze = small_maze()
    nav = MazeNavigator(maze, (0,0), (4,4), simple_mode=True, backtrack_debounce=True, backtrack_threshold=-0.5)
    nav.run(max_steps=40)
    sm = nav.get_statistics()['simple_mode']
    # threshold low => likely no triggers
    assert sm['backtrack_trigger_rate'] <= 0.2


def test_select_action_uses_external_query():
    maze = small_maze()
    nav = MazeNavigator(maze, (0,0), (4,4), simple_mode=True)
    nav.step()  # one step to generate query
    first_query_count = nav._query_generated_count
    nav.step()  # second step
    assert nav._query_generated_count == first_query_count + 1  # exactly one per step


def test_statistics_simple_mode_block():
    maze = small_maze()
    nav = MazeNavigator(maze, (0,0), (4,4), simple_mode=True)
    nav.run(max_steps=5)
    stats = nav.get_statistics()
    sm = stats['simple_mode']
    assert set(['query_generated','queries_per_step','backtrack_trigger_rate']).issubset(sm.keys())
