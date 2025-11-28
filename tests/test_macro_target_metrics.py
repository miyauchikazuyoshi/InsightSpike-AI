import os, sys, math
import numpy as np
import pytest

# Ensure project and navigation source paths are importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
NAV_SRC = os.path.join(ROOT, 'experiments', 'maze-navigation-enhanced', 'src')
if NAV_SRC not in sys.path:
    sys.path.insert(0, NAV_SRC)

if not os.path.exists(NAV_SRC):
    pytest.skip("maze-navigation-enhanced sources are not available", allow_module_level=True)

from navigation.maze_navigator import MazeNavigator  # type: ignore


def _make_open_maze(size: int = 12):
    return np.zeros((size, size), dtype=int)


def _run_steps(nav: MazeNavigator, steps: int = 40):
    # Run a bounded number of steps; stop early if goal reached
    for _ in range(steps):
        if nav.is_goal_reached:
            break
        nav.step()


def test_macro_target_metrics_presence():
    maze = _make_open_maze()
    nav = MazeNavigator(maze=maze, start_pos=(0, 0), goal_pos=(11, 11), simple_mode=True)
    _run_steps(nav, 30)
    stats = nav.get_statistics()
    assert 'macro_target_metrics' in stats, 'macro_target_metrics missing in statistics'
    mt = stats['macro_target_metrics']
    assert isinstance(mt, dict)
    # Core expected fields (may be None but should exist after some steps)
    for k in ['last_candidate_count', 'last_sampled', 'last_chosen_target']:
        assert k in mt, f'{k} not present in macro_target_metrics'
    assert mt.get('last_candidate_count', 0) >= 0


def test_macro_target_analysis_basic():
    maze = _make_open_maze()
    nav = MazeNavigator(maze=maze, start_pos=(0, 0), goal_pos=(11, 11), simple_mode=True)
    _run_steps(nav, 45)
    stats = nav.get_statistics()
    analysis = stats.get('macro_target_analysis')
    assert isinstance(analysis, dict), 'macro_target_analysis not present or not a dict'
    # Not all keys guaranteed, but truncation rate should appear after at least one planner eval
    assert 'bfs_truncation_rate' in analysis, 'bfs_truncation_rate missing in analysis'
    tr = analysis['bfs_truncation_rate']
    assert tr is None or (0.0 <= tr <= 1.0)
    # If score_percentiles exist, validate structure
    if 'score_percentiles' in analysis:
        sp = analysis['score_percentiles']
        for p in ['p25','p50','p75']:
            assert p in sp

def test_macro_target_analysis_disabled():
    maze = _make_open_maze()
    nav = MazeNavigator(maze=maze, start_pos=(0,0), goal_pos=(11,11), simple_mode=True, macro_target_analysis=False)
    _run_steps(nav, 30)
    stats = nav.get_statistics()
    assert 'macro_target_metrics' in stats
    assert 'macro_target_analysis' not in stats, 'analysis should be disabled but key present'


def test_macro_target_snapshot_context_alignment():
    maze = _make_open_maze()
    nav = MazeNavigator(maze=maze, start_pos=(0, 0), goal_pos=(11, 11), simple_mode=True)
    _run_steps(nav, 10)
    planner = getattr(nav, '_macro_target_planner', None)
    assert planner is not None, 'planner not initialized'
    snap = planner.get_snapshot()
    ctx = snap.get('context')
    assert isinstance(ctx, dict)
    assert ctx.get('current_pos') == nav.current_pos
    assert ctx.get('goal_pos') == nav.goal_pos
    assert isinstance(ctx.get('step'), int)
