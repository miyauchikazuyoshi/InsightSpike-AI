import os, sys, numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
NAV_SRC = os.path.join(ROOT, 'experiments', 'maze-navigation-enhanced', 'src')
if NAV_SRC not in sys.path:
    sys.path.insert(0, NAV_SRC)

if not os.path.exists(NAV_SRC):
    pytest.skip("maze-navigation-enhanced sources are not available", allow_module_level=True)

from navigation.maze_navigator import MazeNavigator  # type: ignore


def _maze(n=10):
    return np.zeros((n,n), dtype=int)


def test_adaptive_signals_presence_and_event():
    maze = _maze(12)
    nav = MazeNavigator(maze=maze, start_pos=(0,0), goal_pos=(11,11), simple_mode=True)
    planner = getattr(nav, '_macro_target_planner', None)
    assert planner is not None, 'Planner should be initialized.'
    # Aggressive thresholds to force stagnation quickly
    planner._stagnation_window = 4
    planner._stagnation_min_variation = 999.0
    planner._stagnation_new_neighbor_thresh = 999.0
    planner._recovery_new_neighbor_thresh = 1e9
    # Run several steps injecting synthetic flat geDIG values
    for _ in range(8):
        nav.gedig_history.append(0.0 if not nav.gedig_history else nav.gedig_history[-1])
        nav.step()
    stats = nav.get_statistics()
    mt = stats.get('macro_target_metrics', {})
    adaptive = mt.get('adaptive_signals', {})
    assert 'gedig_ema' in adaptive, 'gedig_ema missing in adaptive_signals'
    # Check if an enter_stagnation event or stagnation flag present
    events = adaptive.get('events_tail', []) or []
    stagnation_flag = adaptive.get('stagnation_active')
    assert stagnation_flag or any(e.get('event')=='enter_stagnation' for e in events), 'No stagnation detected under forced conditions'
