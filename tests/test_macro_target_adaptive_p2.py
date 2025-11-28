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


def test_spike_detection_event():
    maze = np.zeros((10,10), dtype=int)
    nav = MazeNavigator(maze=maze, start_pos=(0,0), goal_pos=(9,9), simple_mode=True)
    planner = getattr(nav, '_macro_target_planner', None)
    assert planner is not None, 'Planner missing'
    # Lower threshold to make spike easier & zero cooldown
    planner._spike_norm_delta_threshold = 0.01
    planner._spike_cooldown_steps = 0
    # Feed increasing geDIG values before steps to trigger large normalized delta
    for i in range(6):
        nav.gedig_history.append(float(i))
        nav.step()
    stats = nav.get_statistics()
    adaptive = stats.get('macro_target_metrics', {}).get('adaptive_signals', {})
    events = adaptive.get('events_tail', []) or []
    spike_flag = adaptive.get('spike_active')
    assert spike_flag or any(e.get('event')=='enter_spike' for e in events), 'Spike not detected under forced increasing geDIG'
