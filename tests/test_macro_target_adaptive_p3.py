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


def test_embedding_collapse_detection_event():
    # Small maze; we will artificially create embedding collapse by overwriting episode vectors
    maze = np.zeros((8,8), dtype=int)
    nav = MazeNavigator(maze=maze, start_pos=(0,0), goal_pos=(7,7), simple_mode=True)
    planner = getattr(nav, '_macro_target_planner', None)
    assert planner is not None, 'Planner missing'
    # Make thresholds very loose except collapse thresholds very high so that low variance triggers quickly
    # Force collapse detection by raising thresholds upward (cv threshold) and spread threshold
    planner._emb_norm_cv_threshold = 0.25  # higher than near-zero cv we will create
    planner._emb_dist_spread_threshold = 0.05
    planner._emb_min_batch = 4  # lower minimum for faster trigger in test
    # Run a few steps to accumulate some episodes first
    for _ in range(5):
        nav.gedig_history.append(0.0)
        nav.step()
    # Overwrite recent episode vectors to be almost identical (collapse scenario)
    # Access internal episodes list if available
    # Prefer episode_manager if available for reliable access
    episodes = []
    mgr = getattr(nav, 'episode_manager', None)
    if mgr is not None:
        eps_map = getattr(mgr, 'episodes', {})
        if isinstance(eps_map, dict):
            episodes = list(eps_map.values())
    if not episodes:
        raw = getattr(nav, 'episodes', None) or getattr(nav, '_episodes', None) or []
        try:
            episodes = list(raw)
        except Exception:
            episodes = []
    assert episodes, 'No episodes accessible for embedding collapse simulation'
    base_vec = np.ones(16, dtype=float) * 0.5
    for ep in episodes[-40:]:
        # Try multiple attribute names
        if hasattr(ep, 'vector') and isinstance(ep.vector, np.ndarray) and ep.vector.shape[0] >= 16:
            ep.vector[:16] = base_vec
        elif hasattr(ep, 'vec'):
            v = getattr(ep, 'vec')
            if isinstance(v, np.ndarray) and v.shape[0] >= 16:
                v[:16] = base_vec
    # Execute more steps to trigger adaptive check (embedding stats computed in _adaptive_update)
    for _ in range(6):
        nav.gedig_history.append(0.0)
        nav.step()
    stats = nav.get_statistics()
    adaptive = stats.get('macro_target_metrics', {}).get('adaptive_signals', {})
    events = adaptive.get('events_tail', []) or []
    collapse_flag = adaptive.get('embedding_collapse_active')
    health = adaptive.get('embedding_health') or {}
    # We expect either active flag or enter event plus health stats containing norm_cv
    assert 'norm_cv' in health, 'embedding health stats missing norm_cv'
    assert collapse_flag or any(e.get('event')=='enter_embedding_collapse' for e in events), 'Embedding collapse not detected under forced identical vectors'
