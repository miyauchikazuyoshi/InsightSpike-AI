import numpy as np
import os, sys
THIS_DIR = os.path.dirname(__file__)
SRC_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..', 'src'))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)
from navigation.maze_navigator import MazeNavigator  # type: ignore

def make_maze(w=10,h=10):
    # simple empty maze
    return np.zeros((h,w), dtype=int)

def test_position_capacity_clamp():
    maze = make_maze(12,12)
    nav = MazeNavigator(
        maze,
        start_pos=(0,0),
        goal_pos=(11,11),
        wiring_strategy='query',
        enable_flush=True,
        flush_interval=10,
        max_in_memory=1000,
        max_in_memory_positions=30,
        wiring_top_k=2,
        snapshot_skip_idle=True,
        dense_metric_interval=5,
    )
    # Run enough steps to exceed raw unique positions
    nav.run(max_steps=500)
    stats = nav.get_statistics()
    # Collect passable positions actually retained
    passable_positions = stats.get('passable_positions')
    assert passable_positions is not None
    assert passable_positions <= 30, f"position capacity exceeded: {passable_positions} > 30"
    # Ensure at least one eviction event occurred
    evicted_total = stats.get('episodes_evicted_total',0)
    assert evicted_total > 0
