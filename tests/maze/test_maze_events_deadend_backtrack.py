import os, sys, numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
EXP_PATH = os.path.abspath(os.path.join(ROOT, 'experiments', 'maze-navigation-enhanced', 'src'))
if EXP_PATH not in sys.path:
    sys.path.insert(0, EXP_PATH)

if not os.path.exists(EXP_PATH):
    pytest.skip("maze-navigation-enhanced sources are not available", allow_module_level=True)

from navigation.maze_navigator import MazeNavigator  # type: ignore

def build_deadend_maze():
    # Simple corridor leading to dead-end forcing backtrack
    maze = np.zeros((4,4), dtype=int)
    # Surround so only path is along first row then down
    maze[1,1] = 1; maze[1,2] = 1; maze[2,1] = 1; maze[2,2] = 1
    return maze


def test_deadend_and_backtrack_events():
    nav = MazeNavigator(
        maze=build_deadend_maze(),
        start_pos=(0,0),
        goal_pos=(3,3),
        enable_flush=False,
        simple_mode=True,
        backtrack_debounce=True,
        max_in_memory=200,
        flush_interval=20,
    )
    for _ in range(120):
        done = nav.step()
        if done:
            break
    types = {e['type'] for e in nav.event_log}
    assert MazeNavigator.EventType.DEAD_END.value in types, "DEAD_END not emitted"
    assert MazeNavigator.EventType.BACKTRACK_TRIGGER.value in types or \
           MazeNavigator.EventType.BACKTRACK_PLAN.value in types, "Backtrack events missing"
