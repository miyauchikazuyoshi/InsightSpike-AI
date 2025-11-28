import os, sys
import numpy as np
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
from core.episode_manager import EpisodeManager  # type: ignore
from core.graph_manager import GraphManager  # type: ignore
from core.vector_processor import VectorProcessor  # type: ignore
from navigation.decision_engine import DecisionEngine  # type: ignore
from navigation.branch_detector import BranchDetector as BacktrackDetector  # type: ignore

# Minimal harness: we only rely on MazeNavigator internal logic producing events on a tiny maze.

def build_small_maze():
    # 3x3 grid with one wall forcing a dead-end and potential backtrack
    maze = np.zeros((3,3), dtype=int)
    maze[1,1] = 1  # wall center
    return maze


def test_events_emitted_basic():
    maze = build_small_maze()
    nav = MazeNavigator(
        maze=maze,
        start_pos=(0,0),
        goal_pos=(2,2),
        enable_flush=False,
        simple_mode=True,
        backtrack_debounce=True,
        max_in_memory=500,
        flush_interval=50,
    )
    # Run limited steps
    for _ in range(60):
        done = nav.step()
        if done:
            break
    types = {e['type'] for e in nav.event_log}
    # Expect at least some structural / plan related events
    expected_any = {
        MazeNavigator.EventType.BACKTRACK_TRIGGER.value,
        MazeNavigator.EventType.BACKTRACK_PLAN.value,
        MazeNavigator.EventType.DEAD_END.value,
        MazeNavigator.EventType.SHORTCUT_CANDIDATE.value,
    }
    assert types & expected_any  # at least one of them
    # No unexpected empty log
    assert len(nav.event_log) > 0
