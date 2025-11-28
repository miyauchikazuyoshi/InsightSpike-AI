import numpy as np
import importlib.util, sys
from pathlib import Path
import pytest


def _load_maze_navigator():
    root = Path(__file__).resolve().parent.parent
    target = root / 'experiments' / 'maze-navigation-enhanced' / 'src' / 'navigation' / 'maze_navigator.py'
    if not target.exists():
        pytest.skip("maze-navigation-enhanced sources are not available", allow_module_level=True)
    spec = importlib.util.spec_from_file_location('maze_navigator_dynamic', target)
    module = importlib.util.module_from_spec(spec)  # type: ignore
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore
    return module.MazeNavigator


MazeNavigator = _load_maze_navigator()


def test_maze_navigator_init_basic():
    maze = np.zeros((5, 5), dtype=int)
    nav = MazeNavigator(maze, (0, 0), (4, 4), gedig_mode='core_raw')
    assert nav.current_pos == (0, 0)
    assert nav.gedig_mode == 'core_raw'


def test_maze_navigator_init_core_raw():
    maze = np.zeros((3, 3), dtype=int)
    nav = MazeNavigator(maze, (0, 0), (2, 2), gedig_mode='core_raw')
    assert nav.goal_pos == (2, 2)
    assert nav.gedig_mode == 'core_raw'
