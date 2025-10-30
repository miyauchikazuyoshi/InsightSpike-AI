"""ASCII maze loader.

Format:
  '#' = wall (1)
  '.' = path (0)
  'S' = start (0)
  'G' = goal (0)
Lines can have trailing spaces which are ignored. Empty lines skipped.

Returns: maze ndarray (int), start(tuple), goal(tuple)
"""
from __future__ import annotations
import numpy as np
from typing import Tuple

def load_ascii_maze(path: str) -> Tuple[np.ndarray, Tuple[int,int], Tuple[int,int]]:
    lines = []
    with open(path, 'r') as f:
        for raw in f.readlines():
            line = raw.rstrip('\n')
            if not line.strip():
                continue
            lines.append(line)
    if not lines:
        raise ValueError("Empty maze file")
    width = max(len(l) for l in lines)
    grid = []
    start = None
    goal = None
    for y, line in enumerate(lines):
        row = []
        for x, ch in enumerate(line):
            if ch == '#':
                row.append(1)
            elif ch in ('.',' '):
                row.append(0)
            elif ch == 'S':
                start = (x,y)
                row.append(0)
            elif ch == 'G':
                goal = (x,y)
                row.append(0)
            else:
                raise ValueError(f"Unsupported char '{ch}' at {(x,y)}")
        # pad if shorter
        if len(row) < width:
            row.extend([1]*(width-len(row)))  # pad as walls
        grid.append(row)
    maze = np.array(grid, dtype=int)
    if start is None or goal is None:
        raise ValueError("Maze must contain S and G")
    return maze, start, goal

__all__ = ["load_ascii_maze"]