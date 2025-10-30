"""Centralized results directory helpers for maze navigation experiments.

Legacy scripts wrote under top-level `results/maze_report`. We relocate to
`experiments/maze-navigation-enhanced/results/maze_report` so artifacts stay
scoped to this experiment suite.

Update experiment scripts to import `RESULTS_BASE` instead of hard-coding
paths; future structural moves then only change this module.
"""
from __future__ import annotations

import os

RESULTS_BASE = os.path.join('experiments', 'maze-navigation-enhanced', 'results', 'maze_report')

def run_dir(prefix: str, timestamp: str) -> str:
    return os.path.join(RESULTS_BASE, f"{prefix}_{timestamp}")

def ensure(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

__all__ = ["RESULTS_BASE", "run_dir", "ensure"]
