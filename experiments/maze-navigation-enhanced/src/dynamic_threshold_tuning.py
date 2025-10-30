"""Dynamic threshold tuning runner.

Runs the complex maze v2 with several dynamic_offset values and summarizes
escalation rate, branch completion windows collected, and basic drop/growth stats.
"""
from __future__ import annotations
import os, sys, statistics
from typing import List, Dict
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'navigation'))
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
for p in [PARENT_DIR, BASE_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from complex_maze_branch_probe_v2 import build_maze  # type: ignore
from navigation.maze_navigator import MazeNavigator  # type: ignore

offsets = [0.04, 0.05, 0.06, 0.07]
WARMUP = 12
WINDOW = 30
MAX_STEPS = 3000


def run_once(dynamic_offset: float) -> Dict[str, float | int | object]:
    maze, start, goal = build_maze()
    nav = MazeNavigator(
        maze=maze,
        start_pos=start,
        goal_pos=goal,
        wiring_strategy='simple',
        use_escalation=True,
        escalation_threshold='dynamic',
        dynamic_escalation=True,
        dynamic_offset=dynamic_offset,
        gedig_threshold=0.25,
        backtrack_threshold=-0.05,
        dynamic_warmup=WARMUP,
        dynamic_window=WINDOW,
    )
    nav.run(max_steps=MAX_STEPS)
    esc = [r for r in nav.gedig_structural if r.get('escalated')]
    drops = [r.get('drop_ratio') for r in nav.gedig_structural if r.get('drop_ratio') is not None]
    growth = [r.get('growth_ratio') for r in nav.gedig_structural]
    dead = sum(1 for r in nav.gedig_structural if r.get('dead_end'))
    shortcut = sum(1 for r in nav.gedig_structural if r.get('shortcut'))
    return {
        'offset': dynamic_offset,
        'steps': nav.step_count,
        'goal': nav.is_goal_reached,
        'escalation_rate': len(esc)/len(nav.gedig_structural) if nav.gedig_structural else 0.0,
        'branch_entries': sum(1 for e in nav.event_log if e['type']=='branch_entry'),
        'branch_completions': sum(1 for e in nav.event_log if e['type']=='branch_completion'),
        'dead_end_events': dead,
        'shortcut_events': shortcut,
        'median_drop_ratio': float(statistics.median(drops)) if drops else None,
        'median_growth_ratio': float(statistics.median(growth)) if growth else None,
    }


def main():
    rows: List[Dict[str, object]] = []
    for off in offsets:
        res = run_once(off)
        rows.append(res)
    print("\n=== Dynamic Threshold Tuning Summary ===")
    for r in rows:
        print(r)

if __name__ == '__main__':
    main()
