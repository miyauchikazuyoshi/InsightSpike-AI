"""Threshold tuning runner.

Runs complex maze v2 multiple times with different dynamic offset settings
(median - offset) to inspect escalation rate and branch completion signature separation.

Outputs concise table:
  offset, escalations, total_evals, rate, mean_drop_ratio (completion window), mean_growth_ratio(pre), dead_end_count
"""
from __future__ import annotations
import os, sys, statistics
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'navigation'))
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
for p in [PARENT_DIR, BASE_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from complex_maze_branch_probe_v2 import build_maze  # type: ignore
from navigation.maze_navigator import MazeNavigator  # type: ignore

OFFSETS = [0.05, 0.06, 0.07, 0.08]
RUNS_PER_OFFSET = 1  # can raise if needed
MAX_STEPS = 3000


def dynamic_threshold_from_history(hist: list[float], offset: float) -> float:
    if len(hist) < 10:
        return 0.1
    med = statistics.median(hist[-25:])
    return max(-0.01, med - offset)


def run_with_offset(offset: float):
    maze,start,goal = build_maze()
    nav = MazeNavigator(
        maze=maze,
        start_pos=start,
        goal_pos=goal,
        wiring_strategy='simple',
        use_escalation=True,
        escalation_threshold='dynamic',  # sentinel triggers dynamic inside navigator
        gedig_threshold=0.25,
        backtrack_threshold=-0.05,
        dynamic_escalation=True,
        dynamic_offset=offset,
    )
    # Monkey patch helper so we log chosen dynamic threshold each step (optional)
    nav.run(max_steps=MAX_STEPS)
    # Post metrics
    structural = nav.gedig_structural
    escalated = [r for r in structural if r.get('escalated')]
    completions = [e for e in nav.event_log if e['type']=='branch_completion']
    dead_ends = [e for e in nav.event_log if e['type']=='dead_end_detected']
    drops = [r.get('drop_ratio') for r in structural if r.get('drop_ratio') is not None]
    mean_drop = float(np.mean(drops)) if drops else 0.0
    return {
        'offset': offset,
        'escalations': len(escalated),
        'total': len(structural),
        'rate': (len(escalated)/len(structural)) if structural else 0,
        'branch_completions': len(completions),
        'dead_ends': len(dead_ends),
        'mean_drop_ratio': mean_drop,
    }


def main():
    results = []
    for off in OFFSETS:
        res = run_with_offset(off)
        results.append(res)
        print(f"offset={res['offset']:.3f} rate={res['rate']:.2%} esc={res['escalations']}/{res['total']} comps={res['branch_completions']} dead_ends={res['dead_ends']} mean_drop={res['mean_drop_ratio']:.3f}")
    print('\nSummary:')
    for r in results:
        print(r)

if __name__ == '__main__':
    main()
