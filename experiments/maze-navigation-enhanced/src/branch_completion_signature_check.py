"""Extract and print geDIG value signature around branch completion events.

Focus:
  - Show sustained growth values prior to completion
  - Show drop at completion (nodes_added -> 0, value collapse)
  - Include multihop variation if available
"""
from __future__ import annotations
import os, sys
from collections import defaultdict
from typing import Dict, List
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'navigation'))
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
for p in [PARENT_DIR, BASE_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from complex_maze_branch_probe_v2 import build_maze  # type: ignore
from navigation.maze_navigator import MazeNavigator  # type: ignore

def main():
    maze, start, goal = build_maze()
    nav = MazeNavigator(
        maze=maze,
        start_pos=start,
        goal_pos=goal,
        wiring_strategy='simple',
        use_escalation=True,
        escalation_threshold='dynamic',
        dynamic_escalation=True,
        dynamic_offset=0.06,
        gedig_threshold=0.25,
        backtrack_threshold=-0.05,
    )
    nav.run(max_steps=3500)

    # Index structural records by step
    rec_by_step: Dict[int, Dict] = {r['step']: r for r in nav.gedig_structural}
    completions = [e for e in nav.event_log if e['type']=='branch_completion']
    if not completions:
        print("No branch completions detected.")
        return
    print(f"Total branch completions: {len(completions)}")
    shown = 0
    for ev in completions[:8]:
        c_step = ev['step']
        window_steps = list(range(c_step-5, c_step+6))
        print(f"\n=== Completion @ step {c_step} ===")
        print("step | val     | nodes_added | edges_added | escal | drop_ratio | growth_ratio | mh_var | dead_end | shortcut")
        for s in window_steps:
            r = rec_by_step.get(s)
            if not r: continue
            print(f"{s:4d} | {r['value']:+.6f} | {r['nodes_added']:11d} | {r['edges_added']:11d} |"
                  f"  {int(bool(r.get('escalated')))}    | {r.get('drop_ratio') if r.get('drop_ratio') is not None else '-':>9} |"
                  f" {r.get('growth_ratio'):.6f}    | {r.get('multihop_variation') if r.get('multihop_variation') is not None else '-':>5} |"
                  f"    {int(bool(r.get('dead_end')))}     |    {int(bool(r.get('shortcut')))}")
    # Aggregate drop magnitudes
    drops = [r.get('drop_ratio') for r in nav.gedig_structural if r.get('drop_ratio')]
    if drops:
        print(f"\nMedian drop_ratio: {np.median(drops):.3f}  Mean drop_ratio: {np.mean(drops):.3f}")
    # Multihop coverage stats
    mh_valid = [r for r in nav.gedig_structural if r.get('multihop') and r.get('multihop_variation') is not None]
    if mh_valid:
        print(f"Multihop variation mean: {np.mean([r['multihop_variation'] for r in mh_valid]):.4f}")
        print(f"Multihop variation <0.05 count: {sum(1 for r in mh_valid if r['multihop_variation'] < 0.05)}")

if __name__ == '__main__':
    main()
