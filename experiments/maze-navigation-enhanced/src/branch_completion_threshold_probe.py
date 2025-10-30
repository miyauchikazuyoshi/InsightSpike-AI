"""Branch completion geDIG threshold probing script.

Purpose:
  Observe geDIG (hop0) and multi-hop behavior specifically at the moment a side branch exploration completes
  (all neighboring cells of that branch become walls or already explored), and contrast with normal exploration steps.

Outputs:
  - Lists branch exploration phases vs completion step scores
  - Multi-hop escalation snapshot (forced) around completion
  - Suggested threshold candidates derived from score distributions
"""
from __future__ import annotations

import os, sys
from statistics import mean, stdev
from collections import defaultdict
import numpy as np

# Path adjustments (reuse pattern from smoke test)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'navigation'))
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
for p in [PARENT_DIR, BASE_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from navigation.maze_navigator import MazeNavigator  # type: ignore


def build_probe_maze():
    # Maze (8x7) with a main corridor to goal and one side branch (dead-end) and one short false branch
    # S = start, G = goal, X walls
    # Layout y rows:
    # S . . . X . . .
    # X X . X X . X .
    # . . . . . . X .
    # . X X X . . X .
    # . . B . . . . .   (B central branch entry leads up to a pocket)
    # . X X X X X . .
    # . . . . . . . G
    maze = np.array([
        [0,0,0,0,1,0,0,0],
        [1,1,0,1,1,0,1,0],
        [0,0,0,0,0,0,1,0],
        [0,1,1,1,0,0,1,0],
        [0,0,0,0,0,0,0,0],
        [0,1,1,1,1,1,0,0],
        [0,0,0,0,0,0,0,0],
    ], dtype=int)
    start = (0,0)
    goal = (7,6)
    return maze, start, goal


def collect_branch_completion_signature(nav: MazeNavigator):
    # Map step->gedig record for convenience
    rec_by_step = {r['step']: r for r in nav.gedig_structural}
    # Find branch_completion event
    completion_events = [e for e in nav.event_log if e['type']=='branch_completion']
    if not completion_events:
        print('[WARN] No branch_completion event found.')
        return None
    comp_step = completion_events[0]['step']
    print(f"Branch completion step: {comp_step}")
    window_steps = list(range(max(0, comp_step-5), comp_step+3))
    rows = []
    for s in window_steps:
        rec = rec_by_step.get(s)
        if not rec: continue
        rows.append({
            'step': s,
            'value': rec['value'],
            'escalated': rec.get('escalated'),
            'shortcut': rec.get('shortcut'),
            'dead_end': rec.get('dead_end'),
            'nodes_added': rec.get('nodes_added'),
            'edges_added': rec.get('edges_added'),
            'density_change': rec.get('density_change'),
        })
    print('\nContext window around branch completion:')
    for r in rows:
        print(r)
    return comp_step


def derive_thresholds(nav: MazeNavigator, comp_step: int | None):
    if not nav.gedig_structural:
        return
    # Separate scores into: during exploration (nodes_added>0), stagnation-ish (nodes_added==0, edges_added<=1)
    exploration_scores = []
    low_growth_scores = []
    for rec in nav.gedig_structural:
        val = rec['value']
        if rec.get('nodes_added',0) > 0:
            exploration_scores.append(val)
        if rec.get('nodes_added',0)==0 and rec.get('edges_added',0)<=1:
            low_growth_scores.append(val)
    if not exploration_scores:
        return
    def stats(arr):
        if len(arr)==1:
            return {'mean': arr[0], 'std': 0.0}
        return {'mean': mean(arr), 'std': stdev(arr)}
    expl_stats = stats(exploration_scores)
    low_stats = stats(low_growth_scores) if low_growth_scores else None
    print('\nScore distribution:')
    print('  exploration count', len(exploration_scores), expl_stats)
    if low_stats:
        print('  low_growth  count', len(low_growth_scores), low_stats)
    # Candidate thresholds:
    # T1: midpoint between low_growth mean and exploration mean
    candidates = {}
    if low_stats:
        t1 = (expl_stats['mean'] + low_stats['mean'])/2
        candidates['midpoint'] = t1
    # T2: exploration mean - 1 std (conservative)
    candidates['expl_mean_minus1std'] = expl_stats['mean'] - expl_stats['std']
    # T3: low_growth mean (if available)
    if low_stats:
        candidates['low_growth_mean'] = low_stats['mean']
    print('\nThreshold candidates:')
    for k,v in candidates.items():
        print(f"  {k}: {v:.4f}")


def force_multihop_snapshots(nav: MazeNavigator, steps: list[int]):
    if not nav.gedig_structural:
        return
    print('\nForced multi-hop snapshots (re-evaluated)')
    structural = {r['step']: r for r in nav.gedig_structural}
    # Access underlying evaluator
    evaluator = nav.gedig_evaluator
    g_hist = nav.graph_manager.graph_history
    for s in steps:
        if s <=0 or s >= len(g_hist): continue
        g_prev = g_hist[s-1]
        g_now = g_hist[s]
        res = evaluator.evaluate_escalating(g_prev, g_now, escalation_threshold=10.0)  # always escalate
        print(f" step {s}: hop0={res['score']:.4f} multihop={res['multihop']} shortcut={res['shortcut']}")


def main():
    maze, start, goal = build_probe_maze()
    nav = MazeNavigator(
        maze=maze,
        start_pos=start,
        goal_pos=goal,
        wiring_strategy='simple',
        use_escalation=True,
        escalation_threshold=0.35,  # moderate: escalates when hop0 < 0.35
        gedig_threshold=0.25,
        backtrack_threshold=-0.05,
    )
    nav.run(max_steps=400)
    comp_step = collect_branch_completion_signature(nav)
    derive_thresholds(nav, comp_step)
    if comp_step is not None:
        force_multihop_snapshots(nav, [comp_step-1, comp_step, comp_step+1])

if __name__ == '__main__':
    main()
