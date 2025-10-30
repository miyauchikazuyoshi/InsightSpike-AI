#!/usr/bin/env python3
"""
Sweep NA/BT thresholds using query-0hop (virtual) while running the maze,
and write a compact CSV summary. Designed for quick calibration by "if"-like gating.

Usage (examples):
  python experiments/maze-navigation-enhanced/scripts/threshold_sweep_query0hop.py \
      --seeds 16,17,19 \
      --size 15 --max-steps 300 \
      --na-list -0.06,-0.055,-0.05,-0.045,-0.04 \
      --bt-list -0.06,-0.055,-0.05 \
      --out experiments/maze-navigation-enhanced/results/threshold_sweep/query0hop_sweep.csv

Notes:
  - Forces: MAZE_QUERY_EVAL_MODE=virtual, MAZE_NA_USE_QUERY=1, MAZE_NA_USE_QUERY_ONLY=1, MAZE_BT_USE_QUERY_MIN=1
  - Keeps structure clean (no hub persist). Measures steps/goal/BT数/loop_sp数/query_evalの要約。
"""
from __future__ import annotations

import os, sys, csv, argparse
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
SRC = ROOT / 'experiments' / 'maze-navigation-enhanced' / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from navigation.maze_navigator import MazeNavigator  # type: ignore
from visualization.export_threshold_dfs_run import generate_test_maze  # type: ignore


def run_once(seed: int, size: int, max_steps: int, na_thr: float, bt_thr: float) -> Dict[str, Any]:
    # Environment gating knobs (query-0hop only)
    os.environ['MAZE_QUERY_EVAL_MODE'] = 'virtual'
    os.environ['MAZE_NA_USE_QUERY'] = '1'
    os.environ['MAZE_NA_USE_QUERY_ONLY'] = '1'
    os.environ['MAZE_BT_USE_QUERY_MIN'] = '1'
    os.environ['MAZE_USE_QUERY_HUB'] = '0'
    os.environ['MAZE_QUERY_HUB_PERSIST'] = '0'

    os.environ['MAZE_NA_GE_THRESH'] = str(na_thr)
    os.environ['MAZE_BACKTRACK_THRESHOLD'] = str(bt_thr)

    # Reasonable defaults for reproducibility
    os.environ.setdefault('MAZE_L1_NORM_SEARCH', '1')
    os.environ.setdefault('MAZE_L1_NORM_TAU', '0.6')
    os.environ.setdefault('MAZE_EVAL_CAND_TOPK', '8')
    os.environ.setdefault('MAZE_WIRING_MIN_ACCEPT', '1')
    os.environ.setdefault('MAZE_WIRING_TOPK', '1')

    maze = generate_test_maze(size=size, seed=seed, braid_prob=0.0)
    start = (1, 1); goal = (size-2, size-2)
    nav = MazeNavigator(maze=maze, start_pos=start, goal_pos=goal,
                        wiring_strategy='gedig', gedig_threshold=-0.03,
                        backtrack_threshold=bt_thr, simple_mode=True,
                        use_escalation=True, escalation_threshold=None,
                        backtrack_target_strategy='heuristic')
    nav.run(max_steps=max_steps)

    # Summarize
    ev = nav.event_log or []
    bt = [e for e in ev if e.get('type') == 'backtrack_plan']
    loops = [e for e in ev if e.get('type') == 'backtrack_plan' and (e.get('message') or {}).get('reason') == 'loop_sp']
    q = [rec.get('query_eval') for rec in (nav.gedig_structural or [])]
    qv = [float(v) for v in q if isinstance(v, (int, float))]
    g = [rec.get('value') for rec in (nav.gedig_structural or [])]
    gv = [float(v) for v in g if isinstance(v, (int, float))]
    full = []
    for rec in (nav.gedig_structural or []):
        mh = rec.get('multihop')
        if isinstance(mh, dict) and mh:
            try:
                full.append(min(float(v) for v in mh.values() if isinstance(v, (int, float))))
            except Exception:
                pass
    out: Dict[str, Any] = {
        'seed': seed,
        'size': size,
        'max_steps': max_steps,
        'na_thr': na_thr,
        'bt_thr': bt_thr,
        'steps': nav.step_count,
        'goal': int(bool(nav.is_goal_reached)),
        'bt_plans': len(bt),
        'loop_sp_plans': len(loops),
        'query_eval_mean': (sum(qv)/len(qv) if qv else ''),
        'g0_min': (min(gv) if gv else ''),
        'gfull_min': (min(full) if full else ''),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description='Sweep NA/BT thresholds (query-0hop) and summarize')
    ap.add_argument('--seeds', type=str, default='16,17,19')
    ap.add_argument('--size', type=int, default=15)
    ap.add_argument('--max-steps', type=int, default=300)
    ap.add_argument('--na-list', type=str, default='-0.06,-0.055,-0.05,-0.045,-0.04')
    ap.add_argument('--bt-list', type=str, default='-0.06,-0.055,-0.05')
    ap.add_argument('--out', type=str, default='')
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(',') if x.strip()]
    na_list = [float(x) for x in args.na_list.split(',') if x.strip()]
    bt_list = [float(x) for x in args.bt_list.split(',') if x.strip()]

    rows: List[Dict[str, Any]] = []
    for s in seeds:
        for na in na_list:
            for bt in bt_list:
                rec = run_once(s, args.size, args.max_steps, na, bt)
                rows.append(rec)
                print(f"seed={s} na={na} bt={bt} -> steps={rec['steps']} goal={rec['goal']} bt={rec['bt_plans']}")

    out_path = Path(args.out) if args.out else (ROOT / 'experiments' / 'maze-navigation-enhanced' / 'results' / 'threshold_sweep' / 'query0hop_sweep.csv')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['seed','size','max_steps','na_thr','bt_thr','steps','goal','bt_plans','loop_sp_plans','query_eval_mean','g0_min','gfull_min']
    with out_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)
    print(f"[Saved] {out_path}")


if __name__ == '__main__':
    main()

