#!/usr/bin/env python3
"""
Visualize a maze trajectory with backtrack triggers overlaid.

Usage:
  python visualize_path_backtracks.py --log /path/to/detailed_log.json [--out out.png]

If --log is omitted, the script searches the latest baseline_compare_* run under:
  experiments/maze-navigation-enhanced/results/maze_report/

Optional:
  --maze {ultra50hd,ultra50,ultra50md}
    ultra50hd  -> create_ultra_maze_50_dense_deadends (default)
    ultra50    -> create_ultra_maze_50
    ultra50md  -> create_ultra_maze_50_moderate_deadends
  --title <string>  Title override for the plot
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Resolve paths to import experiment-local modules without absolute paths
HERE = Path(__file__).resolve()
EXP_ROOT = HERE.parents[1]  # experiments/maze-navigation-enhanced
SRC_EXP = EXP_ROOT / 'src' / 'experiments'
SRC_ROOT = EXP_ROOT / 'src'
import sys
for p in [SRC_EXP, SRC_ROOT]:
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

# Import maze layout generators and default start/goal
from maze_layouts import (
    create_ultra_maze_50_dense_deadends,
    create_ultra_maze_50,
    create_ultra_maze_50_moderate_deadends,
    ULTRA50HD_DEFAULT_START, ULTRA50HD_DEFAULT_GOAL,
    ULTRA50_DEFAULT_START, ULTRA50_DEFAULT_GOAL,
    ULTRA50MD_DEFAULT_START, ULTRA50MD_DEFAULT_GOAL,
)


def find_latest_log(base: Path) -> Path | None:
    """Find the newest detailed_log.json under baseline_compare_* in base dir."""
    if not base.exists():
        return None
    candidates: list[Path] = []
    for child in base.iterdir():
        if not child.is_dir():
            continue
        if not child.name.startswith('baseline_compare_'):
            continue
        p = child / 'detailed_log.json'
        if p.exists():
            candidates.append(p)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def extract_backtrack_positions(events: Iterable[Any]) -> List[Tuple[int, int]]:
    out: list[Tuple[int, int]] = []
    for e in events:
        try:
            etype = str(e.get('type') or e.get('event') or e.get('name') or '').lower()
            if 'backtrack_trigger' not in etype:
                continue
            payload = e.get('payload') if isinstance(e, dict) else None
            pos = None
            if isinstance(payload, dict):
                pos = payload.get('position') or payload.get('pos')
            if pos is None:
                pos = e.get('position') or e.get('pos')
            if isinstance(pos, (list, tuple)) and len(pos) == 2:
                r, c = int(pos[0]), int(pos[1])
                out.append((r, c))
        except Exception:
            continue
    return out


def load_events_any(js: dict) -> List[dict]:
    for key in ('events_full', 'events', 'event_log', 'eventLog'):
        v = js.get(key)
        if isinstance(v, list):
            return v  # type: ignore[return-value]
    return []


def main() -> None:
    ap = argparse.ArgumentParser(description='Visualize path with backtrack triggers')
    ap.add_argument('--log', type=str, default='', help='Path to detailed_log.json')
    ap.add_argument('--out', type=str, default='', help='Output PNG path')
    ap.add_argument('--maze', type=str, default='ultra50hd', choices=['ultra50hd','ultra50','ultra50md'], help='Maze layout variant')
    ap.add_argument('--title', type=str, default='', help='Plot title override')
    args = ap.parse_args()

    # Resolve log path
    if args.log:
        log_path = Path(args.log).expanduser().resolve()
    else:
        base = EXP_ROOT / 'results' / 'maze_report'
        lp = find_latest_log(base)
        if lp is None:
            print(f'Error: no detailed_log.json found under {base}')
            return
        log_path = lp

    if not log_path.exists():
        print(f'Error: log not found: {log_path}')
        return

    # Resolve output path
    if args.out:
        out_path = Path(args.out).expanduser().resolve()
    else:
        if log_path.name == 'detailed_log.json':
            out_path = log_path.parent / 'path_with_backtracks.png'
        else:
            out_path = log_path.with_suffix('.png')

    # Load JSON
    with log_path.open('r', encoding='utf-8') as f:
        js = json.load(f)

    # Extract path array
    path_data = js.get('path') or js.get('path_xy')
    if not path_data:
        print('Error: no path found in log')
        return
    path_arr = np.array(path_data, dtype=int)
    if path_arr.ndim != 2 or path_arr.shape[1] != 2:
        print('Error: path is not Nx2 array')
        return

    # Extract events and backtrack points
    events = load_events_any(js)
    bt_points = extract_backtrack_positions(events)
    bt_arr = np.array(bt_points, dtype=int) if bt_points else None

    # Build maze for background
    if args.maze == 'ultra50':
        maze = create_ultra_maze_50()
        start, goal = ULTRA50_DEFAULT_START, ULTRA50_DEFAULT_GOAL
    elif args.maze == 'ultra50md':
        maze = create_ultra_maze_50_moderate_deadends()
        start, goal = ULTRA50MD_DEFAULT_START, ULTRA50MD_DEFAULT_GOAL
    else:  # ultra50hd
        maze = create_ultra_maze_50_dense_deadends()
        start, goal = ULTRA50HD_DEFAULT_START, ULTRA50HD_DEFAULT_GOAL

    # Plot
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(maze, cmap='gray_r', interpolation='nearest')
    ax.plot(path_arr[:, 1], path_arr[:, 0], 'b-', alpha=0.6, linewidth=1.5, label='Agent Path')
    ax.plot(start[1], start[0], 'go', markersize=12, label='Start', markeredgecolor='black')
    ax.plot(goal[1], goal[0], 'r*', markersize=16, label='Goal', markeredgecolor='black')
    if bt_arr is not None and bt_arr.size > 0:
        ax.plot(bt_arr[:, 1], bt_arr[:, 0], 'cx', markersize=10, markeredgewidth=2.5, label=f'Backtrack Trigger (n={len(bt_arr)})')

    title = args.title or f'Maze Trajectory with Backtrack Events (seed={js.get("seed", "?")})'
    ax.set_title(title)
    ax.axis('off')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.85)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.1, dpi=150)
    print(f'[Saved] {out_path}')
    plt.close(fig)


if __name__ == '__main__':
    main()

