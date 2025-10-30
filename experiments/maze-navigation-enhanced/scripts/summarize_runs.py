#!/usr/bin/env python3
"""Summarize run_summary.json folders into CSV on stdout.

Usage:
  python summarize_runs.py path/to/run_dir1 path/to/run_dir2 ... > out.csv
"""
from __future__ import annotations
import sys, json, pathlib
from typing import Any


def load_run(path: pathlib.Path) -> dict[str, Any] | None:
    p = path / 'run_summary.json'
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def coverage(maze: list[list[int]] | None, path_xy: list[list[int]] | None) -> float | None:
    if not maze or not path_xy:
        return None
    open_cells = sum(1 for row in maze for v in row if v == 0)
    uniq = len({tuple(p) for p in path_xy})
    return uniq / max(1, open_cells)


def main(argv: list[str]) -> None:
    if len(argv) < 2:
        print('seed_dir,... -> CSV on stdout', file=sys.stderr)
        sys.exit(1)
    dirs = [pathlib.Path(a) for a in argv[1:]]
    cols = [
        'size','seed','steps','goal','edges','l1_force_edges','l1_sim_edges','bt_triggers',
        'gedig_mean','gedig_min','sp_neg_count','nodes_total','edges_total','coverage'
    ]
    print(','.join(cols))
    for d in dirs:
        js = load_run(d)
        if not js:
            continue
        size = js.get('size'); seed = js.get('seed'); steps = js.get('steps'); goal = js.get('goal_reached')
        edges = js.get('graph_edges') or []
        edge_total = len(edges)
        l1_force = sum(1 for e in edges if isinstance(e, dict) and isinstance(e.get('s'), str) and e['s'].startswith('l1_force'))
        l1_sim = sum(1 for e in edges if isinstance(e, dict) and isinstance(e.get('s'), str) and ('l1_sim' in e['s']))
        # events
        ev = js.get('events') or []
        bt = sum(1 for e in ev if isinstance(e, dict) and e.get('type') == 'backtrack_trigger')
        # geDIG
        g = [v for v in (js.get('gedig_t') or []) if isinstance(v, (int, float))]
        gmean = sum(g)/len(g) if g else None
        gmin = min(g) if g else None
        # ΔSP negatives
        sp = [v for v in (js.get('sp_delta_t') or []) if isinstance(v, (int, float))]
        spneg = sum(1 for v in sp if v < 0)
        # growth totals
        gn = js.get('graph_nodes_t') or []
        ge = js.get('graph_edges_t') or []
        n_total = int(gn[-1]) if gn else 0
        e_total = int(ge[-1]) if ge else 0
        cov = coverage(js.get('maze'), js.get('path'))
        row = [
            size, seed, steps, goal, edge_total, l1_force, l1_sim, bt,
            (f"{gmean:.6f}" if gmean is not None else ''), (f"{gmin:.6f}" if gmin is not None else ''), spneg,
            n_total, e_total, (f"{cov:.4f}" if cov is not None else '')
        ]
        print(','.join(str(x) for x in row))


if __name__ == '__main__':
    main(sys.argv)
