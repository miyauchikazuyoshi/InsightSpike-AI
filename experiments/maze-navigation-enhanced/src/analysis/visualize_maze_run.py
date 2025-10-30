#!/usr/bin/env python3
"""
Visualize a single maze run: layout, path, backtrack events, and graph growth.

Outputs under: experiments/maze-navigation-enhanced/results/visualizations/
"""

from __future__ import annotations

import os
import sys
import json
import time
from typing import Tuple, List, Dict, Any

import numpy as np

# Local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from navigation.maze_navigator import MazeNavigator  # type: ignore
try:
    # Reuse maze generator
    from analysis.clean_maze_run import generate_maze  # type: ignore
except Exception:
    generate_maze = None  # type: ignore

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _generate_maze(size: int, seed: int) -> np.ndarray:
    if generate_maze is not None:
        return generate_maze(size, seed)  # type: ignore
    # Fallback simple generator
    rng = np.random.default_rng(seed)
    maze = np.ones((size, size), dtype=int)
    def carve(x: int, y: int):
        maze[y, x] = 0
    carve(1, 1)
    def nbs(x: int, y: int):
        for dx, dy in ((2, 0), (-2, 0), (0, 2), (0, -2)):
            nx, ny = x + dx, y + dy
            if 1 <= nx < size-1 and 1 <= ny < size-1:
                yield nx, ny, dx, dy
    stack = [(1, 1)]
    seen = {(1, 1)}
    while stack:
        x, y = stack[-1]
        cand = [(nx, ny, dx, dy) for nx, ny, dx, dy in nbs(x, y) if (nx, ny) not in seen]
        if not cand:
            stack.pop(); continue
        nx, ny, dx, dy = cand[rng.integers(0, len(cand))]
        maze[y + dy//2, x + dx//2] = 0
        maze[ny, nx] = 0
        seen.add((nx, ny))
        stack.append((nx, ny))
    maze[1, 1] = 0; maze[size-2, size-2] = 0
    return maze


def run_and_collect(size: int, seed: int, strategy: str, max_steps_factor: float | None = None, fast: bool = True,
                    gedig_threshold: float = -0.15, backtrack_threshold: float = -0.3, wiring_top_k: int | None = None,
                    overlay_sample: int = 5) -> Dict[str, Any]:
    maze = _generate_maze(size, seed)
    start = (1, 1); goal = (size-2, size-2)
    if max_steps_factor is None:
        max_steps_factor = 3.0 if fast else 4.0
    max_steps = int(size*size*max_steps_factor)

    # Configure fast mode via env flag
    if fast:
        os.environ['MAZE_FAST_MODE'] = '1'

    nav = MazeNavigator(
        maze=maze,
        start_pos=start,
        goal_pos=goal,
        wiring_strategy=strategy,
        gedig_threshold=gedig_threshold,
        backtrack_threshold=backtrack_threshold,
        simple_mode=fast,
        backtrack_debounce=True,
        wiring_top_k=(wiring_top_k if wiring_top_k is not None else (3 if fast else 4)),
        enable_diameter_metrics=(False if fast else True),
        dense_metric_interval=(25 if fast else 1),
        snapshot_skip_idle=(True if fast else False),
        max_graph_snapshots=(0 if fast else None),
        enable_flush=False,
        vector_index=None,
        ann_backend=None,
    )

    path: List[Tuple[int, int]] = []
    nodes_t: List[int] = []
    edges_t: List[int] = []
    gedig_t: List[float] = []
    events: List[Dict[str, Any]] = []
    # Lightweight L1 overlay (sampled) for viewer compatibility
    l1_topk_series: List[List[Dict[str, Any]]] = []
    # Approximate SP delta series (sampled)
    sp_delta_t: List[float | None] = []
    sp_prev: float | None = None
    last_ev = 0

    for step in range(max_steps):
        _ = nav.step()
        path.append(nav.current_pos)
        gs = nav.graph_manager.get_graph_statistics()
        nodes_t.append(gs.get('num_nodes', 0))
        edges_t.append(gs.get('num_edges', 0))
        gedig_t.append(nav.gedig_history[-1] if nav.gedig_history else np.nan)

        # Collect new events with step annotation
        evlog = getattr(nav, 'event_log', [])
        if evlog and len(evlog) > last_ev:
            for e in evlog[last_ev:]:
                e2 = dict(e)
                e2['at_step'] = step
                events.append(e2)
            last_ev = len(evlog)
        # L1 overlay (every k steps)
        try:
            k = max(0, int(overlay_sample))
        except Exception:
            k = 0
        if k == 0:
            # disabled
            pass
        else:
            if step % k == 0:
                try:
                    cx, cy = nav.current_pos
                    qv = nav.vector_processor.create_query_vector((cx, cy), prefer_unexplored=True)
                    eps = getattr(getattr(nav, 'episode_manager'), 'episodes_by_id', {})
                    cands = []
                    for eid, ep in eps.items():
                        try:
                            if getattr(ep, 'timestamp', 0) >= step:
                                continue
                            if getattr(ep, 'is_wall', False):
                                continue
                            v = np.asarray(ep.vector, dtype=float)
                            dv = float(np.linalg.norm(qv.astype(float) - v))
                            pos = (int(ep.position[0]), int(ep.position[1]))
                            cands.append((dv, pos, int(eid)))
                        except Exception:
                            continue
                    cands.sort(key=lambda t: t[0])
                    topk = (wiring_top_k if wiring_top_k is not None else (3 if fast else 4))
                    c = [
                        {'pos': [int(p[0]), int(p[1])], 'dv': float(dv), 'id': int(eid)}
                        for dv, p, eid in cands[: int(topk)]
                    ]
                    l1_topk_series.append(c)
                except Exception:
                    l1_topk_series.append([])
            else:
                l1_topk_series.append([])
            # SP delta approx (sampled at same cadence)
            try:
                import random
                import networkx as _nx
                g = nav.graph_manager.graph
                nodes = list(g.nodes())
                if len(nodes) >= 2:
                    S = min(20, max(2, len(nodes)//2))
                    pairs = [tuple(random.sample(nodes, 2)) for _ in range(S)]
                    dists = []
                    for u, v in pairs:
                        try:
                            d = _nx.shortest_path_length(g, u, v)
                            if isinstance(d, (int, float)):
                                dists.append(float(d))
                        except Exception:
                            continue
                    sp_avg = (sum(dists) / len(dists)) if dists else None
                else:
                    sp_avg = None
                if sp_avg is None:
                    sp_delta_t.append(None)
                else:
                    delta = 0.0 if sp_prev is None else (sp_avg - sp_prev)
                    sp_prev = sp_avg
                    sp_delta_t.append(float(delta))
            except Exception:
                sp_delta_t.append(None)
        
        if nav.current_pos == goal:
            break

    # Optional: multi-hop / query / bt-eval (best-effort extraction)
    gedig_full_t: List[float | None] = []
    query_eval_t: List[float | None] = []
    query_full_t: List[float | None] = []
    bt_eval_t: List[float | None] = []
    try:
        recs = getattr(nav, 'gedig_structural', []) or []
        for rec in recs:
            if isinstance(rec, dict):
                mh = rec.get('multihop')
                qv = rec.get('query_eval')
                qfull = rec.get('query_full_min')
                bt = rec.get('bt_eval_value')
                if isinstance(mh, dict) and mh:
                    try:
                        gedig_full_t.append(float(min(float(v) for v in mh.values())))
                    except Exception:
                        gedig_full_t.append(None)
                else:
                    gedig_full_t.append(None)
                query_eval_t.append(float(qv) if isinstance(qv, (int, float)) else None)
                query_full_t.append(float(qfull) if isinstance(qfull, (int, float)) else None)
                bt_eval_t.append(float(bt) if isinstance(bt, (int, float)) else None)
    except Exception:
        pass
    # Normalize series lengths to match gedig_t
    n = len(gedig_t)
    if not any((x is not None) for x in gedig_full_t):
        gedig_full_t = [float(x) if isinstance(x, (int, float)) else None for x in gedig_t]
    if len(gedig_full_t) != n:
        gedig_full_t = [float(x) if isinstance(x, (int, float)) else None for x in gedig_t]
    if len(query_eval_t) != n:
        query_eval_t = [None] * n
    if len(query_full_t) != n:
        query_full_t = [None] * n
    if len(bt_eval_t) != n:
        bt_eval_t = [None] * n
    if len(sp_delta_t) != n:
        # pad with None to match timeline
        sp_delta_t = (sp_delta_t + [None] * n)[:n]

    # Replace NaN with None for JSON compatibility
    def _nan_to_none(arr: List[float]) -> List[float | None]:
        out: List[float | None] = []
        for v in arr:
            try:
                if v is None:
                    out.append(None)
                elif isinstance(v, (int, float)) and not np.isnan(v):
                    out.append(float(v))
                else:
                    out.append(None)
            except Exception:
                out.append(None)
        return out

    gedig_t_sanitized = _nan_to_none(gedig_t)
    gedig_full_t = _nan_to_none(gedig_full_t)

    return {
        'maze': maze.tolist(),
        'size': size,
        'seed': seed,
        'strategy': strategy,
        'max_steps': max_steps,
        'goal_reached': (path[-1] == goal) if path else False,
        'steps': len(path),
        'path': path,
        # For viewer compatibility
        'graph_nodes_t': nodes_t,
        'graph_edges_t': edges_t,
        'l1_topk': l1_topk_series,
        'gedig_full_t': gedig_full_t,
        'query_eval_t': query_eval_t,
        'query_full_t': query_full_t,
        'sp_delta_t': sp_delta_t,
        'bt_eval_t': bt_eval_t,
        'nodes_t': nodes_t,
        'edges_t': edges_t,
        'gedig_t': gedig_t_sanitized,
        'events': events,
    }


def _to_jsonable(o: Any):
    try:
        import numpy as _np
    except Exception:
        _np = None
    # numpy scalars
    if _np is not None and isinstance(o, _np.generic):
        return o.item()
    # numpy arrays
    if _np is not None and isinstance(o, _np.ndarray):
        return o.tolist()
    # tuples -> lists
    if isinstance(o, tuple):
        return [_to_jsonable(x) for x in o]
    # lists
    if isinstance(o, list):
        return [_to_jsonable(x) for x in o]
    # dicts
    if isinstance(o, dict):
        return {str(k): _to_jsonable(v) for k, v in o.items()}
    # enums or other objects
    try:
        import enum as _enum
        if isinstance(o, _enum.Enum):
            return str(o.value)
    except Exception:
        pass
    # pass-through for primitives
    return o


def _write_html(run: Dict[str, Any], outdir: str) -> None:
    html = f"""
<!DOCTYPE html>
<html lang=\"en\">
<meta charset=\"utf-8\" />
<title>Maze {run['size']} seed={run['seed']} {run['strategy']}</title>
<style>body{{font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Ubuntu; margin:20px}} img{{max-width:100%}}</style>
<h2>Maze {run['size']}×{run['size']} | seed={run['seed']} | strategy={run['strategy']}</h2>
<ul>
  <li>goal_reached: {str(run.get('goal_reached'))}</li>
  <li>steps: {run.get('steps')}</li>
  <li>gedig_threshold: {run.get('gedig_threshold','')}</li>
  <li>backtrack_threshold: {run.get('backtrack_threshold','')}</li>
  <li>wiring_top_k: {run.get('wiring_top_k','')}</li>
  <li>max_steps: {run.get('max_steps')}</li>
  <li>events: {len(run.get('events') or [])}</li>
 </ul>
<h3>Path</h3>
<img src=\"maze_path.png\" />
<h3>Graph growth</h3>
<img src=\"graph_growth.png\" />
<h3>geDIG timeline</h3>
<img src=\"gedig_timeline.png\" />
<p><small>Generated {time.strftime('%Y-%m-%d %H:%M:%S')}</small></p>
"""
    with open(os.path.join(outdir, 'index.html'), 'w') as f:
        f.write(html)

def save_visuals(run: Dict[str, Any], outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    # Save JSON
    with open(os.path.join(outdir, 'run_summary.json'), 'w') as f:
        json.dump(_to_jsonable(run), f, indent=2, ensure_ascii=False)

    size = run['size']; maze = np.array(run['maze'], dtype=int)
    path = run['path']
    nodes_t = run['nodes_t']; edges_t = run['edges_t']; gedig_t = run['gedig_t']
    events = run['events']

    # 1) Maze with path overlay
    plt.figure(figsize=(5, 5))
    plt.imshow(maze, cmap='gray_r')
    if path:
        xs = [p[0] for p in path]; ys = [p[1] for p in path]
        plt.plot(xs, ys, 'r-', linewidth=1.5, alpha=0.8, label='path')
        plt.scatter([1, size-2], [1, size-2], c=['green', 'blue'], s=40, label='start/goal')
    plt.title(f"Maze {size}x{size} | seed={run['seed']} | {run['strategy']} | steps={run['steps']}")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'maze_path.png'), dpi=160)
    plt.close()

    # 2) Graph growth
    plt.figure(figsize=(7, 3))
    plt.plot(nodes_t, label='nodes', color='tab:blue')
    plt.plot(edges_t, label='edges', color='tab:orange')
    plt.xlabel('step'); plt.ylabel('count'); plt.title('Graph growth over steps')
    plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'graph_growth.png'), dpi=160)
    plt.close()

    # 3) geDIG timeline with backtrack markers
    plt.figure(figsize=(7, 3))
    plt.plot(gedig_t, label='geDIG', color='tab:green')
    # mark backtrack triggers if available
    bt_steps = [e['at_step'] for e in events if str(e.get('type','')).lower().endswith('backtrack_trigger') or e.get('type')=='backtrack_trigger']
    for s in bt_steps:
        plt.axvline(s, color='red', linestyle='--', alpha=0.4)
    plt.xlabel('step'); plt.ylabel('geDIG'); plt.title('geDIG timeline')
    plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'gedig_timeline.png'), dpi=160)
    plt.close()
    # HTML bundle
    _write_html(run, outdir)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description='Visualize a single maze run with geDIG')
    parser.add_argument('--size', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--strategy', type=str, default='gedig', choices=['gedig','simple','gedig_optimized'])
    parser.add_argument('--factor', type=float, default=None, help='Step budget factor * size^2 (default 3.0 in fast mode)')
    parser.add_argument('--gedig-threshold', type=float, default=-0.15)
    parser.add_argument('--backtrack-threshold', type=float, default=-0.3)
    parser.add_argument('--topk', type=int, default=None)
    parser.add_argument('--fast', action='store_true', help='Enable fast mode')
    parser.add_argument('--overlay-sample', type=int, default=5, help='Record L1 overlay every k steps (0=off)')
    parser.add_argument('--outdir', type=str, default=None)
    args = parser.parse_args()

    ts = time.strftime('%Y%m%d_%H%M%S')
    base_out = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'visualizations'))
    if args.outdir:
        outdir = args.outdir
    else:
        outdir = os.path.join(base_out, f"maze{args.size}_seed{args.seed}_{args.strategy}_{ts}")

    run = run_and_collect(size=args.size, seed=args.seed, strategy=args.strategy, max_steps_factor=args.factor, fast=args.fast,
                         gedig_threshold=args.gedig_threshold, backtrack_threshold=args.backtrack_threshold, wiring_top_k=args.topk,
                         overlay_sample=int(args.overlay_sample))
    # annotate thresholds for HTML
    run['gedig_threshold'] = args.gedig_threshold
    run['backtrack_threshold'] = args.backtrack_threshold
    run['wiring_top_k'] = args.topk
    save_visuals(run, outdir)
    print(f"Saved visualization to: {outdir}")


if __name__ == '__main__':
    main()
