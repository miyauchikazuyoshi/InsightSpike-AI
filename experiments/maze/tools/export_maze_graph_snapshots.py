#!/usr/bin/env python3
from __future__ import annotations

"""
Export paired maze+graph static snapshots from a run_experiment_query.py run.

Inputs:
  --summary: summary JSON
  --steps:   steps JSON
  --sqlite:  SQLite DS path (to reconstruct graph up to step)
  --namespace: DS namespace used at run
  --out-dir: output directory for PNGs
  --select:  which scenes to export: auto | corridor | tjunction | deadend
  --seed:    optional seed to pick records from (default: first seed)
  --limit:   max number of scenes to export (default: 3 for auto)

Outputs: one PNG per selected scene (left: maze, right: graph) as
  maze_graph_seed{seed}_step{step}_{type}.png
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_pos(pos: Any) -> Optional[Tuple[int, int]]:
    if isinstance(pos, list) and len(pos) >= 2:
        return int(pos[0]), int(pos[1])
    return None


def load_ds_graph_upto_step(sqlite_path: Path, namespace: str, step: int) -> Tuple[List[List[int]], List[List[List[int]]]]:
    # Best-effort import by path to avoid package name issues
    import importlib.util as _imp_util
    import sys as _sys
    present_path = (Path(__file__).with_name("..") / "qhlib" / "present.py").resolve()
    if not present_path.exists():
        present_path = (Path(__file__).with_name("present.py")).resolve()
    spec = _imp_util.spec_from_file_location("qh_present", str(present_path))
    mod = _imp_util.module_from_spec(spec)
    _sys.modules["qh_present"] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod.load_ds_graph_upto_step(str(sqlite_path), namespace, int(step))


def draw_maze(ax: plt.Axes, layout: np.ndarray, path: List[Tuple[int, int]], cur: Tuple[int, int]) -> None:
    h, w = layout.shape
    ax.set_aspect("equal")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    # Cells
    for r in range(h):
        for c in range(w):
            if int(layout[r, c]) == 1:
                ax.add_patch(plt.Rectangle((c, r), 1, 1, color="#1f2937"))
            else:
                ax.add_patch(plt.Rectangle((c, r), 1, 1, color="#f8fafc"))
    # Grid
    for r in range(h + 1):
        ax.plot([0, w], [r, r], color="#e5e7eb", lw=0.5)
    for c in range(w + 1):
        ax.plot([c, c], [0, h], color="#e5e7eb", lw=0.5)
    # Path
    if path:
        xs = [p[1] + 0.5 for p in path]
        ys = [p[0] + 0.5 for p in path]
        ax.plot(xs, ys, color="#4c6ef5", lw=2.5, alpha=0.7)
    # Current
    ax.add_patch(plt.Circle((cur[1] + 0.5, cur[0] + 0.5), 0.35, color="#2563eb"))


def draw_graph(ax: plt.Axes, nodes: List[List[int]], edges: List[List[List[int]]]) -> None:
    ax.set_aspect("equal")
    ax.axis("off")
    if not nodes:
        ax.text(0.5, 0.5, "(no graph)", ha="center", va="center", transform=ax.transAxes)
        return
    # Map node id [r,c,d] to 2D positions on grid with small dir offset
    pts: Dict[Tuple[int, int, int], Tuple[float, float]] = {}
    for n in nodes:
        if len(n) < 3:
            continue
        r, c, d = int(n[0]), int(n[1]), int(n[2])
        x, y = float(c), float(r)
        # directional offset
        dx = dy = 0.0
        if d == 0:  # N
            dy = -0.25
        elif d == 1:  # E
            dx = 0.25
        elif d == 2:  # S
            dy = 0.25
        elif d == 3:  # W
            dx = -0.25
        pts[(r, c, d)] = (x + dx, y + dy)
    # Draw edges
    for e in edges:
        if not (isinstance(e, list) and len(e) == 2):
            continue
        u, v = e
        if len(u) < 3 or len(v) < 3:
            continue
        ku = (int(u[0]), int(u[1]), int(u[2]))
        kv = (int(v[0]), int(v[1]), int(v[2]))
        if ku in pts and kv in pts:
            xu, yu = pts[ku]
            xv, yv = pts[kv]
            ax.plot([xu, xv], [yu, yv], color="#93c5fd", lw=1.6, alpha=0.8)
    # Draw nodes (query vs direction)
    for (r, c, d), (x, y) in pts.items():
        if d == -1:
            ax.scatter([x], [y], s=60, c="#a855f7", edgecolors="#7e22ce", zorder=3)
        else:
            ax.scatter([x], [y], s=40, c="#60a5fa", edgecolors="#1d4ed8", zorder=2)
    # Frame to fit content
    xs = [p[0] for p in pts.values()]
    ys = [p[1] for p in pts.values()]
    if xs and ys:
        pad = 1.5
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(max(ys) + pad, min(ys) - pad)


def select_scenes(steps: List[Dict[str, Any]], limit: int, mode: str) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    if mode == "auto":
        # pick first corridor (2 moves), first tjunction (>=3 moves), first deadend (<=1 move)
        flags = {"corridor": False, "tjunction": False, "deadend": False}
        for rec in steps:
            pm = rec.get("possible_moves") or []
            typ = None
            if isinstance(pm, list):
                if len(pm) == 2 and not flags["corridor"]:
                    typ = "corridor"
                    flags["corridor"] = True
                elif len(pm) >= 3 and not flags["tjunction"]:
                    typ = "tjunction"
                    flags["tjunction"] = True
                elif len(pm) <= 1 and not flags["deadend"]:
                    typ = "deadend"
                    flags["deadend"] = True
            if typ is not None:
                out.append((int(rec.get("step", 0)), typ))
            if len(out) >= limit:
                break
    else:
        for rec in steps:
            pm = rec.get("possible_moves") or []
            if mode == "corridor" and isinstance(pm, list) and len(pm) == 2:
                out.append((int(rec.get("step", 0)), mode))
                break
            if mode == "tjunction" and isinstance(pm, list) and len(pm) >= 3:
                out.append((int(rec.get("step", 0)), mode))
                break
            if mode == "deadend" and isinstance(pm, list) and len(pm) <= 1:
                out.append((int(rec.get("step", 0)), mode))
                break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Export maze+graph snapshots")
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--steps", type=Path, required=True)
    ap.add_argument("--sqlite", type=Path, required=True)
    ap.add_argument("--namespace", type=str, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--select", type=str, default="auto", choices=["auto","corridor","tjunction","deadend"]) 
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--limit", type=int, default=3)
    args = ap.parse_args()

    summary = load_json(args.summary)
    steps_all = load_json(args.steps)
    # Maze layout
    maze = (summary.get("maze_data") or {}).get("0") or summary.get("maze_data")
    if isinstance(maze, dict) and "layout" in maze:
        layout = np.array(maze["layout"], dtype=int)
        start = tuple(maze.get("start_pos") or (0, 0))
    else:
        # fallback: infer size from first record
        layout = np.zeros((25, 25), dtype=int)
        start = (0, 0)

    # Group steps by seed
    by_seed: Dict[int, List[Dict[str, Any]]] = {}
    for rec in steps_all:
        if not isinstance(rec, dict):
            continue
        s = int(rec.get("seed", 0))
        by_seed.setdefault(s, []).append(rec)
    seed = args.seed if args.seed is not None else (sorted(by_seed.keys())[0] if by_seed else 0)
    recs = sorted(by_seed.get(seed, []), key=lambda r: int(r.get("step", 0)))

    scenes = select_scenes(recs, args.limit, args.select)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for step_idx, typ in scenes:
        rec = next((r for r in recs if int(r.get("step", 0)) == step_idx), None)
        if not rec:
            continue
        # Build path up to step
        path: List[Tuple[int, int]] = []
        for r in recs:
            s = int(r.get("step", 0))
            if s > step_idx:
                break
            p = normalize_pos(r.get("position"))
            if p: path.append(p)
        cur = path[-1] if path else start

        # Load DS graph up to step
        ds_nodes, ds_edges = load_ds_graph_upto_step(args.sqlite, args.namespace, step_idx)

        # Compose figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        draw_maze(ax1, layout, path, cur)
        ax1.set_title(f"Maze (seed={seed}, step={step_idx}, {typ})", fontsize=10)
        draw_graph(ax2, ds_nodes, ds_edges)
        ax2.set_title("Graph (DS up to step)", fontsize=10)
        fig.tight_layout()
        out = args.out_dir / f"maze_graph_seed{seed}_step{step_idx}_{typ}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print("[wrote]", out)


if __name__ == "__main__":
    main()

