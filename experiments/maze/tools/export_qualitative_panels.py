#!/usr/bin/env python3
from __future__ import annotations

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


def draw_panel(
    layout: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    pos: Tuple[int, int],
    chosen_action: Optional[int],
    possible_moves: List[int],
    out: Path,
) -> None:
    h, w = layout.shape
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.set_aspect("equal")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    # Draw walls
    for r in range(h):
        for c in range(w):
            if layout[r, c] == 1:
                ax.add_patch(plt.Rectangle((c, r), 1, 1, color="#222", lw=0))
    # Grid lines (light)
    for r in range(h + 1):
        ax.plot([0, w], [r, r], color="#e0e6ef", lw=0.5)
    for c in range(w + 1):
        ax.plot([c, c], [0, h], color="#e0e6ef", lw=0.5)
    # Start/goal
    ax.add_patch(plt.Circle((start[1] + 0.5, start[0] + 0.5), 0.35, color="#0ea5e9", alpha=0.5))
    ax.add_patch(plt.Circle((goal[1] + 0.5, goal[0] + 0.5), 0.35, color="#ef4444", alpha=0.5))
    # Current pos
    ax.add_patch(plt.Circle((pos[1] + 0.5, pos[0] + 0.5), 0.32, color="#1c7ed6"))
    # Arrows for moves
    # 0: up, 1: right, 2: down, 3: left
    dir_vec = {
        0: (-1, 0),
        1: (0, 1),
        2: (1, 0),
        3: (0, -1),
    }
    # possible moves (light cyan)
    for a in possible_moves:
        dr, dc = dir_vec.get(int(a), (0, 0))
        ax.arrow(
            pos[1] + 0.5,
            pos[0] + 0.5,
            0.55 * dc,
            0.55 * dr,
            width=0.05,
            head_width=0.25,
            head_length=0.25,
            length_includes_head=True,
            color="#7dd3fc",
            alpha=0.8,
        )
    # chosen action (bold blue)
    if chosen_action is not None and chosen_action in dir_vec:
        dr, dc = dir_vec[int(chosen_action)]
        ax.arrow(
            pos[1] + 0.5,
            pos[0] + 0.5,
            0.7 * dc,
            0.7 * dr,
            width=0.08,
            head_width=0.32,
            head_length=0.32,
            length_includes_head=True,
            color="#2563eb",
        )
    ax.axis("off")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def find_first(records: List[Dict[str, Any]], cond) -> Optional[Dict[str, Any]]:
    for rec in records:
        if cond(rec):
            return rec
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Export qualitative panels from steps.json")
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--steps", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--maze-json", type=Path, default=None, help="Optional maze snapshot JSON with layout/start/goal/size")
    args = ap.parse_args()

    summary = load_json(args.summary)
    steps = load_json(args.steps)
    # Load maze snapshot
    maze: Dict[str, Any] = {}
    if args.maze_json and args.maze_json.exists():
        maze = load_json(args.maze_json)
    elif isinstance(summary, dict):
        md = (summary.get("maze_data") or {})
        if md:
            # pick first seed
            any_seed = next(iter(md.values()))
            if isinstance(any_seed, dict):
                maze = any_seed
    if not maze:
        raise SystemExit("Maze snapshot not found. Provide --maze-json or ensure summary.maze_data is present.")

    layout = np.array(maze.get("layout"), dtype=int)
    start = tuple(maze.get("start_pos") or (1, 1))
    goal = tuple(maze.get("goal_pos") or (layout.shape[0] - 2, layout.shape[1] - 2))

    # Conditions
    def is_corridor(rec):
        pm = rec.get("possible_moves") or []
        return (not rec.get("is_dead_end", False)) and (not rec.get("is_junction", False)) and (len(pm) == 2)

    def is_t_junction(rec):
        pm = rec.get("possible_moves") or []
        # Fallback: treat 3-way or more as junction when explicit flag is absent
        return bool(rec.get("is_junction", False)) or (len(pm) >= 3)

    def is_dead_end(rec):
        return bool(rec.get("is_dead_end", False))

    rc = find_first(steps, is_corridor)
    rt = find_first(steps, is_t_junction)
    rd = find_first(steps, is_dead_end)

    # Draw panels
    if rc:
        pos = tuple(rc.get("position") or (start[0], start[1]))
        action = rc.get("action")
        pm = list(rc.get("possible_moves") or [])
        draw_panel(layout, start, goal, pos, None, pm, args.out_dir / "corridor_panel.png")
    if rt:
        pos = tuple(rt.get("position") or (start[0], start[1]))
        action = rt.get("action")
        pm = list(rt.get("possible_moves") or [])
        draw_panel(layout, start, goal, pos, None, pm, args.out_dir / "t_junction_panel.png")
    if rd:
        pos = tuple(rd.get("position") or (start[0], start[1]))
        action = rd.get("action")
        pm = list(rd.get("possible_moves") or [])
        draw_panel(layout, start, goal, pos, None, pm, args.out_dir / "dead_end_panel.png")

    print(f"Wrote panels under {args.out_dir}")


if __name__ == "__main__":
    main()
