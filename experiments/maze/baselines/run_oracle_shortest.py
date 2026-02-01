#!/usr/bin/env python3
"""
Oracle shortest-path baseline (Dijkstra/BFS on full maze grid).

Given a generated DFS maze (seeded), compute the shortest path from start to goal
with full knowledge (omniscient). This serves as an oracle/baseline for steps and
success rate, independent of Query-Hub/L3.

Outputs a compact JSON with success_rate and avg_steps (baseline_*) across seeds.

Example:
  PYTHONPATH=src python experiments/maze-query-hub-prototype/baselines/run_oracle_shortest.py \
    --maze-size 51 --seeds 40 --seed-start 0 \
    --out-json docs/paper/data/oracle_51x51_s250.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np

from insightspike.environments.proper_maze_generator import ProperMazeGenerator


def build_graph(maze: np.ndarray) -> nx.Graph:
    h, w = maze.shape
    g = nx.Graph()
    def passable(r: int, c: int) -> bool:
        if 0 <= r < h and 0 <= c < w:
            return maze[r, c] == 0
        return False
    for r in range(h):
        for c in range(w):
            if not passable(r, c):
                continue
            g.add_node((r, c))
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if passable(nr, nc):
                    g.add_edge((r, c), (nr, nc), weight=1.0)
    return g


def shortest_length(
    g: nx.Graph,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    *,
    method: str = "bfs",
) -> int | None:
    try:
        m = str(method).lower()
        if m == "bfs":
            return nx.shortest_path_length(g, source=start, target=goal)
        elif m == "dijkstra":
            return int(nx.dijkstra_path_length(g, source=start, target=goal, weight="weight"))
        elif m == "astar":
            # Manhattan heuristic on grid coordinates
            def h(u: Tuple[int, int], v: Tuple[int, int]) -> float:
                return float(abs(u[0] - v[0]) + abs(u[1] - v[1]))
            return int(nx.astar_path_length(g, source=start, target=goal, heuristic=h, weight="weight"))
        else:
            return nx.shortest_path_length(g, source=start, target=goal)
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Oracle shortest-path baseline for maze")
    ap.add_argument("--maze-size", type=int, default=15)
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument(
        "--method",
        type=str,
        default="bfs",
        choices=["bfs", "dijkstra", "astar"],
        help="Shortest-path method: bfs (unweighted), dijkstra, or astar (Manhattan)",
    )
    args = ap.parse_args()

    size = (args.maze_size, args.maze_size)
    lengths: List[int] = []
    success = 0
    for i in range(args.seeds):
        seed = args.seed_start + i
        maze = ProperMazeGenerator.generate_dfs_maze(size, seed=seed)
        start = (1, 1)
        goal = (size[0]-2, size[1]-2)
        g = build_graph(maze)
        L = shortest_length(g, start, goal, method=args.method)
        if L is not None:
            success += 1
            lengths.append(int(L))

    out: Dict[str, Any] = {
        "maze_size": args.maze_size,
        "seeds": args.seeds,
        "seed_start": args.seed_start,
        "baseline_success_rate": float(success / (args.seeds or 1)),
        "baseline_avg_steps": (float(sum(lengths) / len(lengths)) if lengths else 0.0),
        "method": str(args.method),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
