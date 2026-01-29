#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


Pos = Tuple[int, int]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_steps(path: Path) -> List[Dict[str, Any]]:
    payload = _read_json(path)
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("steps"), list):
        return [row for row in payload["steps"] if isinstance(row, dict)]
    raise SystemExit(f"Unsupported steps file format: {path}")


def _pos_from_row(row: Dict[str, Any], key: str) -> Optional[Pos]:
    val = row.get(key)
    if isinstance(val, list) and len(val) >= 2:
        return (int(val[0]), int(val[1]))
    return None


def _pos_from_query_node(row: Dict[str, Any], key: str) -> Optional[Pos]:
    val = row.get(key)
    if isinstance(val, list) and len(val) >= 2:
        return (int(val[0]), int(val[1]))
    return None


def _get_pre_post(row: Dict[str, Any]) -> Tuple[Optional[Pos], Optional[Pos]]:
    pre = _pos_from_row(row, "position_pre") or _pos_from_query_node(row, "query_node_pre")
    post = _pos_from_row(row, "position_post") or _pos_from_query_node(row, "query_node_post")
    return pre, post


def _bfs_shortest_len(adj: Dict[Pos, List[Pos]], start: Pos, goal: Pos) -> Optional[int]:
    if start == goal:
        return 0
    q: deque[Pos] = deque([start])
    dist: Dict[Pos, int] = {start: 0}
    while q:
        u = q.popleft()
        du = dist[u]
        for v in adj.get(u, []):
            if v in dist:
                continue
            dv = du + 1
            if v == goal:
                return dv
            dist[v] = dv
            q.append(v)
    return None


def _oracle_shortest_len(layout: List[List[int]], start: Pos, goal: Pos) -> Optional[int]:
    h = len(layout)
    w = len(layout[0]) if h else 0
    if start == goal:
        return 0

    def passable(r: int, c: int) -> bool:
        if 0 <= r < h and 0 <= c < w:
            return int(layout[r][c]) != 1
        return False

    if not passable(*start) or not passable(*goal):
        return None

    q: deque[Pos] = deque([start])
    dist: Dict[Pos, int] = {start: 0}
    while q:
        r, c = q.popleft()
        d = dist[(r, c)]
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if not passable(nr, nc):
                continue
            v = (nr, nc)
            if v in dist:
                continue
            nd = d + 1
            if v == goal:
                return nd
            dist[v] = nd
            q.append(v)
    return None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate step reduction potential via path compression from experience.")
    ap.add_argument("--summary", type=Path, required=True, help="Maze run summary JSON (output of run_experiment_query.py).")
    ap.add_argument("--steps", type=Path, required=True, help="Maze steps JSON (list of step dicts).")
    ap.add_argument("--out-json", type=Path, default=None, help="Optional output JSON path.")
    return ap.parse_args()


@dataclass
class SeedEval:
    seed: int
    success: bool
    steps: int
    oracle_len: Optional[int]
    experience_len: Optional[int]
    gain_steps: Optional[int]
    gap_to_oracle: Optional[int]
    explored_nodes: int
    explored_edges: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "success": self.success,
            "steps": self.steps,
            "oracle_len": self.oracle_len,
            "experience_len": self.experience_len,
            "gain_steps": self.gain_steps,
            "gap_to_oracle": self.gap_to_oracle,
            "explored_nodes": self.explored_nodes,
            "explored_edges": self.explored_edges,
        }


def main() -> None:
    args = parse_args()

    summary = _read_json(args.summary)
    steps = _load_steps(args.steps)

    steps_by_seed: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in steps:
        try:
            seed = int(row.get("seed", 0))
        except Exception:
            seed = 0
        steps_by_seed[seed].append(row)
    for seed in steps_by_seed:
        steps_by_seed[seed].sort(key=lambda r: int(r.get("step", 0)))

    maze_data = summary.get("maze_data") or {}
    runs = summary.get("runs") or []

    seed_evals: List[SeedEval] = []
    for run in runs:
        seed = int(run.get("seed", 0))
        success = bool(run.get("success", False))
        steps_taken = int(run.get("steps", 0))

        md = maze_data.get(str(seed)) or {}
        start = md.get("start_pos")
        goal = md.get("goal_pos")
        layout = md.get("layout")

        if not (isinstance(start, list) and len(start) >= 2 and isinstance(goal, list) and len(goal) >= 2 and isinstance(layout, list)):
            oracle_len = None
        else:
            oracle_len = _oracle_shortest_len(layout, (int(start[0]), int(start[1])), (int(goal[0]), int(goal[1])))

        adj: Dict[Pos, List[Pos]] = defaultdict(list)
        nodes: set[Pos] = set()
        edge_set: set[Tuple[Pos, Pos]] = set()
        for row in steps_by_seed.get(seed, []):
            pre, post = _get_pre_post(row)
            if pre is None or post is None:
                continue
            nodes.add(pre)
            nodes.add(post)
            moved = row.get("moved")
            if moved is None:
                moved = (pre != post)
            if not moved:
                continue
            # Maze motion is reversible; treat as undirected to estimate "post-sleep planning" potential.
            if (pre, post) not in edge_set:
                adj[pre].append(post)
                edge_set.add((pre, post))
            if (post, pre) not in edge_set:
                adj[post].append(pre)
                edge_set.add((post, pre))

        if isinstance(start, list) and len(start) >= 2 and isinstance(goal, list) and len(goal) >= 2:
            start_pos = (int(start[0]), int(start[1]))
            goal_pos = (int(goal[0]), int(goal[1]))
            experience_len = _bfs_shortest_len(adj, start_pos, goal_pos)
        else:
            experience_len = None

        if experience_len is not None:
            gain_steps = int(steps_taken - experience_len)
        else:
            gain_steps = None

        if oracle_len is not None and experience_len is not None:
            gap_to_oracle = int(experience_len - oracle_len)
        else:
            gap_to_oracle = None

        seed_evals.append(
            SeedEval(
                seed=seed,
                success=success,
                steps=steps_taken,
                oracle_len=oracle_len,
                experience_len=experience_len,
                gain_steps=gain_steps,
                gap_to_oracle=gap_to_oracle,
                explored_nodes=len(nodes),
                explored_edges=len(edge_set),
            )
        )

    out = {
        "input": {"summary": str(args.summary), "steps": str(args.steps)},
        "seeds": [ev.to_dict() for ev in seed_evals],
    }
    text = json.dumps(out, indent=2, ensure_ascii=False)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

