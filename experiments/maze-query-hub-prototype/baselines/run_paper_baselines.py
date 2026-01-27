#!/usr/bin/env python3
"""
Run lightweight, reproducible maze baselines for paper tables.

Baselines implemented here are intentionally dependency-light and use the same
maze generator as the Query-Hub runs (seeded DFS maze via SimpleMaze).

Outputs a compact JSON with:
  - per-baseline summary (success rate, Wilson 95% CI, step stats)
  - per-seed outcomes (for paired tests like McNemar)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics as st
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from insightspike.environments.maze import SimpleMaze


def wilson_ci95(k: int, n: int) -> Tuple[float, float]:
    """Wilson score interval (95%)."""
    if n <= 0:
        return 0.0, 0.0
    z = 1.959963984540054  # scipy.stats.norm.ppf(0.975)
    phat = k / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2.0 * n)) / denom
    half = (z * math.sqrt((phat * (1.0 - phat) + (z * z) / (4.0 * n)) / n)) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return lo, hi


def _action_to(frm: Tuple[int, int], to: Tuple[int, int]) -> int:
    dy = int(to[0] - frm[0])
    dx = int(to[1] - frm[1])
    for a, (ady, adx) in SimpleMaze.ACTIONS.items():
        if (int(ady), int(adx)) == (dy, dx):
            return int(a)
    raise ValueError("No valid action for given delta")


def run_random_walk(
    *,
    maze_size: int,
    maze_type: str,
    max_steps: int,
    seeds: int,
    seed_start: int,
    anti_backtrack: bool,
) -> List[Dict]:
    """Randomly sample from passable moves; optionally avoid immediate backtrack."""
    out: List[Dict] = []
    prev_deltas: Dict[int, Tuple[int, int] | None] = {}
    for seed in range(seed_start, seed_start + seeds):
        random.seed(seed)
        np.random.seed(seed)
        env = SimpleMaze(size=(maze_size, maze_size), maze_type=maze_type)
        obs = env.reset()
        prev_deltas[seed] = None
        for _ in range(max_steps):
            if obs.is_goal:
                break
            moves = list(obs.possible_moves)
            pd = prev_deltas[seed]
            if anti_backtrack and pd is not None and len(moves) > 1:
                opp = (-int(pd[0]), -int(pd[1]))
                filtered = [a for a in moves if SimpleMaze.ACTIONS[a] != opp]
                if filtered:
                    moves = filtered
            a = random.choice(moves)
            dy, dx = SimpleMaze.ACTIONS[a]
            obs, _, done, _ = env.step(int(a))
            prev_deltas[seed] = (int(dy), int(dx))
            if done:
                break
        success = bool(obs.is_goal)
        out.append(
            {
                "seed": seed,
                "success": success,
                "steps": int(env.steps if success else max_steps),
            }
        )
    return out


def run_greedy_dfs(
    *,
    maze_size: int,
    maze_type: str,
    max_steps: int,
    seeds: int,
    seed_start: int,
) -> List[Dict]:
    """Online DFS with visited-set + stack; choose the smallest action-id among unvisited neighbors."""
    out: List[Dict] = []
    for seed in range(seed_start, seed_start + seeds):
        random.seed(seed)
        np.random.seed(seed)
        env = SimpleMaze(size=(maze_size, maze_size), maze_type=maze_type)
        obs = env.reset()
        start = obs.position
        visited = {start}
        stack: List[Tuple[int, int]] = [start]
        for _ in range(max_steps):
            if obs.is_goal:
                break
            cur = obs.position
            # Expand to an unvisited neighbor if exists
            candidates: List[Tuple[int, Tuple[int, int]]] = []
            for a in obs.possible_moves:
                dy, dx = SimpleMaze.ACTIONS[a]
                nb = (cur[0] + dy, cur[1] + dx)
                if nb in visited:
                    continue
                candidates.append((int(a), nb))
            if candidates:
                candidates.sort(key=lambda x: x[0])
                a, nb = candidates[0]
                obs, _, done, _ = env.step(a)
                visited.add(nb)
                stack.append(nb)
                if done:
                    break
                continue
            # Otherwise backtrack
            if len(stack) <= 1:
                break
            cur = stack.pop()
            prev = stack[-1]
            a = _action_to(cur, prev)
            obs, _, done, _ = env.step(a)
            if done:
                break
        success = bool(obs.is_goal)
        out.append(
            {
                "seed": seed,
                "success": success,
                "steps": int(env.steps if success else max_steps),
            }
        )
    return out


def summarize_runs(runs: List[Dict], *, max_steps: int) -> Dict:
    succ = [bool(r["success"]) for r in runs]
    steps = [int(r["steps"]) for r in runs]
    n = len(runs) or 1
    k = sum(1 for ok in succ if ok)
    lo, hi = wilson_ci95(k, n)
    half = (hi - lo) / 2.0
    steps_succ = [s for s, ok in zip(steps, succ) if ok]
    return {
        "n": int(n),
        "k_success": int(k),
        "success_rate": float(k / n),
        "success_ci95_wilson": [float(lo), float(hi)],
        "success_ci95_half": float(half),
        # Steps: report both overall (failures are clamped to max_steps) and success-only stats.
        "mean_steps_all": float(st.mean(steps)) if steps else float(max_steps),
        "mean_steps_success": float(st.mean(steps_succ)) if steps_succ else float(max_steps),
        "std_steps_success": float(st.stdev(steps_succ)) if len(steps_succ) > 1 else 0.0,
        "n_success_steps": int(len(steps_succ)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run paper baselines (maze)")
    ap.add_argument("--maze-size", type=int, default=15)
    ap.add_argument("--maze-type", type=str, default="dfs")
    ap.add_argument("--max-steps", type=int, default=250)
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--rw-anti-backtrack", action="store_true", help="Avoid immediate backtrack in Random Walk (uses 1-step memory).")
    ap.add_argument("--out-json", type=Path, default=Path("docs/paper/data/maze_15x15_baselines_s250.json"))
    args = ap.parse_args()

    runs_rw = run_random_walk(
        maze_size=args.maze_size,
        maze_type=args.maze_type,
        max_steps=args.max_steps,
        seeds=args.seeds,
        seed_start=args.seed_start,
        anti_backtrack=bool(args.rw_anti_backtrack),
    )
    runs_dfs = run_greedy_dfs(
        maze_size=args.maze_size,
        maze_type=args.maze_type,
        max_steps=args.max_steps,
        seeds=args.seeds,
        seed_start=args.seed_start,
    )

    out = {
        "config": {
            "maze_size": int(args.maze_size),
            "maze_type": str(args.maze_type),
            "max_steps": int(args.max_steps),
            "seeds": int(args.seeds),
            "seed_start": int(args.seed_start),
            "random_walk": {
                "avoid_walls": True,
                "anti_backtrack": bool(args.rw_anti_backtrack),
                "policy": "uniform_over_possible_moves",
            },
            "greedy_dfs": {
                "neighbor_choice": "min_action_id_unvisited",
                "visited_memory": True,
            },
        },
        "baselines": {
            "random_walk": summarize_runs(runs_rw, max_steps=args.max_steps),
            "greedy_dfs": summarize_runs(runs_dfs, max_steps=args.max_steps),
        },
        "runs": {
            "random_walk": runs_rw,
            "greedy_dfs": runs_dfs,
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out["baselines"], ensure_ascii=False, indent=2))
    print(f"[done] wrote {args.out_json}")


if __name__ == "__main__":
    main()

