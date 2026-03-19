#!/usr/bin/env python3
"""Minimal maze runner using β₁-based geDIG evaluator.

This runner uses the existing SimpleMaze environment but replaces
the SP-based evaluator with β₁-based evaluation.

Usage:
  .venv/bin/python3 experiments/maze_b1/run_b1_experiment.py \
    --size 15 --seeds 5 --steps 200

  .venv/bin/python3 experiments/maze_b1/run_b1_experiment.py \
    --size 25 --seeds 60 --steps 500 --output results/25x25_60seeds
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from collections import defaultdict

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "qhlib"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "maze"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from insightspike.environments.maze import SimpleMaze
from evaluator_b1 import compute_betti_1, delta_betti_1, evaluate_multihop_b1, EvalResult


def maze_to_graph(maze: SimpleMaze) -> nx.Graph:
    """Convert maze grid to networkx graph (passable cells as nodes)."""
    g = nx.Graph()
    for r in range(maze.height):
        for c in range(maze.width):
            if maze.grid[r, c] == 0:  # passable
                g.add_node((r, c))
                # Connect to adjacent passable cells
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < maze.height and 0 <= nc < maze.width:
                        if maze.grid[nr, nc] == 0:
                            g.add_edge((r, c), (nr, nc))
    return g


def build_observation_graph(
    maze_graph: nx.Graph,
    agent_pos: tuple,
    visited: set,
    observation_radius: int = 2,
) -> nx.Graph:
    """Build agent's current observation graph (known portion of maze)."""
    g = nx.Graph()
    # Add visited cells and their edges
    for node in visited:
        if node in maze_graph:
            g.add_node(node)
    # Add observed cells within radius
    for r in range(agent_pos[0] - observation_radius, agent_pos[0] + observation_radius + 1):
        for c in range(agent_pos[1] - observation_radius, agent_pos[1] + observation_radius + 1):
            if (r, c) in maze_graph:
                g.add_node((r, c))
    # Add edges between known nodes
    for u in g.nodes():
        for v in maze_graph.neighbors(u):
            if v in g.nodes():
                g.add_edge(u, v)
    return g


def get_candidate_edges(
    maze_graph: nx.Graph,
    known_graph: nx.Graph,
    agent_pos: tuple,
    radius: int = 3,
) -> list:
    """Get candidate edges at frontier of knowledge."""
    candidates = []
    for node in known_graph.nodes():
        for nbr in maze_graph.neighbors(node):
            if not known_graph.has_edge(node, nbr):
                # This is a frontier edge
                dist = abs(node[0] - agent_pos[0]) + abs(node[1] - agent_pos[1])
                if dist <= radius:
                    candidates.append((node, nbr, {"weight": 1.0}))
    return candidates[:32]  # cap candidates


def choose_action_gedig(
    maze: SimpleMaze,
    maze_graph: nx.Graph,
    agent_pos: tuple,
    visited: set,
    prev_known: nx.Graph,
    lambda_weight: float = 1.0,
    gamma: float = 0.5,
    max_hops: int = 3,
    obs_radius: int = 2,
) -> tuple:
    """Choose next action using β₁-based geDIG evaluation.

    Returns (action, eval_result, new_known_graph).
    """
    # Build current known graph
    current_known = build_observation_graph(maze_graph, agent_pos, visited, obs_radius)

    # Get candidate edges
    ecand = get_candidate_edges(maze_graph, current_known, agent_pos)

    # Evaluate with β₁
    result = evaluate_multihop_b1(
        lambda_weight=lambda_weight,
        gamma=gamma,
        prev_graph=prev_known,
        stage_graph=current_known,
        g_before_for_expansion=prev_known,
        anchors_core={agent_pos},
        anchors_top_before=set(prev_known.nodes()),
        anchors_top_after=set(current_known.nodes()),
        ecand=ecand,
        base_ig=0.1,
        denom_cmax_base=max(current_known.number_of_edges(), 1),
        max_hops=max_hops,
    )

    # Choose action: move toward the direction with lowest g (most promising)
    # Simple heuristic: prefer unexplored neighbors, then use gmin direction
    actions = list(SimpleMaze.ACTIONS.keys())
    random.shuffle(actions)

    best_action = None
    best_score = float("inf")

    for action in actions:
        dr, dc = SimpleMaze.ACTIONS[action]
        nr, nc = agent_pos[0] + dr, agent_pos[1] + dc

        # Check if valid move
        if not (0 <= nr < maze.height and 0 <= nc < maze.width):
            continue
        if maze.grid[nr, nc] != 0:  # wall
            continue

        # Score: prefer unvisited, then use graph structure
        if (nr, nc) not in visited:
            score = -1.0  # strongly prefer unvisited
        else:
            # Use β₁ of local subgraph as tiebreaker
            local = current_known.subgraph(
                set(current_known.neighbors((nr, nc))) | {(nr, nc)}
            )
            score = -compute_betti_1(local) * 0.1  # prefer cyclic structure

        if score < best_score:
            best_score = score
            best_action = action

    if best_action is None:
        best_action = random.choice(actions)

    return best_action, result, current_known


def run_single_seed(
    maze_size: int,
    seed: int,
    max_steps: int,
    lambda_weight: float = 1.0,
    gamma: float = 0.5,
    max_hops: int = 3,
    obs_radius: int = 2,
    verbose: bool = False,
) -> dict:
    """Run a single maze experiment with given seed."""
    random.seed(seed)
    np.random.seed(seed)

    maze = SimpleMaze(size=(maze_size, maze_size), maze_type='dfs_loops')
    maze_graph = maze_to_graph(maze)

    agent_pos = maze.start_pos if hasattr(maze, 'start_pos') else (1, 1)
    goal_pos = maze.goal_pos if hasattr(maze, 'goal_pos') else (maze_size - 2, maze_size - 2)

    visited = {agent_pos}
    prev_known = build_observation_graph(maze_graph, agent_pos, visited, obs_radius)

    goal_reached = False
    steps_to_goal = max_steps
    b1_history = []
    g_history = []

    t0 = time.time()

    for step in range(max_steps):
        action, result, current_known = choose_action_gedig(
            maze, maze_graph, agent_pos, visited, prev_known,
            lambda_weight=lambda_weight, gamma=gamma,
            max_hops=max_hops, obs_radius=obs_radius,
        )

        # Execute action
        dr, dc = SimpleMaze.ACTIONS[action]
        nr, nc = agent_pos[0] + dr, agent_pos[1] + dc
        if 0 <= nr < maze.height and 0 <= nc < maze.width and maze.grid[nr, nc] == 0:
            agent_pos = (nr, nc)
            visited.add(agent_pos)

        # Track metrics
        b1_history.append(result.delta_b1)
        g_history.append(result.g0)

        prev_known = current_known

        # Check goal
        if agent_pos == goal_pos:
            goal_reached = True
            steps_to_goal = step + 1
            break

        if verbose and (step + 1) % 50 == 0:
            b1_now = compute_betti_1(current_known)
            print(f"    step {step+1}: pos={agent_pos} visited={len(visited)} "
                  f"β₁={b1_now} g0={result.g0:.3f}", flush=True)

    elapsed = time.time() - t0

    return {
        "seed": seed,
        "maze_size": maze_size,
        "max_steps": max_steps,
        "goal_reached": goal_reached,
        "steps_to_goal": steps_to_goal,
        "n_visited": len(visited),
        "n_passable": maze_graph.number_of_nodes(),
        "coverage": len(visited) / max(maze_graph.number_of_nodes(), 1),
        "elapsed_s": elapsed,
        "mean_b1": float(np.mean(b1_history)) if b1_history else 0.0,
        "mean_g0": float(np.mean(g_history)) if g_history else 0.0,
        "lambda": lambda_weight,
        "gamma": gamma,
    }


def main():
    parser = argparse.ArgumentParser(description="β₁-based maze experiment")
    parser.add_argument("--size", type=int, default=15)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lambda", type=float, default=1.0, dest="lambda_weight")
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--max-hops", type=int, default=3)
    parser.add_argument("--obs-radius", type=int, default=2)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(f"=== β₁-based Maze Experiment ===", flush=True)
    print(f"  Size: {args.size}x{args.size}", flush=True)
    print(f"  Seeds: {args.seeds} (start={args.seed_start})", flush=True)
    print(f"  Steps: {args.steps}", flush=True)
    print(f"  F = ΔGED - {args.lambda_weight}·(ΔH + {args.gamma}·Δβ₁)", flush=True)
    print(flush=True)

    results = []
    for i in range(args.seeds):
        seed = args.seed_start + i
        print(f"  [{i+1}/{args.seeds}] seed={seed} ...", end=" ", flush=True)
        r = run_single_seed(
            maze_size=args.size,
            seed=seed,
            max_steps=args.steps,
            lambda_weight=args.lambda_weight,
            gamma=args.gamma,
            max_hops=args.max_hops,
            obs_radius=args.obs_radius,
            verbose=args.verbose,
        )
        results.append(r)
        status = "✅ GOAL" if r["goal_reached"] else "❌ timeout"
        print(f"{status} steps={r['steps_to_goal']} "
              f"coverage={r['coverage']:.1%} "
              f"β₁={r['mean_b1']:.2f} "
              f"({r['elapsed_s']:.1f}s)", flush=True)

    # Summary
    n_goal = sum(1 for r in results if r["goal_reached"])
    avg_steps = np.mean([r["steps_to_goal"] for r in results if r["goal_reached"]]) if n_goal > 0 else float("inf")
    avg_coverage = np.mean([r["coverage"] for r in results])

    print(flush=True)
    print(f"=== Summary ===", flush=True)
    print(f"  Goal reached: {n_goal}/{args.seeds} ({n_goal/args.seeds:.0%})", flush=True)
    print(f"  Avg steps to goal: {avg_steps:.1f}", flush=True)
    print(f"  Avg coverage: {avg_coverage:.1%}", flush=True)

    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "results.json", "w") as f:
            json.dump({
                "config": {
                    "size": args.size,
                    "seeds": args.seeds,
                    "steps": args.steps,
                    "lambda": args.lambda_weight,
                    "gamma": args.gamma,
                    "formula": "F = ΔGED - λ(ΔH + γΔβ₁)",
                },
                "summary": {
                    "goal_rate": n_goal / args.seeds,
                    "n_goal": n_goal,
                    "avg_steps_to_goal": avg_steps if n_goal > 0 else None,
                    "avg_coverage": avg_coverage,
                },
                "per_seed": results,
            }, f, indent=2)
        print(f"\nSaved to {out_dir}/results.json", flush=True)


if __name__ == "__main__":
    main()
