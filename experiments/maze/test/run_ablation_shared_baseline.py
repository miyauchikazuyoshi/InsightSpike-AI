#!/usr/bin/env python3
"""Day 3 ablation: shared baseline warmup, then B(legacy) vs C(threelayer) eval.

Flow per seed:
  1. Warmup (extended + legacy) → graph
  2. Sleep optimize → optimized_graph
  3. Eval B: run_episode_query(search_mode=legacy, inherited_graph=optimized_graph)
  4. Eval C: run_episode_query(search_mode=threelayer, inherited_graph=optimized_graph)

Same optimized_graph used for both B and C evals.
"""

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import networkx as nx

# Ensure imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from run_experiment_query import run_episode_query, EpisodeArtifacts
from graph_persistent_dg.sleep_propagate import sleep_optimize


def build_config(args, search_mode: str):
    """Build QueryHubConfig with specified search_mode."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from qhlib.models import QueryHubConfig

    selector_params = {
        "theta_cand": 0.3,
        "theta_link": 0.8,
        "candidate_cap": 32,
        "top_m": 8,
        "cand_radius": 1.0,
        "link_radius": 0.05,
    }
    gedig_params = {
        "lambda_weight": 1.0,
        "max_hops": args.max_hops,
        "decay_factor": 0.7,
        "adaptive_hops": True,
        "sp_beta": 0.2,
        "sp_scope_mode": "auto",
        "sp_hop_expand": 3,
        "sp_boundary_mode": "",
        "ig_hop_apply": "all",
    }
    return QueryHubConfig(
        maze_size=args.maze_size,
        maze_type="dfs_loops",
        max_steps=args.max_steps,
        selector=selector_params,
        gedig=gedig_params,
        linkset_mode=False,
        linkset_base="all",
        theta_ag=args.theta_ag,
        theta_dg=0.0,
        top_link=4,
        commit_budget=8,
        commit_from="ecand",
        norm_base="cand",
        action_policy="softmax",
        action_temp=0.5,
        action_source="obs",
        anti_backtrack=True,
        anchor_recent_q=0,
        sp_cache=True,
        sp_cache_mode="core",
        sp_cand_topk=args.sp_cand_topk,
        sp_pair_samples=400,
        eval_all_hops=False,
        ged_hop0_const=True,
        gh_mode="query_hub",
        pre_eval=False,
        snapshot_mode="obs",
        timeline_to_graph=True,
        vector_mode="extended",
        propagated_alpha=1.0,
        sleep_propagate_gamma=args.sleep_gamma,
        sleep_propagate_iters=args.sleep_iters,
        sp_mode="both",
        steps_ultra_light=True,
        search_mode=search_mode,
        theta_attention=0.3,
        attention_decay=0.95,
        attention_boost=0.1,
        attention_alpha=0.5,
        min_layer1_candidates=2,
        dg_gate_tau=args.dg_gate_tau,
    )


def run_one_seed(seed: int, args) -> dict:
    """Run warmup → sleep → eval(B) + eval(C) for one seed."""
    print(f"  [warmup] seed={seed}, steps={args.warmup_steps}...")
    t0 = time.time()

    # Warmup config: always legacy search
    warmup_cfg = build_config(args, search_mode="legacy")
    warmup_cfg.max_steps = args.warmup_steps
    warmup_arts = run_episode_query(seed=seed, config=warmup_cfg)
    warmup_time = time.time() - t0
    warmup_graph = warmup_arts.graph

    print(f"  [warmup] done in {warmup_time:.1f}s  "
          f"nodes={warmup_graph.number_of_nodes()} edges={warmup_graph.number_of_edges()} "
          f"success={warmup_arts.summary.get('success')}")

    # Sleep optimize
    print(f"  [sleep] propagating rewards...")
    optimized_graph = sleep_optimize(
        warmup_graph,
        gamma=args.sleep_gamma,
        n_iters=args.sleep_iters,
    )

    results = {
        "seed": seed,
        "warmup": {
            "steps": int(warmup_arts.summary.get("steps", 0)),
            "success": bool(warmup_arts.summary.get("success", False)),
            "nodes": warmup_graph.number_of_nodes(),
            "edges": warmup_graph.number_of_edges(),
            "time_s": round(warmup_time, 1),
        },
    }

    # Eval B: legacy search with shared graph
    print(f"  [eval-B] legacy search...")
    t1 = time.time()
    eval_cfg_b = build_config(args, search_mode="legacy")
    eval_arts_b = run_episode_query(
        seed=seed, config=eval_cfg_b,
        inherited_graph=copy.deepcopy(optimized_graph),
    )
    eval_time_b = time.time() - t1
    print(f"  [eval-B] done in {eval_time_b:.1f}s  "
          f"steps={eval_arts_b.summary.get('steps')} success={eval_arts_b.summary.get('success')}")

    results["eval_B"] = {
        "steps": int(eval_arts_b.summary.get("steps", 0)),
        "success": bool(eval_arts_b.summary.get("success", False)),
        "edges": int(eval_arts_b.summary.get("edges", 0)),
        "time_s": round(eval_time_b, 1),
        "betti1_final": eval_arts_b.summary.get("betti1_series", [0])[-1],
        "summary": dict(eval_arts_b.summary),
    }

    # Eval C: threelayer search with same shared graph
    print(f"  [eval-C] threelayer search...")
    t2 = time.time()
    eval_cfg_c = build_config(args, search_mode="threelayer")
    eval_arts_c = run_episode_query(
        seed=seed, config=eval_cfg_c,
        inherited_graph=copy.deepcopy(optimized_graph),
    )
    eval_time_c = time.time() - t2
    print(f"  [eval-C] done in {eval_time_c:.1f}s  "
          f"steps={eval_arts_c.summary.get('steps')} success={eval_arts_c.summary.get('success')}")

    results["eval_C"] = {
        "steps": int(eval_arts_c.summary.get("steps", 0)),
        "success": bool(eval_arts_c.summary.get("success", False)),
        "edges": int(eval_arts_c.summary.get("edges", 0)),
        "time_s": round(eval_time_c, 1),
        "betti1_final": eval_arts_c.summary.get("betti1_series", [0])[-1],
        "summary": dict(eval_arts_c.summary),
    }

    return results


def print_report(all_results: list):
    """Print comparison table."""
    print("\n" + "=" * 60)
    print("Shared-Baseline Ablation Results")
    print("=" * 60)

    print(f"\n{'seed':>4} | {'warmup':^20} | {'B (legacy)':^20} | {'C (threelayer)':^20}")
    print(f"{'':>4} | {'steps':>6} {'succ':>5} {'nodes':>6} | {'steps':>6} {'succ':>5} {'β₁':>6} | {'steps':>6} {'succ':>5} {'β₁':>6}")
    print("-" * 80)

    for r in all_results:
        w = r["warmup"]
        b = r["eval_B"]
        c = r["eval_C"]
        print(f"{r['seed']:>4} | {w['steps']:>6} {str(w['success']):>5} {w['nodes']:>6} "
              f"| {b['steps']:>6} {str(b['success']):>5} {b['betti1_final']:>6.0f} "
              f"| {c['steps']:>6} {str(c['success']):>5} {c['betti1_final']:>6.0f}")

    # Averages
    n = len(all_results)
    if n > 0:
        avg_b_steps = sum(r["eval_B"]["steps"] for r in all_results) / n
        avg_c_steps = sum(r["eval_C"]["steps"] for r in all_results) / n
        b_succ = sum(1 for r in all_results if r["eval_B"]["success"]) / n
        c_succ = sum(1 for r in all_results if r["eval_C"]["success"]) / n
        avg_b_b1 = sum(r["eval_B"]["betti1_final"] for r in all_results) / n
        avg_c_b1 = sum(r["eval_C"]["betti1_final"] for r in all_results) / n

        print("-" * 80)
        print(f" avg | {'':>20} "
              f"| {avg_b_steps:>6.1f} {b_succ:>5.0%} {avg_b_b1:>6.1f} "
              f"| {avg_c_steps:>6.1f} {c_succ:>5.0%} {avg_c_b1:>6.1f}")

        delta_steps = avg_c_steps - avg_b_steps
        print(f"\nΔ(C-B): steps={delta_steps:+.1f}  success={c_succ-b_succ:+.0%}")


def main():
    parser = argparse.ArgumentParser(description="Shared-baseline ablation: B vs C")
    parser.add_argument("--maze-size", type=int, default=25)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--max-hops", type=int, default=10)
    parser.add_argument("--sp-cand-topk", type=int, default=5)
    parser.add_argument("--theta-ag", type=float, default=0.4)
    parser.add_argument("--sleep-gamma", type=float, default=0.95)
    parser.add_argument("--sleep-iters", type=int, default=50)
    parser.add_argument("--dg-gate-tau", type=float, default=1.0)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print(f"=== Shared-Baseline Ablation ===")
    print(f"maze={args.maze_size}x{args.maze_size} warmup={args.warmup_steps} eval={args.max_steps} seeds={args.seeds}")
    print()

    all_results = []
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        print(f"--- seed {seed} ---")
        result = run_one_seed(seed, args)
        all_results.append(result)
        print()

    print_report(all_results)

    # Save JSON
    out_path = args.output or f"results/ablation_shared_{args.maze_size}x{args.maze_size}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"config": vars(args), "results": all_results}, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
