#!/usr/bin/env python3
"""Day 3 ablation analysis: compare conditions A/B/C.

Usage:
    python analyze_ablation_day3.py results/ablation_day3/
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any


def load_condition(results_dir: Path, cond: str, n_seeds: int) -> Dict[str, List]:
    """Load summary and step-log data for a condition across seeds."""
    summaries = []
    step_logs = []
    for seed in range(n_seeds):
        summary_path = results_dir / f"cond{cond}_seed{seed}.json"
        step_path = results_dir / f"cond{cond}_seed{seed}_steps.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summaries.append(json.load(f))
        if step_path.exists():
            with open(step_path) as f:
                step_logs.append(json.load(f))
    return {"summaries": summaries, "step_logs": step_logs}


def analyze_summary(data: Dict) -> Dict[str, Any]:
    """Extract key metrics from summaries."""
    summaries = data["summaries"]
    if not summaries:
        return {}

    results = []
    for s in summaries:
        summary = s.get("summary", {})
        runs = s.get("runs", [])
        for run in runs:
            results.append({
                "success": bool(run.get("success", False)),
                "steps": int(run.get("steps", 0)),
                "edges": int(run.get("edges", 0)),
                "dead_end_escape_rate": float(run.get("dead_end_escape_rate", 0.0)),
            })

    n = len(results)
    if n == 0:
        return {}

    success_count = sum(1 for r in results if r["success"])
    avg_steps = sum(r["steps"] for r in results) / n
    avg_edges = sum(r["edges"] for r in results) / n
    avg_de_escape = sum(r["dead_end_escape_rate"] for r in results) / n

    return {
        "n_runs": n,
        "success_rate": success_count / n,
        "avg_steps": avg_steps,
        "avg_edges": avg_edges,
        "avg_dead_end_escape_rate": avg_de_escape,
    }


def analyze_steps(data: Dict) -> Dict[str, Any]:
    """Extract step-level metrics (search layer, betti1, AG/DG fire)."""
    step_logs = data["step_logs"]
    if not step_logs:
        return {}

    all_steps = []
    for log in step_logs:
        # step-log is typically {"seed_X": [step_records...]}
        if isinstance(log, dict):
            for key, steps in log.items():
                if isinstance(steps, list):
                    all_steps.extend(steps)
        elif isinstance(log, list):
            all_steps.extend(log)

    n = len(all_steps)
    if n == 0:
        return {}

    # Search layer distribution
    layer_counts = {-1: 0, 0: 0, 1: 0, 2: 0}
    search_times = []
    l1_candidates = []
    ag_fires = 0
    dg_fires = 0

    for s in all_steps:
        layer = int(s.get("search_layer_used", -1))
        layer_counts[layer] = layer_counts.get(layer, 0) + 1
        search_times.append(float(s.get("search_time_ms", 0.0)))
        l1_candidates.append(int(s.get("search_l1_candidates", 0)))
        if s.get("ag_fire", False):
            ag_fires += 1
        if s.get("dg_fire", False):
            dg_fires += 1

    result = {
        "total_steps": n,
        "search_layer_dist": {f"L{k}" if k >= 0 else "none": v for k, v in sorted(layer_counts.items())},
        "l1_skip_rate": (layer_counts.get(0, 0) + layer_counts.get(1, 0)) / n if n else 0,
        "avg_search_time_ms": sum(search_times) / n if n else 0,
        "avg_l1_candidates": sum(l1_candidates) / n if n else 0,
        "ag_fire_rate": ag_fires / n if n else 0,
        "dg_fire_rate": dg_fires / n if n else 0,
    }
    return result


def analyze_betti1(data: Dict) -> Dict[str, Any]:
    """Extract betti1 trajectory from summaries."""
    summaries = data["summaries"]
    if not summaries:
        return {}

    all_betti1 = []
    all_node_count = []
    all_edge_count = []
    for s in summaries:
        runs = s.get("runs", [])
        for run in runs:
            b1 = run.get("betti1_series", [])
            nc = run.get("node_count_series", [])
            ec = run.get("edge_count_series", [])
            if b1:
                all_betti1.append(b1)
            if nc:
                all_node_count.append(nc)
            if ec:
                all_edge_count.append(ec)

    if not all_betti1:
        return {}

    # Summary stats from final values
    final_b1 = [series[-1] for series in all_betti1 if series]
    final_nc = [series[-1] for series in all_node_count if series]
    final_ec = [series[-1] for series in all_edge_count if series]

    return {
        "avg_final_betti1": sum(final_b1) / len(final_b1) if final_b1 else 0,
        "avg_final_nodes": sum(final_nc) / len(final_nc) if final_nc else 0,
        "avg_final_edges": sum(final_ec) / len(final_ec) if final_ec else 0,
    }


def print_comparison(conditions: Dict[str, Dict]):
    """Print formatted comparison table."""
    labels = {"A": "baseline (standard, legacy)", "B": "DG-only (extended, legacy)", "C": "DG+threelayer (extended, threelayer)"}

    print("=" * 70)
    print("Day 3 Ablation Results")
    print("=" * 70)

    # Summary metrics
    print("\n### Episode Metrics")
    header = f"{'Metric':<30} {'A (baseline)':>12} {'B (DG-only)':>12} {'C (DG+3L)':>12}"
    print(header)
    print("-" * len(header))

    metrics = ["success_rate", "avg_steps", "avg_edges", "avg_dead_end_escape_rate"]
    fmt = {"success_rate": ".1%", "avg_steps": ".1f", "avg_edges": ".1f", "avg_dead_end_escape_rate": ".2f"}

    for m in metrics:
        vals = []
        for c in ["A", "B", "C"]:
            v = conditions[c].get("summary", {}).get(m, "-")
            if isinstance(v, (int, float)):
                vals.append(format(v, fmt.get(m, ".2f")))
            else:
                vals.append(str(v))
        print(f"{m:<30} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12}")

    # Step-level metrics
    print("\n### Step-Level Metrics")
    step_metrics = ["total_steps", "l1_skip_rate", "avg_search_time_ms", "avg_l1_candidates", "ag_fire_rate", "dg_fire_rate"]
    sfmt = {"l1_skip_rate": ".1%", "avg_search_time_ms": ".3f", "avg_l1_candidates": ".1f", "ag_fire_rate": ".1%", "dg_fire_rate": ".1%", "total_steps": ".0f"}

    header2 = f"{'Metric':<30} {'A (baseline)':>12} {'B (DG-only)':>12} {'C (DG+3L)':>12}"
    print(header2)
    print("-" * len(header2))

    for m in step_metrics:
        vals = []
        for c in ["A", "B", "C"]:
            v = conditions[c].get("steps", {}).get(m, "-")
            if isinstance(v, (int, float)):
                vals.append(format(v, sfmt.get(m, ".2f")))
            else:
                vals.append(str(v))
        print(f"{m:<30} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12}")

    # Search layer distribution (only for C)
    c_steps = conditions["C"].get("steps", {})
    layer_dist = c_steps.get("search_layer_dist", {})
    if layer_dist:
        print(f"\n### Search Layer Distribution (Condition C)")
        for layer, count in sorted(layer_dist.items()):
            total = c_steps.get("total_steps", 1)
            pct = count / total * 100 if total else 0
            print(f"  {layer}: {count} ({pct:.1f}%)")

    # Betti1 metrics
    print("\n### Betti1 (Graph Topology)")
    b1_metrics = ["avg_final_betti1", "avg_final_nodes", "avg_final_edges"]
    header3 = f"{'Metric':<30} {'A (baseline)':>12} {'B (DG-only)':>12} {'C (DG+3L)':>12}"
    print(header3)
    print("-" * len(header3))

    for m in b1_metrics:
        vals = []
        for c in ["A", "B", "C"]:
            v = conditions[c].get("betti1", {}).get(m, "-")
            if isinstance(v, (int, float)):
                vals.append(format(v, ".1f"))
            else:
                vals.append(str(v))
        print(f"{m:<30} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12}")

    # Condition descriptions
    print(f"\n### Conditions")
    for c, label in labels.items():
        print(f"  {c}: {label}")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <results_dir> [n_seeds]")
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    if not results_dir.exists():
        print(f"Error: {results_dir} does not exist")
        sys.exit(1)

    conditions = {}
    for cond in ["A", "B", "C"]:
        data = load_condition(results_dir, cond, n_seeds)
        conditions[cond] = {
            "summary": analyze_summary(data),
            "steps": analyze_steps(data),
            "betti1": analyze_betti1(data),
        }

    print_comparison(conditions)

    # Save as JSON for further processing
    report_path = results_dir / "ablation_report.json"
    with open(report_path, "w") as f:
        json.dump(conditions, f, indent=2)
    print(f"\nJSON report saved: {report_path}")


if __name__ == "__main__":
    main()
