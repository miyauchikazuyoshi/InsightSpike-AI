#!/usr/bin/env python3
"""Lightweight calibration of (k, τ) for geDIG-RAG v3.

Uses ParameterizedGeDIGSystem (from run_parameter_sweep) to evaluate a small
set of queries, optimizing:
  score = acceptance_rate * (avg_novelty)

Writes best params to results/calibration/calibration.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple, List

from run_parameter_sweep import (
    ParameterizedGeDIGSystem,
    create_high_quality_knowledge_base,
    create_meaningful_queries,
    ExperimentConfig,
)


def evaluate_params(k: float, threshold_base: float, n_queries: int = 50) -> Tuple[float, Dict[str, float]]:
    cfg = ExperimentConfig()
    kb = create_high_quality_knowledge_base()
    queries = create_meaningful_queries()[:n_queries]
    params = {
        'k': k,
        'node_weight': 0.1,
        'edge_weight': 0.05,
        'novelty_weight': 0.5,
        'connectivity_weight': 0.2,
        'threshold_base': threshold_base,
        'threshold_novelty_high': threshold_base - 0.2,
        'threshold_novelty_low': threshold_base + 0.1,
    }
    system = ParameterizedGeDIGSystem('gedig', cfg, params)
    system.add_initial_knowledge(kb)
    novs: List[float] = []
    for q, depth in queries:
        system.process_query(q, depth)
        # Use last computed novelty estimate if available via metadata
        if system.gedig_scores and system.ig_values:
            # novelty proxy: ig/(node+edge weights) not available, fallback to ig values
            novs.append(system.ig_values[-1])
    acceptance = system.updates_applied / max(1, len(queries))
    avg_nov = sum(novs) / max(1, len(novs))
    score = acceptance * max(0.0, avg_nov)
    return score, {
        'acceptance_rate': acceptance,
        'avg_novelty_proxy': avg_nov,
        'final_nodes': len(system.nx_graph.nodes),
        'final_edges': len(system.nx_graph.edges),
    }


def main() -> None:
    k_grid = [0.1, 0.15, 0.2, 0.3]
    tau_grid = [-0.1, -0.05, 0.0, 0.05]

    best = None
    results: Dict[str, Dict[str, float]] = {}
    for k in k_grid:
        for tau in tau_grid:
            score, metrics = evaluate_params(k, tau)
            results[f"k={k},tau={tau}"] = {'score': round(score, 4), **{k2: round(v, 4) for k2, v in metrics.items()}}
            if best is None or score > best[0]:
                best = (score, {'k': k, 'threshold_base': tau, **metrics})

    out_dir = Path(__file__).resolve().parents[1] / 'results' / 'calibration'
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'grid_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    if best is not None:
        with open(out_dir / 'calibration.json', 'w') as f:
            json.dump(best[1], f, indent=2)
        print('✅ Best:', best[1])


if __name__ == '__main__':
    main()

