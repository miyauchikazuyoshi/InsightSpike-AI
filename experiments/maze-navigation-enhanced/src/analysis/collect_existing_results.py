#!/usr/bin/env python3
"""
Aggregate existing result JSONs into a clean scaling summary without re-running heavy experiments.

Sources used (if present):
- results/metrics/complex_*/summary.json      (treat as size=15 complex)
- results/metrics/simple_*/summary.json       (sanity, not used for scaling)
- ../maze-unified-v2/results/experiment_*.json (size=11 baseline)
- results/real_gedig_test/summary_*.json     (per-run sanity)

Writes: results/scaling/scaling_summary_from_existing.json
"""

from __future__ import annotations

import os
import sys
import json
from glob import glob
from typing import Dict, Any


def load_json(path: str) -> Any:
    with open(path, 'r') as f:
        return json.load(f)


def main() -> None:
    # project base: experiments/maze-navigation-enhanced
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    res_dir = os.path.join(base, 'results')

    summary: Dict[str, Any] = {}

    # 15x15 complex metrics
    comp = glob(os.path.join(res_dir, 'metrics', 'complex_*', 'summary.json'))
    if comp:
        comp_path = sorted(comp)[-1]
        data = load_json(comp_path)
        # Map: insightspike -> geDIG, dfs -> DFS, random -> Random
        ged = data.get('insightspike', {})
        dfs = data.get('dfs', {})
        rnd = data.get('random', {})
        summary['15'] = {
            'success_rate_gedig': ged.get('success_rate'),
            'avg_steps_gedig': ged.get('avg_steps'),
            'success_rate_dfs': dfs.get('success_rate'),
            'avg_steps_dfs': dfs.get('avg_steps'),
            'success_rate_random': rnd.get('success_rate'),
            'avg_steps_random': rnd.get('avg_steps'),
            'ratio_gedig_vs_dfs': (ged.get('avg_steps') / dfs.get('avg_steps') if ged.get('avg_steps') and dfs.get('avg_steps') else None),
        }

    # 11x11 unified v2
    root = os.path.abspath(os.path.join(base, '..'))
    univ2 = glob(os.path.join(root, 'maze-unified-v2', 'results', 'experiment_*.json'))
    if univ2:
        data = load_json(sorted(univ2)[-1])
        agg = data.get('aggregate_stats', {})
        summary['11'] = {
            'success_rate_gedig': agg.get('success_rate'),
            'avg_steps_gedig': agg.get('avg_steps'),
            'avg_graph_nodes': agg.get('avg_graph_nodes'),
            'avg_graph_edges': agg.get('avg_graph_edges'),
        }

    # Real/optimized quick summaries (sanity for 15x15)
    real = glob(os.path.join(res_dir, 'real_gedig_test', 'summary_*.json'))
    if real:
        data = load_json(sorted(real)[-1])
        by_strat = {d['strategy']: d for d in data if 'strategy' in d}
        simp = by_strat.get('simple', {})
        ged = by_strat.get('gedig', {})
        summary.setdefault('15', {}).update({
            'quick_simple_avg_steps': simp.get('avg_steps'),
            'quick_gedig_avg_steps': ged.get('avg_steps'),
        })

    out_dir = os.path.join(res_dir, 'scaling')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'scaling_summary_from_existing.json')
    with open(out_path, 'w') as f:
        json.dump({'summary': summary}, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
