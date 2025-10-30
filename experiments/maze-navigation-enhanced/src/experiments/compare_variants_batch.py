#!/usr/bin/env python3
"""Compare multi-seed aggregates between complex and ultra variants.

Runs multi-seed batches for each variant and outputs combined summary.

Example:
  python compare_variants_batch.py --seeds 101 202 303 --max_steps 1200
"""
from __future__ import annotations
import os, sys, argparse, json, datetime, statistics
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from maze_layouts import (
    create_complex_maze, create_ultra_maze,
    COMPLEX_DEFAULT_START, COMPLEX_DEFAULT_GOAL,
    ULTRA_DEFAULT_START, ULTRA_DEFAULT_GOAL
)
from navigation.maze_navigator import MazeNavigator  # type: ignore
from generate_complex_maze_report import loop_erased_length


def run_variant(variant: str, seeds, max_steps: int, temperature: float, gedig_threshold: float, backtrack_threshold: float):
    results=[]
    for s in seeds:
        np.random.seed(s)
        if variant=='complex':
            maze=create_complex_maze(); start, goal = COMPLEX_DEFAULT_START, COMPLEX_DEFAULT_GOAL
        else:
            maze=create_ultra_maze(); start, goal = ULTRA_DEFAULT_START, ULTRA_DEFAULT_GOAL
        weights = np.array([1.0,1.0,0.0,0.0,3.0,2.0,0.1,0.0])
        nav = MazeNavigator(maze, start, goal, weights=weights, temperature=temperature,
                            gedig_threshold=gedig_threshold, backtrack_threshold=backtrack_threshold,
                            wiring_strategy='simple', simple_mode=True, backtrack_debounce=True)
        success = nav.run(max_steps=max_steps)
        stats = nav.get_statistics()
        loop_len = loop_erased_length(nav.path)
        redundancy = (len(nav.path)/loop_len) if loop_len else None
        gedig_stats = stats.get('gedig_stats', {})
        sm = stats.get('simple_mode', {})
        results.append({
            'seed': s,
            'success': success,
            'steps': stats.get('steps'),
            'unique_positions': stats.get('unique_positions'),
            'loop_redundancy': redundancy,
            'mean_geDIG': gedig_stats.get('mean'),
            'backtrack_trigger_rate': sm.get('backtrack_trigger_rate')
        })
    return results


def aggregate(rows):
    if not rows: return {}
    out={}
    for key in ['steps','unique_positions','loop_redundancy','mean_geDIG','backtrack_trigger_rate']:
        vals=[r[key] for r in rows if r.get(key) is not None]
        if not vals: continue
        out[key]={
            'mean': float(statistics.fmean(vals)),
            'stdev': float(statistics.pstdev(vals)) if len(vals)>1 else 0.0
        }
    out['success_rate'] = sum(1 for r in rows if r['success'])/len(rows)
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[101,202,303])
    ap.add_argument('--max_steps', type=int, default=1500)
    ap.add_argument('--temperature', type=float, default=0.1)
    ap.add_argument('--gedig_threshold', type=float, default=0.3)
    ap.add_argument('--backtrack_threshold', type=float, default=-0.2)
    args=ap.parse_args()

    comp=run_variant('complex', args.seeds, args.max_steps, args.temperature, args.gedig_threshold, args.backtrack_threshold)
    ultra=run_variant('ultra', args.seeds, args.max_steps, args.temperature, args.gedig_threshold, args.backtrack_threshold)
    comp_agg=aggregate(comp); ultra_agg=aggregate(ultra)

    timestamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir=os.path.join('results','maze_report', f'compare_{timestamp}')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir,'comparison.json'),'w') as f:
        json.dump({'params': vars(args), 'complex': {'per_seed': comp, 'aggregate': comp_agg}, 'ultra': {'per_seed': ultra, 'aggregate': ultra_agg}}, f, indent=2)
    # Markdown quick summary
    with open(os.path.join(out_dir,'summary.md'),'w') as f:
        f.write('# Variant Comparison (Complex vs Ultra)\n\n')
        f.write('## Aggregates\n')
        for name, agg in [('Complex', comp_agg), ('Ultra', ultra_agg)]:
            f.write(f'### {name}\n')
            f.write(f"Success rate: {agg.get('success_rate'):.2f}\n")
            for k in ['steps','loop_redundancy','mean_geDIG','backtrack_trigger_rate']:
                if k in agg:
                    f.write(f"- {k}: {agg[k]['mean']:.3f} ± {agg[k]['stdev']:.3f}\n")
            f.write('\n')
        f.write('## Delta (Ultra - Complex)\n')
        for k in ['steps','loop_redundancy','mean_geDIG','backtrack_trigger_rate']:
            if k in comp_agg and k in ultra_agg:
                delta = ultra_agg[k]['mean'] - comp_agg[k]['mean']
                f.write(f'- {k}: {delta:+.3f}\n')
    print(f'Comparison written to {out_dir}')

if __name__ == '__main__':
    main()
