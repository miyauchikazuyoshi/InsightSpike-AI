#!/usr/bin/env python3
"""Scaling experiment: vary maze size to see effect on loop redundancy (Simple vs Random).
Assumes create_complex_maze / create_ultra_maze can be approximated by size parameter via a simple generator fallback here.
"""
from __future__ import annotations
import os, sys, argparse, json, datetime, random
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from maze_layouts import create_ultra_maze, create_complex_maze, COMPLEX_DEFAULT_START, COMPLEX_DEFAULT_GOAL, ULTRA_DEFAULT_START, ULTRA_DEFAULT_GOAL
from navigation.maze_navigator import MazeNavigator  # type: ignore
from generate_complex_maze_report import loop_erased_length
from baseline_explorers import run_random
from metrics_utils import compute_path_metrics

# NOTE: For now we approximate scaling by using existing generators and truncating path length window.


def run_simple(variant, seed, max_steps, gedig_threshold, backtrack_threshold, temperature):
    np.random.seed(seed)
    maze = create_complex_maze() if variant=='complex' else create_ultra_maze()
    start, goal = (COMPLEX_DEFAULT_START, COMPLEX_DEFAULT_GOAL) if variant=='complex' else (ULTRA_DEFAULT_START, ULTRA_DEFAULT_GOAL)
    weights = np.array([1.0,1.0,0.0,0.0,3.0,2.0,0.1,0.0])
    nav=MazeNavigator(maze, start, goal, weights=weights, temperature=temperature, gedig_threshold=gedig_threshold, backtrack_threshold=backtrack_threshold, wiring_strategy='simple', simple_mode=True, backtrack_debounce=True)
    nav.run(max_steps=max_steps)
    metrics = compute_path_metrics(nav.path, nav.gedig_history, gedig_threshold)
    return metrics


def run_random_baseline(variant, seed, max_steps):
    np.random.seed(seed)
    maze = create_complex_maze() if variant=='complex' else create_ultra_maze()
    start, goal = (COMPLEX_DEFAULT_START, COMPLEX_DEFAULT_GOAL) if variant=='complex' else (ULTRA_DEFAULT_START, ULTRA_DEFAULT_GOAL)
    rng=random.Random(seed)
    path,_=run_random(maze,start,goal,max_steps,rng)
    metrics = compute_path_metrics(path, None)
    return metrics


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--variants', nargs='+', default=['complex','ultra'])
    ap.add_argument('--steps_grid', type=int, nargs='+', default=[300,500,700,900])
    ap.add_argument('--seeds', type=int, nargs='+', default=[111,222,333])
    ap.add_argument('--gedig_threshold', type=float, default=0.3)
    ap.add_argument('--backtrack_threshold', type=float, default=-0.2)
    ap.add_argument('--temperature', type=float, default=0.1)
    args=ap.parse_args()

    rows=[]
    for variant in args.variants:
        for max_steps in args.steps_grid:
            simple_vals=[]; rand_vals=[]
            for s in args.seeds:
                simple_vals.append(run_simple(variant,s,max_steps,args.gedig_threshold,args.backtrack_threshold,args.temperature))
                rand_vals.append(run_random_baseline(variant,s,max_steps))
            def avg(vals, key):
                arr=[v[key] for v in vals if v.get(key) is not None]
                return float(np.mean(arr)) if arr else None
            rows.append({'variant':variant,'max_steps':max_steps,'algo':'simple',
                         'loop_redundancy_mean': avg(simple_vals,'loop_redundancy'),
                         'clipped_redundancy_mean': avg(simple_vals,'clipped_redundancy'),
                         'unique_cov_mean': avg(simple_vals,'unique_coverage'),
                         'backtrack_rate_mean': avg(simple_vals,'backtrack_rate')})
            rows.append({'variant':variant,'max_steps':max_steps,'algo':'random',
                         'loop_redundancy_mean': avg(rand_vals,'loop_redundancy'),
                         'clipped_redundancy_mean': avg(rand_vals,'clipped_redundancy'),
                         'unique_cov_mean': avg(rand_vals,'unique_coverage'),
                         'backtrack_rate_mean': avg(rand_vals,'backtrack_rate')})

    ts=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir=os.path.join('results','maze_report', f'scaling_{ts}')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir,'scaling.json'),'w') as f:
        json.dump({'params': vars(args), 'rows': rows}, f, indent=2)

    # Plot lines
    import matplotlib.pyplot as plt
    import pandas as pd
    df=pd.DataFrame(rows)
    fig, ax=plt.subplots(figsize=(6,4))
    for variant in df.variant.unique():
        for algo in ['simple','random']:
            sub=df[(df.variant==variant)&(df.algo==algo)].sort_values('max_steps')
            ax.plot(sub.max_steps, sub.loop_redundancy_mean, marker='o', label=f'{variant}-{algo}')
    ax.set_xlabel('Max Steps (proxy for size)')
    ax.set_ylabel('Mean Loop Redundancy')
    ax.set_title('Scaling of Redundancy vs Steps Budget')
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir,'scaling.png'), dpi=140)
    print(f'Scaling results saved in {out_dir}')

if __name__=='__main__':
    main()
