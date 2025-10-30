#!/usr/bin/env python3
"""Run multi-seed batches for complex / ultra maze variants and aggregate oracle-free stats.

Outputs one JSON + optional markdown summary under results/maze_report/batch_<variant>_<timestamp>/ :
  - aggregate.json : per-seed stats plus aggregates
  - summary.md     : table of key metrics (steps, loop_redundancy, backtrack_trigger_rate, mean_geDIG)

Usage:
  python multi_seed_maze_batch.py --variant ultra --seeds 111 222 333 444 555 --max_steps 1200
"""
from __future__ import annotations
import os, sys, json, argparse, datetime, statistics
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from generate_complex_maze_report import loop_erased_length  # reuse helper
from maze_layouts import (
    create_complex_maze, create_ultra_maze, create_large_maze,
    COMPLEX_DEFAULT_START, COMPLEX_DEFAULT_GOAL,
    ULTRA_DEFAULT_START, ULTRA_DEFAULT_GOAL,
    LARGE_DEFAULT_START, LARGE_DEFAULT_GOAL
)
from results_paths import RESULTS_BASE
from navigation.maze_navigator import MazeNavigator  # type: ignore
try:
    from indexes.vector_index import InMemoryIndex  # type: ignore
except Exception:
    InMemoryIndex = None  # type: ignore


def run_single(variant: str, seed: int, max_steps: int, temperature: float, gedig_threshold: float, backtrack_threshold: float, use_vector_index: bool, wiring_strategy: str):
    np.random.seed(seed)
    if variant == 'complex':
        maze = create_complex_maze(); start, goal = COMPLEX_DEFAULT_START, COMPLEX_DEFAULT_GOAL
    elif variant == 'ultra':
        maze = create_ultra_maze(); start, goal = ULTRA_DEFAULT_START, ULTRA_DEFAULT_GOAL
    else:
        maze = create_large_maze(); start, goal = LARGE_DEFAULT_START, LARGE_DEFAULT_GOAL
    weights = np.array([1.0,1.0,0.0,0.0,3.0,2.0,0.1,0.0])
    vector_index = None
    ws = wiring_strategy
    if use_vector_index and InMemoryIndex is not None:
        vector_index = InMemoryIndex()
        # query 配線を明示指定 (ユーザが別途渡さない限り)
        if ws == 'auto':
            ws = 'query'
    if ws == 'auto':  # 何も指定が無い場合デフォルト simple
        ws = 'simple'
    nav = MazeNavigator(
        maze, start, goal, weights=weights, temperature=temperature,
        gedig_threshold=gedig_threshold, backtrack_threshold=backtrack_threshold,
        wiring_strategy=ws, simple_mode=True, backtrack_debounce=True,
        vector_index=vector_index
    )
    success = nav.run(max_steps=max_steps)
    stats = nav.get_statistics()
    path = nav.path
    loop_len = loop_erased_length(path)
    redundancy = (len(path) / loop_len) if loop_len else None
    gedig_stats = stats.get('gedig_stats', {})
    sm = stats.get('simple_mode', {})
    return {
        'seed': seed,
        'success': success,
        'steps': stats.get('steps'),
        'unique_positions': stats.get('unique_positions'),
        'loop_erased_length': loop_len,
        'loop_redundancy': redundancy,
        'mean_geDIG': gedig_stats.get('mean'),
        'std_geDIG': gedig_stats.get('std'),
        'min_geDIG': gedig_stats.get('min'),
        'max_geDIG': gedig_stats.get('max'),
        'backtrack_trigger_rate': sm.get('backtrack_trigger_rate'),
        'queries_per_step': sm.get('queries_per_step'),
    }


def aggregate(results: list[dict]):
    agg = {}
    if not results:
        return agg
    def col(name):
        vals = [r[name] for r in results if r.get(name) is not None]
        return vals
    for key in ['steps','unique_positions','loop_erased_length','loop_redundancy','mean_geDIG','std_geDIG','min_geDIG','max_geDIG','backtrack_trigger_rate','queries_per_step']:
        vals = col(key)
        if not vals:
            continue
        agg[key] = {
            'mean': float(statistics.fmean(vals)),
            'stdev': float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0,
            'min': float(min(vals)),
            'max': float(max(vals))
        }
    agg['success_rate'] = sum(1 for r in results if r['success']) / len(results)
    return agg


def write_summary_md(path: str, variant: str, params: dict, results: list[dict], agg: dict):
    with open(path,'w') as f:
        f.write(f"# Multi-seed {variant} Maze Batch Summary\n\n")
        f.write('## Parameters\n')
        for k,v in params.items():
            f.write(f'- **{k}**: {v}\n')
        f.write('\n## Per-seed Metrics\n')
        f.write('| seed | steps | unique | loop_red | mean_geDIG | bt_rate | success |\n')
        f.write('|------|-------|--------|----------|------------|---------|---------|\n')
        for r in results:
            f.write(f"| {r['seed']} | {r['steps']} | {r['unique_positions']} | {r['loop_redundancy']:.2f} | {r['mean_geDIG']:.3f} | {r['backtrack_trigger_rate']:.3f} | {int(r['success'])} |\n")
        f.write('\n## Aggregates\n')
        f.write(f"- Success rate: {agg.get('success_rate'):.2f}\n")
        for k in ['steps','loop_redundancy','mean_geDIG','backtrack_trigger_rate']:
            if k in agg:
                a = agg[k]
                f.write(f"- {k}: mean {a['mean']:.3f} ± {a['stdev']:.3f} (min {a['min']:.3f} max {a['max']:.3f})\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', choices=['complex','ultra','large'], default='complex')
    ap.add_argument('--seeds', type=int, nargs='+', default=[101,202,303,404,505])
    ap.add_argument('--max_steps', type=int, default=1500)
    ap.add_argument('--temperature', type=float, default=0.1)
    ap.add_argument('--gedig_threshold', type=float, default=0.3)
    ap.add_argument('--backtrack_threshold', type=float, default=-0.2)
    ap.add_argument('--no_markdown', action='store_true')
    ap.add_argument('--use_vector_index', action='store_true', help='InMemoryIndex を有効化 (query wiring を使用)。')
    ap.add_argument('--wiring_strategy', default='auto', help="'simple' / 'query' / 'gedig' など。'auto' は vector index 有効時 query, それ以外 simple")
    args = ap.parse_args()

    results=[]
    for s in args.seeds:
        r = run_single(
            args.variant, s, args.max_steps, args.temperature,
            args.gedig_threshold, args.backtrack_threshold,
            args.use_vector_index, args.wiring_strategy
        )
        results.append(r)
        print(f"seed {s} steps={r['steps']} loop_red={r['loop_redundancy']:.2f} mean_geDIG={r['mean_geDIG']:.3f} bt_rate={r['backtrack_trigger_rate']:.3f}")
    agg = aggregate(results)

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(RESULTS_BASE, f"batch_{args.variant}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir,'aggregate.json'),'w') as f:
        json.dump({'params': vars(args), 'per_seed': results, 'aggregate': agg}, f, indent=2)
    if not args.no_markdown:
        write_summary_md(os.path.join(out_dir,'summary.md'), args.variant, vars(args), results, agg)
    print(f"Batch results written to {out_dir}")

if __name__ == '__main__':
    main()
