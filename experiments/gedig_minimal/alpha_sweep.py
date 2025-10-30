"""Alpha (frontier_weight) sweep utility for R2 frontier approximation.

Runs batches for a set of alpha values and produces a consolidated CSV & JSON summary
focused on comparing gedig_simple vs gedig_r2.
"""
from __future__ import annotations
import argparse
import json
import pathlib
import statistics as stats
from typing import List, Dict, Any
from run_batch import ensure_dir, parse_args as _parse_args  # reuse for consistency (will ignore extra)
from simple_navigator import run_episode, EpisodeResult


def run_alpha(alpha: float, sizes: List[str], seeds: List[int], step_limit: int) -> Dict[str, Any]:
    rows: List[EpisodeResult] = []
    for size in sizes:
        w,h = map(int, size.lower().split('x'))
        for seed in seeds:
            for policy in ('gedig_simple','gedig_r2'):
                ep, _ = run_episode(policy, w, h, seed, step_limit=step_limit, save_steps=False, frontier_weight=alpha)
                rows.append(ep)
    def mean(xs): return stats.mean(xs) if xs else 0.0
    def agg(policy: str):
        subset = [r for r in rows if r.policy == policy]
        succ = [1 if r.success else 0 for r in subset]
        steps_s = [r.steps for r in subset if r.success]
        return {
            'runs': len(subset),
            'success_rate': mean(succ),
            'mean_steps_success': mean(steps_s),
            'exploration_ratio_mean': mean([r.exploration_ratio for r in subset]),
            'path_efficiency_mean': mean([r.path_efficiency for r in subset if r.success]),
            'gedig_auc_mean': mean([r.gedig_auc for r in subset])
        }
    return {
        'alpha': alpha,
        'gedig_simple': agg('gedig_simple'),
        'gedig_r2': agg('gedig_r2')
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--alphas', nargs='+', default=['0.15','0.25','0.35','0.45','0.60'])
    ap.add_argument('--maze-sizes', nargs='+', default=['8x8','12x12'])
    ap.add_argument('--seeds', nargs='+', default=[str(i) for i in range(10)])
    ap.add_argument('--step-limit', type=int, default=500)
    ap.add_argument('--out-dir', default='results_alpha')
    args = ap.parse_args()

    alphas = [float(a) for a in args.alphas]
    seeds = [int(s) for s in args.seeds]
    out_root = pathlib.Path(args.out_dir)
    ensure_dir(out_root)

    summaries: List[Dict[str,Any]] = []
    for a in alphas:
        print(f"[alpha={a}] running...")
        summary = run_alpha(a, args.maze_sizes, seeds, args.step_limit)
        summaries.append(summary)

    # Write JSON
    (out_root / 'alpha_summary.json').write_text(json.dumps(summaries, ensure_ascii=False, indent=2))

    # CSV
    header = 'alpha,policy,success_rate,mean_steps_success,exploration_ratio_mean,path_efficiency_mean,gedig_auc_mean'
    lines = [header]
    for s in summaries:
        for policy in ('gedig_simple','gedig_r2'):
            d = s[policy]
            lines.append(
                f"{s['alpha']},{policy},{d['success_rate']},{d['mean_steps_success']},{d['exploration_ratio_mean']},{d['path_efficiency_mean']},{d['gedig_auc_mean']}"
            )
    (out_root / 'alpha_summary.csv').write_text('\n'.join(lines))
    print('Saved summaries to', out_root)

if __name__ == '__main__':
    main()
