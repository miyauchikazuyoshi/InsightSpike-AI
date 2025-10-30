"""Batch runner for minimal geDIG (Layer3) experiment.

Generates raw episode JSON files and summary aggregates.
"""
from __future__ import annotations
import argparse
import json
import pathlib
import statistics as stats
from typing import List, Dict, Any
from simple_navigator import run_episode, EpisodeResult


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--maze-sizes', nargs='+', default=['8x8','12x12'])
    ap.add_argument('--seeds', nargs='+', default=[str(i) for i in range(10)])
    ap.add_argument('--policies', nargs='+', default=['random','gedig_simple','gedig_r2','threshold_2','threshold_3'])
    ap.add_argument('--include-threshold1', action='store_true', help='閾値冗長性検証のため threshold_1 を追加する')
    ap.add_argument('--frontier-weight', type=float, default=0.5)
    ap.add_argument('--episodes-per-seed', type=int, default=1)
    ap.add_argument('--step-limit', type=int, default=500)
    ap.add_argument('--out-dir', default='results')
    ap.add_argument('--no-save-steps', action='store_true')
    ap.add_argument('--progress', action='store_true')
    return ap.parse_args()


def ensure_dir(p: pathlib.Path):
    p.mkdir(parents=True, exist_ok=True)


def episode_to_json(ep: EpisodeResult) -> Dict[str, Any]:
    d = {k:v for k,v in ep.__dict__.items() if k != 'meta'}
    if ep.meta:
        d['meta'] = ep.meta
    return d


def aggregate(records: List[EpisodeResult]) -> Dict[str, Any]:
    if not records:
        return {}
    def mean(xs): return stats.mean(xs) if xs else 0.0
    def stdev(xs): return stats.pstdev(xs) if len(xs) > 1 else 0.0
    success = [1 if r.success else 0 for r in records]
    steps_success = [r.steps for r in records if r.success]
    return {
        'runs': len(records),
        'success_rate': mean(success),
        'mean_steps_success': mean(steps_success),
        'path_efficiency_mean': mean([r.path_efficiency for r in records if r.success]),
        'exploration_ratio_mean': mean([r.exploration_ratio for r in records]),
        'gedig_auc_mean': mean([r.gedig_auc for r in records]),
        'gedig_auc_stdev': stdev([r.gedig_auc for r in records]),
    }


def main():
    args = parse_args()
    out_root = pathlib.Path(args.out_dir)
    raw_dir = out_root / 'raw'
    ensure_dir(raw_dir)

    all_eps: List[EpisodeResult] = []

    total = len(args.maze_sizes)*len(args.seeds)*len(args.policies)*args.episodes_per_seed
    done = 0

    for size in args.maze_sizes:
        w,h = map(int, size.lower().split('x'))
        policies = list(args.policies)
        if args.include_threshold1 and 'threshold_1' not in policies:
            policies.insert(2, 'threshold_1')
        for policy in policies:
            for seed_s in args.seeds:
                seed = int(seed_s)
                for rep in range(args.episodes_per_seed):
                    ep, steps = run_episode(policy, w, h, seed + rep, step_limit=args.step_limit, save_steps=not args.no_save_steps, frontier_weight=args.frontier_weight)
                    all_eps.append(ep)
                    out_path = raw_dir / f"{w}x{h}" / policy
                    ensure_dir(out_path)
                    fname = out_path / f"seed_{seed}_rep{rep}.json"
                    data = episode_to_json(ep)
                    if not args.no_save_steps:
                        data['gedig_series'] = ep.gedig_series
                    fname.write_text(json.dumps(data, ensure_ascii=False, indent=2))
                    done += 1
                    if args.progress:
                        print(f"[{done}/{total}] {size} {policy} seed={seed} rep={rep} success={ep.success} steps={ep.steps}")

    # Aggregations
    summary_dir = out_root / 'summary'
    ensure_dir(summary_dir)

    # Aggregate overall
    summary_global = aggregate(all_eps)
    summary_global['note'] = 'Overall aggregate of all episodes.'
    (summary_dir / 'aggregate.json').write_text(json.dumps(summary_global, ensure_ascii=False, indent=2))

    # Aggregate by policy
    by_policy = {}
    for policy in {ep.policy for ep in all_eps}:
        by_policy[policy] = aggregate([r for r in all_eps if r.policy == policy])
    (summary_dir / 'by_policy.json').write_text(json.dumps(by_policy, ensure_ascii=False, indent=2))

    # Aggregate by (maze_size, policy)
    by_size_policy = {}
    for size in args.maze_sizes:
        w,h = map(int, size.lower().split('x'))
        key_size = f"{w}x{h}"
        by_size_policy[key_size] = {}
        for policy in {ep.policy for ep in all_eps}:
            subset = [r for r in all_eps if r.policy == policy and r.maze_size==(w,h)]
            by_size_policy[key_size][policy] = aggregate(subset)
    (summary_dir / 'by_size_policy.json').write_text(json.dumps(by_size_policy, ensure_ascii=False, indent=2))

    print("Run complete. Summary written to", summary_dir)

if __name__ == '__main__':
    main()
