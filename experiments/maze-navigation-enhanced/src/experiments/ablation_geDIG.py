#!/usr/bin/env python3
"""Ablation: compare normal geDIG weights vs randomized weights vs zeroed weights.
Collect loop redundancy + (if available) terminal phase AUC from deadend probe style labeling.
"""
from __future__ import annotations
import os, sys, argparse, json, datetime, random, traceback
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from maze_layouts import (
    create_ultra_maze, create_complex_maze,
    COMPLEX_DEFAULT_START, COMPLEX_DEFAULT_GOAL,
    ULTRA_DEFAULT_START, ULTRA_DEFAULT_GOAL,
    create_ultra_maze_50 as _create_ultra_maze_50,
    create_ultra_maze_50_dense_deadends as _create_ultra_maze_50_dense_deadends,
    create_ultra_maze_50_moderate_deadends as _create_ultra_maze_50_moderate_deadends,
    ULTRA50_DEFAULT_START, ULTRA50_DEFAULT_GOAL,
    ULTRA50HD_DEFAULT_START, ULTRA50HD_DEFAULT_GOAL,
    ULTRA50MD_DEFAULT_START, ULTRA50MD_DEFAULT_GOAL
)
from results_paths import RESULTS_BASE
try:
    from navigation.maze_navigator import MazeNavigator  # type: ignore
    from indexes.vector_index import InMemoryIndex
except Exception as e:
    print('[EARLY_IMPORT_ERROR]', e)
    traceback.print_exc()
    raise
from generate_complex_maze_report import loop_erased_length
from metrics_utils import compute_path_metrics, effect_size_and_ci


def _build_maze(variant: str):
    if variant == 'complex':
        return create_complex_maze(), COMPLEX_DEFAULT_START, COMPLEX_DEFAULT_GOAL
    if variant == 'ultra':
        return create_ultra_maze(), ULTRA_DEFAULT_START, ULTRA_DEFAULT_GOAL
    if variant == 'ultra50':
        return _create_ultra_maze_50(), ULTRA50_DEFAULT_START, ULTRA50_DEFAULT_GOAL
    if variant == 'ultra50hd':
        return _create_ultra_maze_50_dense_deadends(), ULTRA50HD_DEFAULT_START, ULTRA50HD_DEFAULT_GOAL
    if variant == 'ultra50md':
        return _create_ultra_maze_50_moderate_deadends(), ULTRA50MD_DEFAULT_START, ULTRA50MD_DEFAULT_GOAL
    raise ValueError(f'Unsupported variant: {variant}')

def run_variant(kind, variant, seed, max_steps, temperature, gedig_threshold, backtrack_threshold,
                global_recall=False, recall_threshold=0.01,
                wiring_top_k=4, max_graph_snapshots=None,
                dense_metric_interval=1, snapshot_skip_idle=False,
                wiring_strategy_override=None, use_vector_index=False,
                enable_flush=False, flush_interval=200, max_in_memory=10000,
                max_in_memory_positions=None, persistence_dir=None):
    np.random.seed(seed)
    maze, start, goal = _build_maze(variant)
    if kind=='normal':
        weights = np.array([1.0,1.0,0.0,0.0,3.0,2.0,0.1,0.0])
    elif kind=='zero':
        weights = np.zeros(8)
    else:  # random
        weights = np.random.uniform(-0.5,0.5,size=8)
    nav=MazeNavigator(maze, start, goal, weights=weights, temperature=temperature,
                      gedig_threshold=gedig_threshold, backtrack_threshold=backtrack_threshold,
                      wiring_strategy=wiring_strategy_override or 'simple', simple_mode=True, backtrack_debounce=True,
                      global_recall_enabled=global_recall, recall_score_threshold=recall_threshold,
                      wiring_top_k=wiring_top_k, max_graph_snapshots=max_graph_snapshots,
                      dense_metric_interval=dense_metric_interval, snapshot_skip_idle=snapshot_skip_idle,
                      verbosity=GLOBAL_VERBOSITY, progress_interval=GLOBAL_PROGRESS_INTERVAL,
                      vector_index=(InMemoryIndex() if (use_vector_index and InMemoryIndex) else None),
                      enable_flush=enable_flush, flush_interval=flush_interval,
                      max_in_memory=max_in_memory,
                      max_in_memory_positions=max_in_memory_positions,
                      persistence_dir=persistence_dir)
    nav.run(max_steps=max_steps)
    metrics = compute_path_metrics(nav.path, nav.gedig_history, gedig_threshold)
    # Enriched stats (Phase T3-2)
    nav_stats = nav.get_statistics()
    timing = nav_stats.get('timing', {}) or {}
    reason_counts = {}
    total_plans = 0
    for ev in nav.event_log:
        if ev.get('type') == 'backtrack_path_plan':
            total_plans += 1
            msg = ev.get('message')
            reason = None
            if isinstance(msg, dict):
                reason = msg.get('reason')
            reason_counts[reason or 'unknown'] = reason_counts.get(reason or 'unknown', 0) + 1
    def pick(kind, field):
        return timing.get(kind, {}).get(field)
    enriched = {
        'timing_wiring_mean_ms': pick('wiring_ms','mean_ms'),
        'timing_wiring_p95_ms': pick('wiring_ms','p95_ms'),
        'timing_snapshot_mean_ms': pick('snapshot_ms','mean_ms'),
        'timing_gedig_mean_ms': pick('gedig_ms','mean_ms'),
        'timing_recall_mean_ms': pick('recall_ms','mean_ms'),
        'backtrack_plan_total': total_plans,
        'backtrack_plan_reasons': reason_counts,
        'global_recall_enabled': global_recall,
        'recall_threshold_used': recall_threshold,
        'wiring_top_k': wiring_top_k,
        'dense_metric_interval': dense_metric_interval,
        'snapshot_skip_idle': snapshot_skip_idle,
        'graph_snapshots': nav_stats.get('graph_stats', {}).get('snapshots')
    }
    metrics.update(enriched)
    return {'kind':kind,'seed':seed, **metrics}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--variant', choices=['complex','ultra','ultra50','ultra50hd','ultra50md'], default='ultra50hd')
    ap.add_argument('--seeds', type=int, nargs='+', default=[11,22,33,44,55,66,77,88,99,111])
    ap.add_argument('--max_steps', type=int, default=800)
    ap.add_argument('--gedig_threshold', type=float, default=0.3)
    ap.add_argument('--backtrack_threshold', type=float, default=-0.2)
    ap.add_argument('--temperature', type=float, default=0.1)
    ap.add_argument('--bootstrap_iterations', type=int, default=1000, help='Bootstrap iterations for aggregated mean CIs (0=skip)')
    ap.add_argument('--verbosity', type=int, default=1, help='Navigator verbosity (0,1,2)')
    ap.add_argument('--progress_interval', type=int, default=80, help='Progress print interval')
    ap.add_argument('--record_seeds_path', type=str, default=os.path.join(RESULTS_BASE,'run_config_seeds.json'), help='Path to write seeds JSON before run (overwrites)')
    # Phase0/1 new flags
    ap.add_argument('--global_recall', action='store_true', help='Enable global recall heuristic')
    ap.add_argument('--recall_threshold', type=float, default=0.01, help='geDIG threshold for global recall activation')
    ap.add_argument('--wiring_top_k', type=int, default=4, help='Top-K for query wiring (performance optimization)')
    ap.add_argument('--max_graph_snapshots', type=int, default=None, help='Cap number of stored graph snapshots')
    ap.add_argument('--dense_metric_interval', type=int, default=1, help='Interval for capturing dense geDIG metrics (>=1)')
    ap.add_argument('--snapshot_skip_idle', action='store_true', help='Skip snapshot if no graph growth (after 1 idle)')
    ap.add_argument('--wiring_strategy', choices=['simple','query'], default='simple')
    ap.add_argument('--use_vector_index', action='store_true', help='Enable in-memory vector index (query strategy only)')
    ap.add_argument('--enable_flush', action='store_true')
    ap.add_argument('--flush_interval', type=int, default=200)
    ap.add_argument('--max_in_memory', type=int, default=10000)
    ap.add_argument('--max_in_memory_positions', type=int, default=None)
    ap.add_argument('--persistence_dir', type=str, default=None)
    args=ap.parse_args()

    print(f"[RUN] ablation_geDIG variant={args.variant} seeds={args.seeds} max_steps={args.max_steps} bootstrap={args.bootstrap_iterations} verbosity={args.verbosity} interval={args.progress_interval}")

    # Globals for run_variant
    global GLOBAL_VERBOSITY, GLOBAL_PROGRESS_INTERVAL
    GLOBAL_VERBOSITY = args.verbosity
    GLOBAL_PROGRESS_INTERVAL = max(1, args.progress_interval)

    kinds=['normal','random','zero']
    # Record seeds globally before heavy runs
    try:
        os.makedirs(os.path.dirname(args.record_seeds_path), exist_ok=True)
        with open(args.record_seeds_path,'w') as f:
            json.dump({'seeds': args.seeds, 'written_at': datetime.datetime.now().isoformat(), 'context':'ablation_geDIG'}, f, indent=2)
    except Exception as e:
        print(f"[warn] failed to record seeds: {e}")
    records=[]
    for s in args.seeds:
        for k in kinds:
            records.append(run_variant(k,args.variant,s,args.max_steps,args.temperature,args.gedig_threshold,args.backtrack_threshold,
                                       global_recall=args.global_recall,
                                       recall_threshold=args.recall_threshold,
                                       wiring_top_k=args.wiring_top_k,
                                       max_graph_snapshots=args.max_graph_snapshots,
                                       dense_metric_interval=args.dense_metric_interval,
                                       snapshot_skip_idle=args.snapshot_skip_idle,
                                       wiring_strategy_override=args.wiring_strategy,
                                       use_vector_index=args.use_vector_index,
                                       enable_flush=args.enable_flush,
                                       flush_interval=args.flush_interval,
                                       max_in_memory=args.max_in_memory,
                                       max_in_memory_positions=args.max_in_memory_positions,
                                       persistence_dir=args.persistence_dir))

    import collections, statistics
    agg={}
    for k in kinds:
        vals=[r['loop_redundancy'] for r in records if r['kind']==k and r.get('loop_redundancy') is not None]
        if vals:
            agg[k]={
                'n': len(vals),
                'loop_redundancy_mean': float(np.mean(vals)),
                'loop_redundancy_std': float(np.std(vals, ddof=1)) if len(vals)>1 else 0.0,
                'loop_redundancy_median': float(np.median(vals)),
                'clipped_mean': float(np.mean([r['clipped_redundancy'] for r in records if r['kind']==k and r.get('clipped_redundancy') is not None])),
                'unique_cov_mean': float(np.mean([r['unique_coverage'] for r in records if r['kind']==k and r.get('unique_coverage') is not None])),
                'backtrack_rate_mean': float(np.mean([r['backtrack_rate'] for r in records if r['kind']==k and r.get('backtrack_rate') is not None]))
            }
    ts=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir=os.path.join(RESULTS_BASE, f'gedig_ablation_{args.variant}_{ts}')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir,'records.json'),'w') as f:
        json.dump({'params': vars(args), 'records': records, 'aggregate': agg}, f, indent=2)

    # Effect sizes: normal vs random / zero for key metrics
    metrics_keys = ['loop_redundancy','clipped_redundancy','unique_coverage','backtrack_rate','mean_geDIG','geDIG_low_frac']
    def by(kind, key):
        return [r.get(key) for r in records if r['kind']==kind and r.get(key) is not None]
    stats = {}
    for base in ['random','zero']:
        comp = {}
        for mk in metrics_keys:
            a_vals = by('normal', mk)
            b_vals = by(base, mk)
            if a_vals and b_vals:
                comp[mk] = effect_size_and_ci(a_vals, b_vals)
        stats[f'normal_vs_{base}'] = comp
    with open(os.path.join(out_dir,'stats.json'),'w') as f:
        json.dump({'effect_sizes': stats, 'metrics': metrics_keys}, f, indent=2)

    # simple text summary
    lines=['geDIG Ablation Summary', f"Variant: {args.variant}"]
    for k in kinds:
        if k in agg:
            a=agg[k]
            lines.append(f"{k}: loop_red_mean={a['loop_redundancy_mean']:.3f}±{a['loop_redundancy_std']:.3f} clipped_mean={a['clipped_mean']:.3f} unique_cov_mean={a['unique_cov_mean']:.3f} backtrack_rate_mean={a['backtrack_rate_mean']:.3f} n={a['n']}")
    with open(os.path.join(out_dir,'summary.md'),'w') as f:
        # Append timing + backtrack plan reason aggregates per kind (normal only focus)
        normal_records = [r for r in records if r['kind']=='normal' and r.get('timing_wiring_mean_ms') is not None]
        if normal_records:
            import numpy as _np
            tm_keys = ['timing_wiring_mean_ms','timing_snapshot_mean_ms','timing_gedig_mean_ms','timing_recall_mean_ms']
            timing_means = {k: float(_np.mean([r[k] for r in normal_records if r.get(k) is not None])) for k in tm_keys}
            lines += ['', '## Timing (normal aggregated)', '', '| metric | mean_ms |', '|--------|---------|']
            for k,v in timing_means.items():
                lines.append(f"| {k} | {v:.3f} |")
            # Backtrack plan reasons aggregated
            reason_totals = {}
            for r in normal_records:
                for rk, rv in (r.get('backtrack_plan_reasons') or {}).items():
                    reason_totals[rk] = reason_totals.get(rk,0)+rv
            lines += ['', '## Backtrack Plan Reasons (normal)', '', '| reason | count |', '|--------|-------|']
            for rk, rv in sorted(reason_totals.items(), key=lambda x:-x[1]):
                lines.append(f"| {rk} | {rv} |")
        f.write('\n'.join(lines))
    # effect size markdown
    es_lines = ["# Effect Sizes (normal vs others)", "", "| comparison | metric | diff_mean | diff_CI | d | d_CI | n_normal | n_other |", "|------------|--------|-----------|---------|---|------|----------|--------|"]
    for comp_name, ms in stats.items():
        for mk, res in ms.items():
            es_lines.append(
                f"| {comp_name} | {mk} | {res['diff_mean']:.3f} | [{res['diff_ci_low']:.3f},{res['diff_ci_high']:.3f}] | {res['cohens_d']:.3f} | [{res['d_ci_low']:.3f},{res['d_ci_high']:.3f}] | {res['n_a']} | {res['n_b']} |"
            )
    with open(os.path.join(out_dir,'effect_sizes.md'),'w') as f:
        f.write('\n'.join(es_lines))
    # Aggregated bootstrap stats (per kind means + diff normal - others)
    if args.bootstrap_iterations > 0:
        rng = np.random.default_rng(42)
        kinds_all = kinds
        metrics_keys = ['loop_redundancy','clipped_redundancy','unique_coverage','backtrack_rate','mean_geDIG','geDIG_low_frac']
        algo_means = {}
        diffs = {}
        B = args.bootstrap_iterations
        for mk in metrics_keys:
            per_kind = {k: np.array([r[mk] for r in records if r['kind']==k and r.get(mk) is not None], dtype=float) for k in kinds_all}
            if not per_kind['normal'].size:
                continue
            means_mk = {}
            for k, arr in per_kind.items():
                if not arr.size:
                    continue
                n = arr.size
                bs_means = []
                for _ in range(B):
                    sample = arr[rng.integers(0,n,size=n)]
                    bs_means.append(sample.mean())
                bs_means = np.sort(bs_means)
                means_mk[k] = {
                    'mean': float(arr.mean()),
                    'ci_low': float(bs_means[int(0.025*B)]),
                    'ci_high': float(bs_means[int(0.975*B)-1])
                }
            algo_means[mk] = means_mk
            # diffs normal - others
            diff_mk = {}
            base_arr = per_kind['normal']
            n_base = base_arr.size
            for k in kinds_all:
                if k == 'normal':
                    continue
                arr = per_kind[k]
                if not arr.size:
                    continue
                n_other = arr.size
                diff_samples = []
                for _ in range(B):
                    sa = base_arr[rng.integers(0,n_base,size=n_base)]
                    sb = arr[rng.integers(0,n_other,size=n_other)]
                    diff_samples.append(sa.mean() - sb.mean())
                diff_samples = np.sort(diff_samples)
                diff_mk[k] = {
                    'mean_diff': float(base_arr.mean() - arr.mean()),
                    'ci_low': float(diff_samples[int(0.025*B)]),
                    'ci_high': float(diff_samples[int(0.975*B)-1])
                }
            diffs[mk] = diff_mk
        with open(os.path.join(out_dir,'aggregated_stats.json'),'w') as f:
            json.dump({'metrics': metrics_keys, 'kind_means': algo_means, 'diffs_normal_minus': diffs}, f, indent=2)

    print('\n'.join(lines))
    print(f'Ablation outputs in {out_dir}')

if __name__=='__main__':
    main()
