#!/usr/bin/env python3
"""Generate comparative plots: loop redundancy distributions (Simple vs Random vs DFS).
Requires prior runs or executes fresh quick runs.
"""
from __future__ import annotations
import os, sys, argparse, json, datetime, random, traceback
import numpy as np
import matplotlib.pyplot as plt
try:
    from metrics_utils import compute_path_metrics, label_phases, auc_suite, effect_size_and_ci
except Exception as e:
    print('[EARLY_IMPORT_ERROR metrics_utils]', e)
    traceback.print_exc()
    raise

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from maze_layouts import (
    create_complex_maze, create_ultra_maze, create_large_maze,
    COMPLEX_DEFAULT_START, COMPLEX_DEFAULT_GOAL,
    ULTRA_DEFAULT_START, ULTRA_DEFAULT_GOAL,
    LARGE_DEFAULT_START, LARGE_DEFAULT_GOAL,
    # 50x50 variants (may not exist in older branches, guard with hasattr when used)
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
    from indexes.vector_index import InMemoryIndex  # optional
except Exception as e:
    print('[EARLY_IMPORT_ERROR maze_navigator]', e)
    traceback.print_exc()
    raise
from generate_complex_maze_report import loop_erased_length
from baseline_explorers import run_random, run_dfs

# Fallback globals (may be set by visualize_run or other scripts; define if absent)
try:
    GLOBAL_VERBOSITY  # type: ignore  # noqa: F821
except NameError:  # pragma: no cover
    GLOBAL_VERBOSITY = 0  # type: ignore
try:
    GLOBAL_PROGRESS_INTERVAL  # type: ignore  # noqa: F821
except NameError:  # pragma: no cover
    GLOBAL_PROGRESS_INTERVAL = 200  # type: ignore


def _build_maze(variant: str):
    """Return (maze, start, goal) for supported variants."""
    if variant == 'complex':
        return create_complex_maze(), COMPLEX_DEFAULT_START, COMPLEX_DEFAULT_GOAL
    if variant == 'ultra':
        return create_ultra_maze(), ULTRA_DEFAULT_START, ULTRA_DEFAULT_GOAL
    if variant == 'large':
        return create_large_maze(), LARGE_DEFAULT_START, LARGE_DEFAULT_GOAL
    if variant == 'ultra50':
        return _create_ultra_maze_50(), ULTRA50_DEFAULT_START, ULTRA50_DEFAULT_GOAL
    if variant == 'ultra50hd':
        return _create_ultra_maze_50_dense_deadends(), ULTRA50HD_DEFAULT_START, ULTRA50HD_DEFAULT_GOAL
    if variant == 'ultra50md':
        return _create_ultra_maze_50_moderate_deadends(), ULTRA50MD_DEFAULT_START, ULTRA50MD_DEFAULT_GOAL
    raise ValueError(f'Unsupported variant: {variant}')


def run_simple(variant, seed, max_steps, temperature, gedig_threshold, backtrack_threshold,
               collect_auc=False,
               global_recall=False,
               recall_threshold=0.01,
               wiring_top_k=4,
               max_graph_snapshots=None,
               dense_metric_interval=1,
               snapshot_skip_idle=False,
               wiring_strategy_override=None,
               use_vector_index=False,
               enable_flush=False,
               flush_interval=200,
               max_in_memory=10000,
               max_in_memory_positions=None,
               persistence_dir=None,
               ann_backend=None,
               ann_m=16,
               ann_ef_construction=100,
               ann_ef_search=100,
               ann_upgrade_threshold=600,
               catalog_compaction_on_close=False,
               include_events=False,
               include_trace=False,
               # --- New experimental flags (multihop comparison instrumentation) ---
               use_escalation=True,
               escalation_threshold=None,
               include_structural=False,
               force_multihop=False,  # Disable multi-hop
               max_hops=0,  # No lookahead
               gedig_mode='core_raw',
               gedig_scale=25.0,
               gedig_ig_weight=0.1,
               gedig_allow_hop1_approx=None):  # deprecated param retained (ignored)
    np.random.seed(seed)
    maze, start, goal = _build_maze(variant)
    weights = np.array([1.0,1.0,0.0,0.0,3.0,2.0,0.1,0.0])  # 元の重みに戻す
    nav=MazeNavigator(maze, start, goal, weights=weights, temperature=temperature,
                      gedig_threshold=gedig_threshold, backtrack_threshold=backtrack_threshold,
                      wiring_strategy=wiring_strategy_override or 'simple', simple_mode=True, backtrack_debounce=True,
                      backtrack_target_strategy='semantic',
                      global_recall_enabled=global_recall, recall_score_threshold=recall_threshold,
                      wiring_top_k=wiring_top_k, max_graph_snapshots=max_graph_snapshots,
                      dense_metric_interval=dense_metric_interval, snapshot_skip_idle=snapshot_skip_idle,
                      verbosity=GLOBAL_VERBOSITY, progress_interval=GLOBAL_PROGRESS_INTERVAL,
                      vector_index=(InMemoryIndex() if (use_vector_index and InMemoryIndex) else None),
                      enable_flush=enable_flush,
                      flush_interval=flush_interval,
                      max_in_memory=max_in_memory,
                      max_in_memory_positions=max_in_memory_positions,
                      persistence_dir=persistence_dir,
                      ann_backend=ann_backend,
                      ann_m=ann_m,
                      ann_ef_construction=ann_ef_construction,
                      ann_ef_search=ann_ef_search,
                      ann_upgrade_threshold=ann_upgrade_threshold,
                      catalog_compaction_on_close=catalog_compaction_on_close,
                      force_multihop=force_multihop,
                      use_escalation=use_escalation,
                      escalation_threshold=escalation_threshold,
                      gedig_mode=gedig_mode,
                      gedig_scale=gedig_scale,
                      gedig_ig_weight=gedig_ig_weight,
                      # allow_hop1_approx removed
                     )
    nav.run(max_steps=max_steps)
    metrics = compute_path_metrics(nav.path, nav.gedig_history, gedig_threshold)
    # --- Enriched stats (Phase T3-2) ---
    nav_stats = nav.get_statistics()
    timing = nav_stats.get('timing', {}) or {}
    # Backtrack plan events (reasons)
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
    # Flatten timing aggregates (mean / p95)
    def pick(kind, field):
        return timing.get(kind, {}).get(field)
    enriched = {
        'timing_wiring_mean_ms': pick('wiring_ms','mean_ms'),
        'timing_wiring_p95_ms': pick('wiring_ms','p95_ms'),
        'timing_snapshot_mean_ms': pick('snapshot_ms','mean_ms'),
        'timing_recall_mean_ms': pick('recall_ms','mean_ms'),
        'timing_gedig_mean_ms': pick('gedig_ms','mean_ms'),
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
    if collect_auc:
        phases = label_phases(nav.path, maze)
        aucs = auc_suite(nav.gedig_history, phases)
        metrics.update({'auc': aucs, 'phase_counts': {p: phases.count(p) for p in set(phases)}})
    # Include goal attainment + total steps for downstream success-rate analyses
    result = {
        'algo':'simple',
        'seed':seed,
        **metrics,
        'goal_reached': bool(getattr(nav, 'is_goal_reached', False)),
        'steps': int(getattr(nav, 'step_count', len(nav.path) if hasattr(nav,'path') else -1))
    }
    if include_events:
        # Event counts (focused categories)
        counts={}
        for ev in nav.event_log:
            t=ev.get('type')
            counts[t]=counts.get(t,0)+1
        result['event_counts']=counts
        # Selected raw events (truncated) for ANN/flush insight
        key_types={'ann_init','ann_upgrade','ann_upgrade_failed','flush_eviction','rehydration','catalog_compact','catalog_load'}
        slim=[e for e in nav.event_log if e.get('type') in key_types]
        if len(slim)>200:
            slim=slim[:200]
        result['events']=slim
    if include_trace:
        # Full path & geDIG trace for visualization; truncate very large runs defensively
        result['path']=list(nav.path[:5000]) if hasattr(nav,'path') else []
        # geDIG history may be None for baselines; ensure list
        try:
            result['gedig_history']=list(nav.gedig_history[:5000]) if hasattr(nav,'gedig_history') and nav.gedig_history is not None else []
        except Exception:
            result['gedig_history']=[]
        # Optionally include full events (capped) for plotting markers
        full_events = nav.event_log
        if len(full_events)>5000:
            full_events = full_events[:5000]
        result['events_full']=full_events
    if include_structural:
        # Export structural records (truncated) including multihop projections when escalation active
        try:
            structural = getattr(nav, 'gedig_structural', [])
            if structural:
                if len(structural) > 5000:
                    structural = structural[:5000]
                result['gedig_structural'] = structural
        except Exception:
            pass
    return result


def run_baseline(algo, variant, seed, max_steps):
    np.random.seed(seed)
    maze, start, goal = _build_maze(variant)
    rng=random.Random(seed)
    if algo=='random':
        path, _ = run_random(maze,start,goal,max_steps,rng)
    else:
        path, _ = run_dfs(maze,start,goal,max_steps,rng)
    metrics = compute_path_metrics(path, None, gedig_threshold=0.3)
    goal_reached = bool(path and path[-1] == goal)
    return {'algo':algo,'seed':seed, **metrics, 'goal_reached': goal_reached, 'steps': len(path)}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--variant', choices=['complex','ultra','large','ultra50','ultra50hd','ultra50md'], default='ultra50hd')
    ap.add_argument('--seeds', type=int, nargs='+', default=[101,202,303,404,505])
    ap.add_argument('--max_steps', type=int, default=800)
    ap.add_argument('--gedig_threshold', type=float, default=0.3)
    ap.add_argument('--backtrack_threshold', type=float, default=-0.2)
    ap.add_argument('--temperature', type=float, default=0.1)
    ap.add_argument('--bootstrap_iterations', type=int, default=1000, help='bootstrap iterations for aggregated stats (0=skip)')
    ap.add_argument('--verbosity', type=int, default=1, help='Navigator verbosity level (0=silent,1=progress,2=per-step)')
    ap.add_argument('--progress_interval', type=int, default=80, help='Progress print interval steps (verbosity>=1)')
    ap.add_argument('--record_seeds_path', type=str, default=os.path.join(RESULTS_BASE,'run_config_seeds.json'), help='Path to write seeds JSON before run (appended/overwritten).')
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
    # Phase6 ANN related
    ap.add_argument('--ann_backend', type=str, default=None, choices=[None,'hnsw','HNSW'], help='ANN backend (hnsw)')
    ap.add_argument('--ann_m', type=int, default=16, help='HNSW M (only if ann_backend=hnsw)')
    ap.add_argument('--ann_ef_construction', type=int, default=100, help='HNSW ef_construction')
    ap.add_argument('--ann_ef_search', type=int, default=100, help='HNSW ef_search')
    ap.add_argument('--ann_upgrade_threshold', type=int, default=600, help='Automatic upgrade threshold (linear->ANN)')
    ap.add_argument('--catalog_compaction_on_close', action='store_true', help='Compact eviction catalog file at end of run')
    args=ap.parse_args()

    print(f"[RUN] baseline_vs_simple_plot variant={args.variant} seeds={args.seeds} max_steps={args.max_steps} bootstrap={args.bootstrap_iterations} verbosity={args.verbosity} interval={args.progress_interval}")

    # Expose globals for run_simple (avoid refactor)
    global GLOBAL_VERBOSITY, GLOBAL_PROGRESS_INTERVAL
    GLOBAL_VERBOSITY = args.verbosity
    GLOBAL_PROGRESS_INTERVAL = max(1, args.progress_interval)

    # Record seeds (global config file for reproducibility)
    try:
        os.makedirs(os.path.dirname(args.record_seeds_path), exist_ok=True)
        with open(args.record_seeds_path, 'w') as f:
            json.dump({'seeds': args.seeds, 'written_at': datetime.datetime.now().isoformat()}, f, indent=2)
    except Exception as e:
        print(f"[warn] failed to record seeds: {e}")

    records=[]
    for s in args.seeds:
        # simple のみ AUC 計算 (geDIG 利用)
        records.append(run_simple(
            args.variant,s,args.max_steps,args.temperature,args.gedig_threshold,args.backtrack_threshold,
            collect_auc=True,
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
            persistence_dir=args.persistence_dir,
            ann_backend=args.ann_backend,
            ann_m=args.ann_m,
            ann_ef_construction=args.ann_ef_construction,
            ann_ef_search=args.ann_ef_search
            ,ann_upgrade_threshold=args.ann_upgrade_threshold
            ,catalog_compaction_on_close=args.catalog_compaction_on_close
        ))
        records.append(run_baseline('random',args.variant,s,args.max_steps))
        records.append(run_baseline('dfs',args.variant,s,args.max_steps))

    # Plot distribution
    import collections
    by=collections.defaultdict(list)
    for r in records:
        if r.get('loop_redundancy') is not None:
            by[r['algo']].append(r['loop_redundancy'])
    fig, ax=plt.subplots(figsize=(6,4))
    algos=['simple','dfs','random']
    data=[by[a] for a in algos]
    ax.boxplot(data, labels=algos, showmeans=True, meanline=True)
    ax.set_ylabel('Loop Redundancy')
    ax.set_title(f'Redundancy Comparison ({args.variant})')
    plt.tight_layout()

    ts=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir=os.path.join(RESULTS_BASE, f'baseline_compare_{args.variant}_{ts}')
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir,'redundancy_boxplot.png'), dpi=140)
    # Save extended metrics table
    with open(os.path.join(out_dir,'records.json'),'w') as f:
        json.dump({'params': vars(args), 'records': records}, f, indent=2)
    # Markdown summary quick view
    md_lines = ["# Baseline vs Simple Metrics", "", "| algo | seeds | mean(loop_red) | mean(clipped) | mean(unique_cov) | mean(backtrack_rate) |", "|------|------|----------------|---------------|------------------|--------------------|"]
    import numpy as np
    for algo in sorted(set(r['algo'] for r in records)):
        subset = [r for r in records if r['algo']==algo]
        def mean_key(k):
            vals=[x[k] for x in subset if x.get(k) is not None]
            return f"{np.mean(vals):.3f}" if vals else 'NA'
        md_lines.append(f"| {algo} | {len(subset)} | {mean_key('loop_redundancy')} | {mean_key('clipped_redundancy')} | {mean_key('unique_coverage')} | {mean_key('backtrack_rate')} |")
    with open(os.path.join(out_dir,'summary.md'),'w') as f:
        # Append timing summary (simple only)
        simple_timing = [r for r in records if r['algo']=='simple' and r.get('timing_wiring_mean_ms') is not None]
        if simple_timing:
            import numpy as _np
            tm_keys = ['timing_wiring_mean_ms','timing_snapshot_mean_ms','timing_gedig_mean_ms','timing_recall_mean_ms']
            timing_means = {k: float(_np.mean([r[k] for r in simple_timing if r.get(k) is not None])) for k in tm_keys}
            md_lines += ["", "## Timing (Simple Aggregated)", "", "| metric | mean_ms |", "|--------|---------|"]
            for k,v in timing_means.items():
                md_lines.append(f"| {k} | {v:.3f} |")
            # Backtrack plan reasons aggregated
            reason_totals = {}
            for r in simple_timing:
                for rk, rv in (r.get('backtrack_plan_reasons') or {}).items():
                    reason_totals[rk] = reason_totals.get(rk,0)+rv
            md_lines += ["", "## Backtrack Plan Reasons (Simple)", "", "| reason | count |", "|--------|-------|"]
            for rk, rv in sorted(reason_totals.items(), key=lambda x:-x[1]):
                md_lines.append(f"| {rk} | {rv} |")
        f.write("\n".join(md_lines))
    # save seeds
    with open(os.path.join(out_dir,'seeds.json'),'w') as f:
        json.dump({'seeds': args.seeds}, f, indent=2)

    # AUC summary (simple のみ)
    simple_records = [r for r in records if r['algo']=='simple' and 'auc' in r]
    if simple_records:
        def fmt(v):
            return f"{v:.3f}" if isinstance(v,(int,float)) and v is not None else 'NA'
        auc_lines = [
            "# AUC Summary (Simple Mode)",
            "",
            "| seed | auc_raw | auc_neg_delta | auc_best_linear | best_a | best_b | terminal_steps |",
            "|------|---------|---------------|-----------------|--------|--------|----------------|"
        ]
        for r in simple_records:
            auc = r.get('auc', {}) or {}
            pc = r.get('phase_counts', {})
            auc_lines.append(
                f"| {r['seed']} | {fmt(auc.get('auc_raw'))} | {fmt(auc.get('auc_neg_delta'))} | {fmt(auc.get('auc_best_linear'))} | {auc.get('best_a','NA')} | {auc.get('best_b','NA')} | {pc.get('terminal',0)} |"
            )
        with open(os.path.join(out_dir,'auc_summary.md'),'w') as f:
            f.write("\n".join(auc_lines))

    # Effect size stats (simple vs baselines)
    metrics_keys = ['loop_redundancy','clipped_redundancy','unique_coverage','backtrack_rate','mean_geDIG','geDIG_low_frac']
    simples = [r for r in records if r['algo']=='simple']
    stats = {}
    for algo in ['dfs','random']:
        comp = [r for r in records if r['algo']==algo]
        algo_stats = {}
        for mk in metrics_keys:
            a_vals = [r.get(mk) for r in simples if r.get(mk) is not None]
            b_vals = [r.get(mk) for r in comp if r.get(mk) is not None]
            if a_vals and b_vals:
                res = effect_size_and_ci(a_vals, b_vals)
                algo_stats[mk] = res
        stats[algo] = algo_stats
    with open(os.path.join(out_dir,'stats.json'),'w') as f:
        json.dump({'effect_sizes': stats, 'metrics': metrics_keys}, f, indent=2)
    # Markdown table
    lines = ["# Effect Sizes (Simple vs Baseline)", "", "| baseline | metric | diff_mean | diff_CI | d | d_CI | n_simple | n_base |", "|----------|--------|-----------|---------|---|------|----------|--------|"]
    for algo, ms in stats.items():
        for mk, res in ms.items():
            lines.append(
                f"| {algo} | {mk} | {res['diff_mean']:.3f} | [{res['diff_ci_low']:.3f},{res['diff_ci_high']:.3f}] | {res['cohens_d']:.3f} | [{res['d_ci_low']:.3f},{res['d_ci_high']:.3f}] | {res['n_a']} | {res['n_b']} |"
            )
    with open(os.path.join(out_dir,'effect_sizes.md'),'w') as f:
        f.write("\n".join(lines))

    # Aggregated bootstrap stats (means per algo and diffs simple - baseline)
    if args.bootstrap_iterations > 0:
        rng = np.random.default_rng(42)
        algos_all = ['simple','dfs','random']
        agg = {}
        diffs = {}
        for mk in metrics_keys:
            # collect per algo arrays
            per_algo = {a: np.array([r[mk] for r in records if r['algo']==a and r.get(mk) is not None], dtype=float) for a in algos_all}
            # skip if simple missing
            if not per_algo['simple'].size:
                continue
            agg_mk = {}
            # bootstrap each algo mean
            B = args.bootstrap_iterations
            for a, arr in per_algo.items():
                if not arr.size:
                    continue
                means = []
                n = arr.size
                for _ in range(B):
                    sample = arr[rng.integers(0,n,size=n)]
                    means.append(sample.mean())
                means = np.sort(means)
                agg_mk[a] = {
                    'mean': float(arr.mean()),
                    'ci_low': float(means[int(0.025*B)]),
                    'ci_high': float(means[int(0.975*B)-1])
                }
            # diffs simple - other
            diff_mk = {}
            base_arr = per_algo['simple']
            if base_arr.size:
                for a in ['dfs','random']:
                    arr = per_algo[a]
                    if not arr.size:
                        continue
                    n_b = base_arr.size; n_o = arr.size
                    diff_samples = []
                    for _ in range(B):
                        sa = base_arr[rng.integers(0,n_b,size=n_b)]
                        sb = arr[rng.integers(0,n_o,size=n_o)]
                        diff_samples.append(sa.mean() - sb.mean())
                    diff_samples = np.sort(diff_samples)
                    diff_mk[a] = {
                        'mean_diff': float(base_arr.mean() - arr.mean()),
                        'ci_low': float(diff_samples[int(0.025*B)]),
                        'ci_high': float(diff_samples[int(0.975*B)-1])
                    }
            agg[mk] = agg_mk
            diffs[mk] = diff_mk
        with open(os.path.join(out_dir,'aggregated_stats.json'),'w') as f:
            json.dump({'metrics': metrics_keys, 'algo_means': agg, 'diffs_simple_minus': diffs}, f, indent=2)
    print(f'Baseline comparison written to {out_dir}')

if __name__=='__main__':
    main()
