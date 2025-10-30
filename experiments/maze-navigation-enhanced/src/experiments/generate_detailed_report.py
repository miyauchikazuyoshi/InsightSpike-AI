#!/usr/bin/env python3
"""Generate detailed CSV/Markdown reports including event analysis for multiple variants.
Usage:
  python -m experiments.generate_detailed_report --variants ultra50 ultra50hd --seeds 101 202 --steps 600
"""
from __future__ import annotations
import argparse, os, sys, json, datetime, csv
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiments.baseline_vs_simple_plot import run_simple, run_baseline  # type: ignore

FOCUS_EVENTS = ["ann_upgrade","ann_upgrade_failed","flush_eviction","rehydration","catalog_compact"]
METRICS = ["loop_redundancy","clipped_redundancy","unique_coverage","backtrack_rate","path_length"]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--variants', nargs='+', required=True)
    ap.add_argument('--seeds', nargs='+', type=int, default=[101,202,303])
    ap.add_argument('--steps', type=int, default=1200)
    ap.add_argument('--ann-upgrade-threshold', type=int, default=250)
    ap.add_argument('--out-dir', type=str, default='experiments/maze-navigation-enhanced/results/detailed')
    ap.add_argument('--flush', action='store_true')
    ap.add_argument('--enable-ann', action='store_true')
    ap.add_argument('--flush-interval', type=int, default=60)
    ap.add_argument('--max-in-memory', type=int, default=6000)
    ap.add_argument('--max-in-memory-positions', type=int, default=2000)
    ap.add_argument('--progress-interval', type=int, default=120)
    args=ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    timestamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_id=f"dreport_{timestamp}"
    run_dir=os.path.join(args.out_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    all_records=[]
    for variant in args.variants:
        for seed in args.seeds:
            # Setup globals expected by run_simple
            import experiments.baseline_vs_simple_plot as mod  # type: ignore
            mod.GLOBAL_VERBOSITY = 0
            mod.GLOBAL_PROGRESS_INTERVAL = args.progress_interval
            simple = run_simple(variant, seed, args.steps, 0.1, 0.3, -0.2,
                                 wiring_strategy_override='query', use_vector_index=True,
                                 enable_flush=args.flush, flush_interval=args.flush_interval,
                                 max_in_memory=args.max_in_memory, max_in_memory_positions=args.max_in_memory_positions,
                                 ann_upgrade_threshold=args.ann_upgrade_threshold,
                                 ann_backend=('hnsw' if args.enable_ann else None),
                                 include_events=True)
            # ensure variant field present
            if 'variant' not in simple:
                simple['variant'] = variant
            all_records.append(simple)
            # baselines
            all_records.append(run_baseline('dfs', variant, seed, args.steps))
            all_records.append(run_baseline('random', variant, seed, args.steps))

    # CSV export
    csv_path=os.path.join(run_dir,'metrics.csv')
    fieldnames=[ 'variant','algo','seed', *METRICS ]
    with open(csv_path,'w',newline='') as f:
        w=csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_records:
            row={'variant':r.get('variant') or 'N/A','algo':r.get('algo'),'seed':r.get('seed')}
            for m in METRICS:
                row[m]=r.get(m)
            w.writerow(row)

    # Markdown summary
    md_lines=[f"# Detailed Report {run_id}", '', f"Generated: {timestamp}", '']
    # Aggregates per variant/algo
    for variant in sorted(set(v for v in args.variants)):
        md_lines.append(f"## Variant: {variant}")
        subset=[r for r in all_records if (r.get('variant')==variant or variant in r.get('algo',''))]  # fallback simple run doesn't store variant key
        for algo in ['simple','dfs','random']:
            rs=[r for r in all_records if r.get('algo')==algo and (r.get('variant')==variant or variant in r.get('algo','') )]
            if not rs:
                continue
            md_lines.append(f"### Algo: {algo}")
            for m in METRICS:
                vals=[x.get(m) for x in rs if x.get(m) is not None]
                if not vals:
                    continue
                md_lines.append(f"- {m}: mean={np.mean(vals):.3f} std={np.std(vals):.3f} min={np.min(vals):.3f} max={np.max(vals):.3f}")
        md_lines.append('')
    # Event focus (simple only)
    simples=[r for r in all_records if r.get('algo')=='simple']
    md_lines.append('## Event Counts (simple)')
    agg={}
    for r in simples:
        for k,v in (r.get('event_counts') or {}).items():
            agg[k]=agg.get(k,0)+v
    for k in sorted(agg):
        md_lines.append(f"- {k}: {agg[k]}")
    # Focused raw events
    focused=[]
    for r in simples:
        for ev in (r.get('events') or []):
            if ev.get('type') in FOCUS_EVENTS:
                focused.append(ev)
    md_lines.append('\n## Sample Focus Events (first 50)')
    for ev in focused[:50]:
        md_lines.append(f"- step={ev.get('step')} type={ev.get('type')} msg={ev.get('message')}")
    with open(os.path.join(run_dir,'summary.md'),'w') as f:
        f.write('\n'.join(md_lines))

    with open(os.path.join(run_dir,'records.json'),'w') as f:
        json.dump({'records':all_records,'generated_at':timestamp}, f, indent=2)
    print('Report written to', run_dir)

if __name__=='__main__':
    main()
