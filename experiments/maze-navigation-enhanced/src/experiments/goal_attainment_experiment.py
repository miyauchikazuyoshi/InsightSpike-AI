#!/usr/bin/env python3
"""Goal attainment experiment across variants & parameter configs.

Runs run_simple with different (strategy, thresholds, ANN settings) until either goal reached
or max_steps exhausted. Escalates max_steps schedule for harder variants.

Outputs CSV + Markdown summary with success rates, mean/median steps-to-goal (successful only),
and attempts that timed out.
"""
from __future__ import annotations
import os, sys, json, argparse, datetime, statistics
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from baseline_vs_simple_plot import run_simple  # type: ignore


@dataclass
class Config:
    variant: str
    strategy: str
    ann: bool
    gedig_threshold: float
    backtrack_threshold: float
    global_recall: bool
    recall_threshold: float
    flush: bool
    ann_upgrade_threshold: int

def run_with_escalation(cfg: Config, seed: int, step_schedule) -> dict:
    """Try escalating max_steps until goal or schedule exhausted."""
    result_record = {
        'seed': seed,
        'config': asdict(cfg),
        'attempts': [],
        'goal_reached': False,
        'final_steps': None,
    }
    for steps in step_schedule:
        r = run_simple(
            cfg.variant,
            seed,
            steps,
            temperature=0.1,
            gedig_threshold=cfg.gedig_threshold,
            backtrack_threshold=cfg.backtrack_threshold,
            global_recall=cfg.global_recall,
            recall_threshold=cfg.recall_threshold,
            wiring_strategy_override=cfg.strategy,
            use_vector_index=(cfg.strategy=='query'),
            enable_flush=cfg.flush,
            flush_interval=200,
            max_in_memory=8000,
            max_in_memory_positions=1500,
            ann_backend=('hnsw' if cfg.ann else None),
            ann_upgrade_threshold=cfg.ann_upgrade_threshold,
            catalog_compaction_on_close=True,
            include_events=True
        )
        attempt = {
            'max_steps': steps,
            'goal': r.get('goal_reached'),
            'steps_taken': r.get('steps'),
            'wiring_mean_ms': r.get('timing_wiring_mean_ms'),
            'wiring_p95_ms': r.get('timing_wiring_p95_ms'),
            'ann_index_elements': r.get('ann_index_elements'),
            'event_counts': r.get('event_counts', {}),
        }
        result_record['attempts'].append(attempt)
        if r.get('goal_reached'):
            result_record['goal_reached'] = True
            result_record['final_steps'] = r.get('steps')
            break
    return result_record


def summarize(records: list[dict]) -> dict:
    successes = [r for r in records if r['goal_reached']]
    succ_steps = [r['final_steps'] for r in successes if r['final_steps']]
    return {
        'runs': len(records),
        'successes': len(successes),
        'success_rate': (len(successes)/len(records)) if records else 0.0,
        'mean_steps_success': (statistics.fmean(succ_steps) if succ_steps else None),
        'median_steps_success': (statistics.median(succ_steps) if succ_steps else None),
        'min_steps_success': (min(succ_steps) if succ_steps else None),
        'max_steps_success': (max(succ_steps) if succ_steps else None),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--variants', nargs='+', default=['complex','ultra50hd'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[101,202,303])
    ap.add_argument('--out_dir', type=str, default='experiments/maze-navigation-enhanced/results/goal_attainment')
    ap.add_argument('--escalation', type=int, nargs='+', default=[1000,2000,3000,5000], help='Escalating step ceilings tried sequentially')
    args=ap.parse_args()

    # Parameter grid (tune minimal combos)
    param_grid = []
    for variant in args.variants:
        if variant == 'complex':
            step_sched = args.escalation[:3]  # shorter schedule likely enough
        else:
            step_sched = args.escalation
        for strategy in ['simple','query']:
            for ann in [False, True]:
                # Focused thresholds; lower gedig to encourage exploration for large mazes
                gedig = 0.28 if variant.startswith('ultra') else 0.30
                bt = -0.15 if variant.startswith('ultra') else -0.20
                param_grid.append((variant, step_sched, Config(
                    variant=variant,
                    strategy=strategy,
                    ann=ann,
                    gedig_threshold=gedig,
                    backtrack_threshold=bt,
                    global_recall=True,
                    recall_threshold=0.02 if variant.startswith('ultra') else 0.01,
                    flush=True,
                    ann_upgrade_threshold=180 if ann else 10_000_000,
                )))

    ts=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_run_dir=os.path.join(args.out_dir, f'goal_attain_{ts}')
    os.makedirs(out_run_dir, exist_ok=True)

    all_results = []
    groups = {}
    for variant, step_sched, cfg in param_grid:
        for seed in args.seeds:
            rec = run_with_escalation(cfg, seed, step_sched)
            all_results.append(rec)
            key = (cfg.variant, cfg.strategy, cfg.ann)
            groups.setdefault(key, []).append(rec)

    # Persist raw
    with open(os.path.join(out_run_dir,'raw_results.json'),'w') as f:
        json.dump(all_results, f, indent=2)

    # Build summary table
    summary_rows = []
    for (variant,strategy,ann), recs in groups.items():
        summ = summarize(recs)
        row = {
            'variant': variant,
            'strategy': strategy,
            'ann': ann,
            **summ
        }
        summary_rows.append(row)
    # Write CSV
    import csv
    csv_path = os.path.join(out_run_dir,'summary.csv')
    fieldnames = ['variant','strategy','ann','runs','successes','success_rate','mean_steps_success','median_steps_success','min_steps_success','max_steps_success']
    with open(csv_path,'w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(summary_rows)

    # Markdown
    md_lines = ["# Goal Attainment Summary","", "| variant | strategy | ann | runs | successes | success_rate | mean_steps | median_steps | min_steps | max_steps |", "|---------|----------|-----|------|-----------|-------------|-----------|------------|----------|----------|"]
    for r in summary_rows:
        md_lines.append(f"| {r['variant']} | {r['strategy']} | {int(r['ann'])} | {r['runs']} | {r['successes']} | {r['success_rate']:.2f} | {r['mean_steps_success'] or 'NA'} | {r['median_steps_success'] or 'NA'} | {r['min_steps_success'] or 'NA'} | {r['max_steps_success'] or 'NA'} |")
    with open(os.path.join(out_run_dir,'summary.md'),'w') as f:
        f.write("\n".join(md_lines))

    print(f"Goal attainment results written to {out_run_dir}")


if __name__ == '__main__':
    # Provide globals expected by run_simple
    import builtins
    builtins.GLOBAL_VERBOSITY = 0
    builtins.GLOBAL_PROGRESS_INTERVAL = 200
    main()
