#!/usr/bin/env python3
"""Sweep backtrack thresholds to elicit non-zero trigger behavior.

Runs a single maze variant (complex/ultra) over multiple thresholds & seeds.
Collects trigger counts, trigger rate, first trigger step, loop redundancy.

Output directory: results/maze_report/sweep_<variant>_<timestamp>/
  - sweep.json : raw per (seed, threshold) records
  - summary.md : table + simple recommendations

Example:
  python backtrack_threshold_sweep.py --variant complex --thresholds -0.3 -0.25 -0.2 -0.15 -0.1 --seeds 101 202 303 --max_steps 1200
"""
from __future__ import annotations
import os, sys, json, argparse, datetime
import numpy as np
from statistics import fmean
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from maze_layouts import (
    create_complex_maze, create_ultra_maze,
    COMPLEX_DEFAULT_START, COMPLEX_DEFAULT_GOAL,
    ULTRA_DEFAULT_START, ULTRA_DEFAULT_GOAL
)
from navigation.maze_navigator import MazeNavigator  # type: ignore
from generate_complex_maze_report import loop_erased_length


def run_once(variant: str, seed: int, threshold: float, max_steps: int, temperature: float, gedig_threshold: float):
    np.random.seed(seed)
    if variant == 'complex':
        maze = create_complex_maze(); start, goal = COMPLEX_DEFAULT_START, COMPLEX_DEFAULT_GOAL
    else:
        maze = create_ultra_maze(); start, goal = ULTRA_DEFAULT_START, ULTRA_DEFAULT_GOAL
    weights = np.array([1.0,1.0,0.0,0.0,3.0,2.0,0.1,0.0])
    nav = MazeNavigator(maze, start, goal, weights=weights, temperature=temperature,
                        gedig_threshold=gedig_threshold, backtrack_threshold=threshold,
                        wiring_strategy='simple', simple_mode=True, backtrack_debounce=True)
    nav.run(max_steps=max_steps)
    stats = nav.get_statistics()
    events = nav.event_log
    triggers = [e for e in events if e['type']=='backtrack_trigger']
    steps = stats.get('steps')
    first_trigger = triggers[0]['step'] if triggers else None
    loop_len = loop_erased_length(nav.path)
    redundancy = (len(nav.path)/loop_len) if loop_len else None
    sm = stats.get('simple_mode', {})
    return {
        'seed': seed,
        'threshold': threshold,
        'steps': steps,
        'triggers': len(triggers),
        'trigger_rate': sm.get('backtrack_trigger_rate'),
        'first_trigger_step': first_trigger,
        'loop_redundancy': redundancy,
        'unique_positions': stats.get('unique_positions'),
    }


def summarize(records):
    # group by threshold
    by_th = {}
    for r in records:
        by_th.setdefault(r['threshold'], []).append(r)
    summary = []
    for th, rs in sorted(by_th.items()):
        trigger_rates = [x['trigger_rate'] for x in rs]
        redundancies = [x['loop_redundancy'] for x in rs if x['loop_redundancy'] is not None]
        firsts = [x['first_trigger_step'] for x in rs if x['first_trigger_step'] is not None]
        summary.append({
            'threshold': th,
            'mean_trigger_rate': fmean(trigger_rates) if trigger_rates else 0.0,
            'seeds_with_trigger': sum(1 for x in rs if x['triggers']>0),
            'mean_loop_redundancy': fmean(redundancies) if redundancies else None,
            'mean_first_trigger_step': fmean(firsts) if firsts else None,
        })
    return summary


def write_md(path: str, variant: str, args, records, summary):
    with open(path,'w') as f:
        f.write(f"# Backtrack Threshold Sweep ({variant})\n\n")
        f.write('## Parameters\n')
        f.write(f"- thresholds: {args.thresholds}\n")
        f.write(f"- seeds: {args.seeds}\n")
        f.write(f"- max_steps: {args.max_steps}\n")
        f.write('\n## Per Run\n')
        f.write('| seed | thr | steps | triggers | trig_rate | first_step | loop_red |\n')
        f.write('|------|-----|-------|----------|-----------|-----------|---------|\n')
        for r in records:
            f.write(f"| {r['seed']} | {r['threshold']:.3f} | {r['steps']} | {r['triggers']} | {r['trigger_rate']:.3f} | {r['first_trigger_step'] if r['first_trigger_step'] is not None else '-'} | {r['loop_redundancy']:.2f} |\n")
        f.write('\n## Summary by Threshold\n')
        f.write('| thr | mean_trigger_rate | seeds_with_trigger | mean_first_trigger | mean_loop_red |\n')
        f.write('|-----|-------------------|--------------------|-------------------|--------------|\n')
        for s in summary:
            mfr = '-' if s['mean_first_trigger_step'] is None else f"{s['mean_first_trigger_step']:.1f}"
            mlr = '-' if s['mean_loop_redundancy'] is None else f"{s['mean_loop_redundancy']:.2f}"
            f.write(f"| {s['threshold']:.3f} | {s['mean_trigger_rate']:.3f} | {s['seeds_with_trigger']} | {mfr} | {mlr} |\n")
        f.write('\n## Recommendation\n')
        # naive recommendation: choose threshold giving non-zero triggers for >=50% seeds with modest redundancy (< median of triggered set)
        triggered = [s for s in summary if s['seeds_with_trigger']>0]
        if triggered:
            median_redundancy = np.median([s['mean_loop_redundancy'] for s in triggered if s['mean_loop_redundancy'] is not None]) if [s for s in triggered if s['mean_loop_redundancy'] is not None] else None
            candidates = [s for s in triggered if s['seeds_with_trigger'] >= (len(args.seeds)//2)]
            if median_redundancy is not None:
                candidates = [c for c in candidates if c['mean_loop_redundancy'] is None or c['mean_loop_redundancy'] <= median_redundancy]
            if candidates:
                best = sorted(candidates, key=lambda x: x['mean_trigger_rate'])[-1]
                f.write(f"Suggested threshold: {best['threshold']:.3f} (balances trigger activity and redundancy)\n")
            else:
                f.write('No clear candidate threshold meets selection criteria.\n')
        else:
            f.write('No triggers observed at provided thresholds; consider raising threshold (less negative or positive).\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', choices=['complex','ultra'], default='complex')
    ap.add_argument('--thresholds', type=float, nargs='+', required=True)
    ap.add_argument('--seeds', type=int, nargs='+', default=[101,202,303])
    ap.add_argument('--max_steps', type=int, default=1500)
    ap.add_argument('--temperature', type=float, default=0.1)
    ap.add_argument('--gedig_threshold', type=float, default=0.3)
    args = ap.parse_args()

    records=[]
    for th in args.thresholds:
        for s in args.seeds:
            r = run_once(args.variant, s, th, args.max_steps, args.temperature, args.gedig_threshold)
            records.append(r)
            print(f"thr {th:.3f} seed {s} triggers={r['triggers']} rate={r['trigger_rate']:.3f} loop_red={r['loop_redundancy']:.2f}")
    summary = summarize(records)

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join('results','maze_report', f"sweep_{args.variant}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir,'sweep.json'),'w') as f:
        json.dump({'params': vars(args), 'records': records, 'summary': summary}, f, indent=2)
    write_md(os.path.join(out_dir,'summary.md'), args.variant, args, records, summary)
    print(f"Sweep written to {out_dir}")

if __name__ == '__main__':
    main()
