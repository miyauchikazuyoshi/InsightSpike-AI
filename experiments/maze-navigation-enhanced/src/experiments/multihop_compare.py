#!/usr/bin/env python3
"""Generate comparison table between (A) global single-hop (no escalation) and (B) escalation+multihop (max_hop>=1) evaluation.

Outputs CSV with per-step aggregated statistics:
  variant, seed, mode, steps, mean_score, median_score, low_fraction(<threshold),
  escalated_fraction, dead_end_detects, shortcut_flags, multihop_usage_rate
If structural trace exported (gedig_structural), derives hop1_gain_delta (mean hop1-hop0 where available).

Usage (example):
  python multihop_compare.py --variant ultra50 --seeds 101 202 --max_steps 600 --threshold 0.05 

"""
from __future__ import annotations
import os, sys, json, argparse, statistics, csv
from typing import List, Dict, Any

# Reuse run_simple from baseline_vs_simple_plot
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
from baseline_vs_simple_plot import run_simple  # type: ignore

def summarize(record: Dict[str, Any], threshold: float, mode: str) -> Dict[str, Any]:
    hist: List[float] = record.get('gedig_history', []) or []
    structural = record.get('gedig_structural') or []
    steps = len(hist)
    if steps == 0:
        return {
            'mode': mode, 'steps': 0, 'mean_score': None, 'median_score': None,
            'low_fraction': None, 'escalated_fraction': None, 'dead_end_fraction': None,
            'shortcut_fraction': None, 'multihop_usage_rate': None, 'hop1_gain_delta_mean': None
        }
    low_frac = sum(1 for v in hist if v < threshold)/steps
    # Structural derived flags
    escalated = [r for r in structural if r.get('escalated')]
    dead_ends = [r for r in structural if r.get('dead_end')]
    shortcuts = [r for r in structural if r.get('shortcut')]
    # multihop_scores present => escalated and evaluated
    multihop_used = [r for r in structural if r.get('multihop')]
    hop1_deltas = []
    for r in structural:
        mh = r.get('multihop') or {}
        if 1 in mh:
            hop1_deltas.append(mh[1] - (mh.get(0) or r.get('value')))
    return {
        'mode': mode,
        'steps': steps,
        'mean_score': sum(hist)/steps,
        'median_score': statistics.median(hist),
        'low_fraction': low_frac,
        'escalated_fraction': (len(escalated)/steps) if steps else 0.0,
        'dead_end_fraction': (len(dead_ends)/steps) if steps else 0.0,
        'shortcut_fraction': (len(shortcuts)/steps) if steps else 0.0,
        'multihop_usage_rate': (len(multihop_used)/steps) if steps else 0.0,
        'hop1_gain_delta_mean': (sum(hop1_deltas)/len(hop1_deltas)) if hop1_deltas else None
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True)
    ap.add_argument('--seeds', type=int, nargs='+', required=True)
    ap.add_argument('--max_steps', type=int, default=600)
    ap.add_argument('--gedig_threshold', type=float, default=0.3)
    ap.add_argument('--backtrack_threshold', type=float, default=-0.2)
    ap.add_argument('--temperature', type=float, default=0.1)
    ap.add_argument('--threshold', type=float, default=0.05, help='Low score fraction threshold')
    ap.add_argument('--out', type=str, default='multihop_compare.csv')
    args = ap.parse_args()

    rows: List[Dict[str, Any]] = []
    for seed in args.seeds:
        # Mode A: single-hop (escalation disabled)
        rec_single = run_simple(args.variant, seed, args.max_steps, args.temperature,
                                args.gedig_threshold, args.backtrack_threshold,
                                include_trace=True, include_structural=True,
                                use_escalation=False)
        sum_single = summarize(rec_single, args.threshold, mode='single')
        sum_single.update({'variant': args.variant, 'seed': seed})
        rows.append(sum_single)
        # Mode B: escalation (potential multi-hop) with default dynamic threshold
        rec_multi = run_simple(args.variant, seed, args.max_steps, args.temperature,
                               args.gedig_threshold, args.backtrack_threshold,
                               include_trace=True, include_structural=True,
                               use_escalation=True, escalation_threshold='dynamic')
        sum_multi = summarize(rec_multi, args.threshold, mode='escalation')
        sum_multi.update({'variant': args.variant, 'seed': seed})
        rows.append(sum_multi)

    # Write CSV
    fieldnames = ['variant','seed','mode','steps','mean_score','median_score','low_fraction','escalated_fraction','dead_end_fraction','shortcut_fraction','multihop_usage_rate','hop1_gain_delta_mean']
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)
    print(f"[multihop_compare] wrote {args.out} ({len(rows)} rows)")

    # Also print a small summary grouped by mode
    by_mode: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_mode.setdefault(r['mode'], []).append(r)
    for mode, rs in by_mode.items():
        print(f"Mode: {mode}")
        print(f"  seeds: {len(rs)} avg_mean_score={sum(r['mean_score'] for r in rs)/len(rs):.4f} avg_low_fraction={sum(r['low_fraction'] for r in rs)/len(rs):.3f} avg_multihop_use={sum(r['multihop_usage_rate'] for r in rs)/len(rs):.3f}")

if __name__ == '__main__':
    main()
