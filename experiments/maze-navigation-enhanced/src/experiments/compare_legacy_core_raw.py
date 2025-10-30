#!/usr/bin/env python3
"""Compare legacy vs core_raw geDIG score distributions for quick drift diagnostics.

Runs both modes with identical seeds & params, produces JSON + simple text report.
Usage:
  python compare_legacy_core_raw.py --variant ultra50 --seeds 1 2 3 --max_steps 150 \
     --outdir experiments/maze-navigation-enhanced/results/diff_reports
"""
from __future__ import annotations
import os, sys, json, argparse, statistics, math
from typing import List, Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
from baseline_vs_simple_plot import run_simple  # type: ignore


def run_mode(variant: str, seed: int, max_steps: int, mode: str, force_multihop: bool, escalation_threshold: float) -> Dict[str, Any]:
    rec = run_simple(variant, seed, max_steps, temperature=0.1,
                     gedig_threshold=0.3, backtrack_threshold=-0.2,
                     include_trace=True, include_structural=True,
                     use_escalation=True, escalation_threshold=escalation_threshold,
                     force_multihop=force_multihop, max_hops=2,
                     gedig_mode=mode)
    return rec


def summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    vs = [v for v in values if v == v]
    if not vs:
        return {}
    vs_sorted = sorted(vs)
    def pct(p):
        k = (len(vs_sorted)-1) * p
        f = math.floor(k); c = min(len(vs_sorted)-1, math.ceil(k))
        if f == c:
            return vs_sorted[int(k)]
        return vs_sorted[f] + (vs_sorted[c]-vs_sorted[f]) * (k - f)
    return {
        'count': len(vs_sorted),
        'mean': statistics.fmean(vs_sorted),
        'stdev': statistics.pstdev(vs_sorted),
        'min': vs_sorted[0],
        'p25': pct(0.25),
        'median': pct(0.5),
        'p75': pct(0.75),
        'max': vs_sorted[-1]
    }


def ks_stat(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return float('nan')
    a_sorted = sorted(a); b_sorted = sorted(b)
    i=j=0; na=len(a_sorted); nb=len(b_sorted)
    d=0.0
    while i<na and j<nb:
        av=a_sorted[i]; bv=b_sorted[j]
        if av<=bv:
            i+=1
        if bv<=av:
            j+=1
        d=max(d, abs(i/na - j/nb))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', default='ultra50')
    ap.add_argument('--seeds', type=int, nargs='+', required=True)
    ap.add_argument('--max_steps', type=int, default=200)
    ap.add_argument('--force_multihop', action='store_true')
    ap.add_argument('--escalation_threshold', type=float, default=1.0)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--ks_warn', type=float, default=0.85, help='Warn if KS >= this value')
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    reports = []
    legacy_scores_all: List[float] = []
    raw_scores_all: List[float] = []

    for seed in args.seeds:
        legacy = run_mode(args.variant, seed, args.max_steps, 'legacy', args.force_multihop, args.escalation_threshold)
        core_raw = run_mode(args.variant, seed, args.max_steps, 'core_raw', args.force_multihop, args.escalation_threshold)
        legacy_scores = legacy.get('gedig_history', [])
        raw_scores = core_raw.get('gedig_history', [])
        legacy_scores_all.extend(legacy_scores)
        raw_scores_all.extend(raw_scores)
        reports.append({
            'seed': seed,
            'legacy_summary': summarize(legacy_scores),
            'core_raw_summary': summarize(raw_scores),
            'count_legacy': len(legacy_scores),
            'count_core_raw': len(raw_scores)
        })

    ks = ks_stat(legacy_scores_all, raw_scores_all)
    agg = {
        'variant': args.variant,
        'seeds': args.seeds,
        'legacy_agg': summarize(legacy_scores_all),
        'core_raw_agg': summarize(raw_scores_all),
        'ks_statistic': ks
    }
    out = {'per_seed': reports, 'aggregate': agg}
    out_path = os.path.join(args.outdir, f"diff_{args.variant}_seeds{'-'.join(map(str,args.seeds))}.json")
    with open(out_path,'w') as f:
        json.dump(out, f, indent=2)
    warn = ' (WARN: drift high)' if (not math.isnan(ks) and ks >= args.ks_warn) else ''
    print(f"[compare] Wrote diff report -> {out_path}\nKS={ks:.4f}{warn}")

if __name__ == '__main__':
    main()
