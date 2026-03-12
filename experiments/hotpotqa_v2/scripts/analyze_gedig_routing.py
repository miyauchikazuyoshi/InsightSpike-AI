#!/usr/bin/env python3
"""Analyze geDIG routing experiment results.

Produces:
  1. geDIG value distribution (per-tier, per-domain)
  2. Tier distribution and nDCG comparison
  3. Component-level analysis (ΔH, ΔGED, ΔSP, Δβ₀)
  4. Correlation: geDIG features vs. nDCG
  5. Threshold sensitivity analysis

Usage::

    python experiments/hotpotqa_v2/scripts/analyze_gedig_routing.py \
        --results-dir experiments/hotpotqa_v2/results/v12_bright_gedig_routing

"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_records(results_dir: Path) -> list[dict]:
    """Load all result records from per-domain JSONL files."""
    records = []
    for f in sorted(results_dir.glob("*_results.jsonl")):
        with open(f) as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("error") is None:
                    records.append(r)
    return records


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def analyze_tier_distribution(records: list[dict]):
    """Analyze routing tier distribution and per-tier nDCG."""
    print_section("1. Tier Distribution & nDCG")

    tier_records: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        tier = r.get("routing_tier")
        if tier is not None:
            tier_records[tier].append(r)

    tier_names = {1: "DG (skip CoT)", 2: "Moderate CoT", 3: "AG (aggressive)"}

    print(f"\n  {'Tier':<20s} {'Count':>6s} {'%':>6s} {'nDCG@10':>10s} {'nDCG>0%':>8s} {'BM25_nDCG':>10s} {'Δ':>8s}")
    print(f"  {'-'*18:<20s} {'-'*6:>6s} {'-'*6:>6s} {'-'*10:>10s} {'-'*8:>8s} {'-'*10:>10s} {'-'*8:>8s}")

    total = len(records)
    for tier in sorted(tier_records):
        recs = tier_records[tier]
        n = len(recs)
        pct = 100.0 * n / total if total > 0 else 0
        ndcg_vals = [r["ndcg_10"] for r in recs]
        bm25_vals = [r.get("bm25_ndcg_10", 0) for r in recs]
        avg_ndcg = np.mean(ndcg_vals)
        avg_bm25 = np.mean(bm25_vals)
        ndcg_pos = 100.0 * sum(1 for v in ndcg_vals if v > 0) / n if n > 0 else 0
        delta = avg_ndcg - avg_bm25
        name = tier_names.get(tier, f"Tier {tier}")
        print(f"  {name:<20s} {n:>6d} {pct:>5.1f}% {avg_ndcg:>10.4f} {ndcg_pos:>7.1f}% {avg_bm25:>10.4f} {delta:>+8.4f}")

    # Overall
    if records:
        avg_all = np.mean([r["ndcg_10"] for r in records])
        bm25_all = np.mean([r.get("bm25_ndcg_10", 0) for r in records])
        pos_all = 100.0 * sum(1 for r in records if r["ndcg_10"] > 0) / len(records)
        print(f"  {'OVERALL':<20s} {len(records):>6d} {'':>6s} {avg_all:>10.4f} {pos_all:>7.1f}% {bm25_all:>10.4f} {avg_all-bm25_all:>+8.4f}")


def analyze_gedig_distribution(records: list[dict]):
    """Analyze geDIG value distribution."""
    print_section("2. geDIG Value Distribution")

    gedig_vals = [r["gedig_value"] for r in records if "gedig_value" in r]
    if not gedig_vals:
        print("  No geDIG values found.")
        return

    arr = np.array(gedig_vals)
    print(f"\n  Count:  {len(arr)}")
    print(f"  Mean:   {arr.mean():.4f}")
    print(f"  Std:    {arr.std():.4f}")
    print(f"  Min:    {arr.min():.4f}")
    print(f"  Q25:    {np.percentile(arr, 25):.4f}")
    print(f"  Median: {np.median(arr):.4f}")
    print(f"  Q75:    {np.percentile(arr, 75):.4f}")
    print(f"  Max:    {arr.max():.4f}")

    # Histogram buckets
    print(f"\n  Distribution:")
    bins = [-np.inf, -1.0, -0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5, 1.0, np.inf]
    hist, _ = np.histogram(arr, bins=bins)
    for i in range(len(hist)):
        lo = bins[i]
        hi = bins[i+1]
        lo_str = f"{lo:.1f}" if np.isfinite(lo) else "-∞"
        hi_str = f"{hi:.1f}" if np.isfinite(hi) else "+∞"
        bar = "█" * int(40 * hist[i] / max(hist.max(), 1))
        print(f"    [{lo_str:>5s}, {hi_str:>5s})  {hist[i]:>4d} {bar}")


def analyze_components(records: list[dict]):
    """Analyze geDIG component distributions (ΔH, ΔGED, ΔSP, Δβ₀)."""
    print_section("3. geDIG Component Analysis")

    components = {
        "gedig_ig_value": "ΔH (Info Gain)",
        "gedig_ged_value": "ΔGED (Graph Edit)",
        "gedig_delta_sp_rel": "ΔSP (Shortest Path)",
        "gedig_delta_betti_0": "Δβ₀ (Components)",
    }

    for key, name in components.items():
        vals = [r[key] for r in records if key in r]
        if not vals:
            continue
        arr = np.array(vals, dtype=float)
        print(f"\n  {name}:")
        print(f"    Mean={arr.mean():.4f}  Std={arr.std():.4f}  "
              f"Min={arr.min():.4f}  Median={np.median(arr):.4f}  Max={arr.max():.4f}")


def analyze_per_domain(records: list[dict]):
    """Per-domain breakdown."""
    print_section("4. Per-Domain Analysis")

    domain_recs: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        domain_recs[r["domain"]].append(r)

    tier_names = {1: "DG", 2: "MOD", 3: "AG"}

    for domain in sorted(domain_recs):
        recs = domain_recs[domain]
        ndcg_vals = [r["ndcg_10"] for r in recs]
        bm25_vals = [r.get("bm25_ndcg_10", 0) for r in recs]
        avg_ndcg = np.mean(ndcg_vals)
        avg_bm25 = np.mean(bm25_vals)
        pos_rate = 100.0 * sum(1 for v in ndcg_vals if v > 0) / len(recs)

        # Tier distribution
        tier_counts = defaultdict(int)
        for r in recs:
            t = r.get("routing_tier")
            if t is not None:
                tier_counts[t] += 1
        tier_str = ", ".join(f"T{t}({tier_names.get(t,'?')})={c}"
                            for t, c in sorted(tier_counts.items()))

        # geDIG
        gv = [r["gedig_value"] for r in recs if "gedig_value" in r]
        gedig_str = f"geDIG mean={np.mean(gv):.3f}" if gv else ""

        print(f"\n  {domain}:")
        print(f"    N={len(recs)}, nDCG={avg_ndcg:.4f}, BM25={avg_bm25:.4f}, "
              f"Δ={avg_ndcg-avg_bm25:+.4f}, nDCG>0={pos_rate:.1f}%")
        print(f"    Tiers: {tier_str}")
        if gedig_str:
            print(f"    {gedig_str}")

        # Episode stats
        ep_doc = [r.get("n_doc_episodes", 0) for r in recs]
        ep_q = [r.get("n_query_episodes", 0) for r in recs]
        ep_cross = [r.get("n_episode_cross_edges", 0) for r in recs]
        if any(e > 0 for e in ep_doc):
            print(f"    Episodes: doc avg={np.mean(ep_doc):.0f}, "
                  f"query avg={np.mean(ep_q):.1f}, "
                  f"cross-edges avg={np.mean(ep_cross):.1f}")


def analyze_correlation(records: list[dict]):
    """Correlate geDIG features with nDCG improvement."""
    print_section("5. Correlation: geDIG Features vs nDCG Delta")

    features = {
        "gedig_value": "geDIG",
        "gedig_ig_value": "ΔH",
        "gedig_ged_value": "ΔGED",
        "gedig_delta_sp_rel": "ΔSP",
        "gedig_delta_betti_0": "Δβ₀",
    }

    ndcg_deltas = []
    feature_vals: dict[str, list[float]] = {k: [] for k in features}

    for r in records:
        delta = r.get("ndcg_10", 0) - r.get("bm25_ndcg_10", 0)
        if all(k in r for k in features):
            ndcg_deltas.append(delta)
            for k in features:
                feature_vals[k].append(float(r[k]))

    if len(ndcg_deltas) < 5:
        print("  Not enough data for correlation analysis.")
        return

    y = np.array(ndcg_deltas)
    print(f"\n  N={len(y)} queries with full geDIG features")
    print(f"  nDCG delta: mean={y.mean():.4f}, std={y.std():.4f}")

    print(f"\n  {'Feature':<10s} {'Pearson r':>10s} {'Interpretation':<30s}")
    print(f"  {'-'*10:<10s} {'-'*10:>10s} {'-'*30:<30s}")

    for key, name in features.items():
        x = np.array(feature_vals[key])
        if x.std() < 1e-10:
            print(f"  {name:<10s} {'N/A':>10s} constant feature")
            continue
        r_val = np.corrcoef(x, y)[0, 1]
        if abs(r_val) > 0.5:
            interp = "STRONG"
        elif abs(r_val) > 0.3:
            interp = "moderate"
        elif abs(r_val) > 0.1:
            interp = "weak"
        else:
            interp = "negligible"
        sign = "+" if r_val > 0 else "-"
        print(f"  {name:<10s} {r_val:>+10.4f} {interp} ({sign} geDIG → {sign} nDCG)")


def analyze_threshold_sensitivity(records: list[dict]):
    """What-if analysis for different τ_dg / τ_ag thresholds."""
    print_section("6. Threshold Sensitivity (What-If)")

    gedig_recs = [r for r in records if "gedig_value" in r]
    if not gedig_recs:
        print("  No geDIG data available.")
        return

    # Simulate different thresholds
    tau_dg_options = [-0.5, -0.3, -0.1, 0.0]
    tau_ag_options = [0.0, 0.1, 0.3, 0.5]

    print(f"\n  {'τ_dg':>6s} {'τ_ag':>6s} {'T1(DG)':>8s} {'T2(MOD)':>8s} {'T3(AG)':>8s}"
          f" {'nDCG(T1)':>10s} {'nDCG(T2)':>10s} {'nDCG(T3)':>10s} {'nDCG_all':>10s}")
    print(f"  {'-'*6:>6s} {'-'*6:>6s} {'-'*8:>8s} {'-'*8:>8s} {'-'*8:>8s}"
          f" {'-'*10:>10s} {'-'*10:>10s} {'-'*10:>10s} {'-'*10:>10s}")

    for tau_dg in tau_dg_options:
        for tau_ag in tau_ag_options:
            if tau_dg >= tau_ag:
                continue
            t1, t2, t3 = [], [], []
            for r in gedig_recs:
                gv = r["gedig_value"]
                db0 = r.get("gedig_delta_betti_0", 0)
                # Simulate tier assignment
                if gv < tau_dg:
                    t1.append(r)
                elif gv > tau_ag:
                    t3.append(r)
                elif db0 < -2:
                    t1.append(r)
                elif db0 > 1:
                    t3.append(r)
                else:
                    t2.append(r)

            n1, n2, n3 = len(t1), len(t2), len(t3)
            m1 = np.mean([r["ndcg_10"] for r in t1]) if t1 else 0
            m2 = np.mean([r["ndcg_10"] for r in t2]) if t2 else 0
            m3 = np.mean([r["ndcg_10"] for r in t3]) if t3 else 0
            m_all = np.mean([r["ndcg_10"] for r in gedig_recs])
            print(f"  {tau_dg:>6.1f} {tau_ag:>6.1f} {n1:>8d} {n2:>8d} {n3:>8d}"
                  f" {m1:>10.4f} {m2:>10.4f} {m3:>10.4f} {m_all:>10.4f}")


def analyze_computation_time(records: list[dict]):
    """geDIG computation time analysis."""
    print_section("7. Computation Time")

    times = [r.get("gedig_computation_ms", 0) for r in records if "gedig_computation_ms" in r]
    if not times:
        print("  No timing data.")
        return

    arr = np.array(times)
    print(f"\n  geDIG computation time (ms):")
    print(f"    Mean:   {arr.mean():.1f}")
    print(f"    Median: {np.median(arr):.1f}")
    print(f"    P95:    {np.percentile(arr, 95):.1f}")
    print(f"    Max:    {arr.max():.1f}")

    latencies = [r.get("latency_ms", 0) for r in records if "latency_ms" in r]
    if latencies:
        lat_arr = np.array(latencies)
        pct = 100.0 * arr.mean() / lat_arr.mean() if lat_arr.mean() > 0 else 0
        print(f"\n  Overall query latency (ms):")
        print(f"    Mean:   {lat_arr.mean():.0f}")
        print(f"    geDIG fraction: {pct:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze geDIG routing results"
    )
    parser.add_argument("--results-dir", required=True,
                        help="Directory with *_results.jsonl files")
    parser.add_argument("--output", default=None,
                        help="Output JSON file for analysis summary")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"ERROR: {results_dir} not found")
        sys.exit(1)

    records = load_records(results_dir)
    print(f"Loaded {len(records)} valid records from {results_dir}")

    if not records:
        print("No records to analyze.")
        sys.exit(0)

    analyze_tier_distribution(records)
    analyze_gedig_distribution(records)
    analyze_components(records)
    analyze_per_domain(records)
    analyze_correlation(records)
    analyze_threshold_sensitivity(records)
    analyze_computation_time(records)

    # Save analysis summary if requested
    if args.output:
        gedig_vals = [r["gedig_value"] for r in records if "gedig_value" in r]
        tier_counts = defaultdict(int)
        for r in records:
            t = r.get("routing_tier")
            if t is not None:
                tier_counts[t] += 1

        summary = {
            "n_records": len(records),
            "overall_ndcg_10": round(np.mean([r["ndcg_10"] for r in records]), 4),
            "overall_bm25_ndcg_10": round(np.mean([r.get("bm25_ndcg_10", 0) for r in records]), 4),
            "gedig_stats": {
                "mean": round(np.mean(gedig_vals), 4) if gedig_vals else None,
                "std": round(np.std(gedig_vals), 4) if gedig_vals else None,
                "min": round(min(gedig_vals), 4) if gedig_vals else None,
                "max": round(max(gedig_vals), 4) if gedig_vals else None,
            },
            "tier_distribution": dict(tier_counts),
        }
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nAnalysis summary saved to {args.output}")


if __name__ == "__main__":
    main()
