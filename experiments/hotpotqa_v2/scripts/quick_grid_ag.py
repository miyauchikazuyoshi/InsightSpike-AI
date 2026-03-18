#!/usr/bin/env python3
"""Quick 10q grid search for AG parameters (Spec Q.1).

Runs a small grid on 10 queries to find promising parameter combinations,
then full 50q tests only on the top candidates.

Usage:
    PYTHONPATH=experiments/hotpotqa_v2/src .venv/bin/python3 \
        experiments/hotpotqa_v2/scripts/quick_grid_ag.py
"""
import json
import os
import subprocess
import sys
from itertools import product

# Grid parameters
THRESHOLDS = [0.1, 0.15, 0.2, 0.3, 0.4]
MAX_KS = [5, 10, 20, 50, None]  # None = unlimited
LIMIT = 10  # quick scan

PYTHON = ".venv/bin/python3"
SCRIPT = "experiments/hotpotqa_v2/scripts/run_bright.py"
RESULT_BASE = "experiments/hotpotqa_v2/results/v21_specq1_grid"

COMMON_ARGS = [
    "--mode", "cot_retrieval",
    "--domains", "biology",
    "--scoring-mode", "gedig_refine",
    "--rerank-alpha", "0.1",
    "--graph-top-k", "50",
    "--token-graph",
    "--token-graph-walk-score",
    "--ria-loop", "--ria-max-rounds", "3",
    "--entity-feval",
    "--entity-feval-version", "v2",
]


def run_config(threshold: float, max_k: int | None, limit: int) -> dict:
    """Run a single config and return summary stats."""
    if max_k is None:
        label = f"t{threshold}_kinf"
    else:
        label = f"t{threshold}_k{max_k}"

    outdir = f"{RESULT_BASE}/{label}"
    result_file = f"{outdir}/results.jsonl/biology_results.jsonl"

    # Check if already done
    if os.path.exists(result_file):
        n = sum(1 for _ in open(result_file))
        if n >= limit:
            print(f"  SKIP {label} (already done: {n} queries)")
            return load_results(result_file, label, threshold, max_k)

    # Build command
    cmd = [PYTHON, SCRIPT] + COMMON_ARGS + [
        "--limit", str(limit),
        "--ag-threshold", str(threshold),
        "--output", f"{outdir}/results.jsonl",
    ]
    if max_k is not None:
        cmd += ["--ag-max-k", str(max_k)]

    print(f"  RUN  {label}: threshold={threshold}, max_k={max_k}")

    env = os.environ.copy()
    env["PYTHONPATH"] = "experiments/hotpotqa_v2/src"

    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)

    if proc.returncode != 0:
        print(f"  ERROR: {proc.stderr[-200:]}")
        return {"label": label, "ndcg": 0, "recall": 0, "n": 0}

    return load_results(result_file, label, threshold, max_k)


def load_results(path: str, label: str, threshold: float, max_k) -> dict:
    """Load results from a JSONL file."""
    ndcgs, recalls, mrrs, n_edges, db1s = [], [], [], [], []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            ndcgs.append(r.get("ndcg_10", 0))
            recalls.append(r.get("recall_10", 0))
            mrrs.append(r.get("mrr", 0))
            n_edges.append(r.get("entity_feval_n_bridge", 0))
            db1s.append(r.get("entity_feval_beta1_global", 0))

    n = len(ndcgs)
    avg = lambda x: sum(x) / len(x) if x else 0
    return {
        "label": label,
        "threshold": threshold,
        "max_k": max_k,
        "ndcg": avg(ndcgs),
        "recall": avg(recalls),
        "mrr": avg(mrrs),
        "n_edges": avg(n_edges),
        "delta_beta1": avg(db1s),
        "n": n,
    }


def main():
    print(f"=== Quick Grid Search: {len(THRESHOLDS)} × {len(MAX_KS)} = {len(THRESHOLDS)*len(MAX_KS)} configs ===")
    print(f"    Limit: {LIMIT} queries per config")
    print()

    results = []
    for thresh, maxk in product(THRESHOLDS, MAX_KS):
        r = run_config(thresh, maxk, LIMIT)
        results.append(r)

    # Sort by nDCG
    results.sort(key=lambda x: x["ndcg"], reverse=True)

    print()
    print("=" * 90)
    print(f"{'Rank':4s} {'Config':20s} {'nDCG@10':>10s} {'Recall':>10s} {'MRR':>10s} {'n_edges':>10s} {'Δβ₁':>10s} {'n':>4s}")
    print("-" * 90)
    for i, r in enumerate(results):
        marker = " ★" if i < 5 else ""
        print(f"{i+1:4d} {r['label']:20s} {r['ndcg']:10.4f} {r['recall']:10.4f} "
              f"{r['mrr']:10.4f} {r['n_edges']:10.1f} {r['delta_beta1']:10.1f} {r['n']:4d}{marker}")

    print()
    print("Top-5 candidates for full 50q test:")
    for r in results[:5]:
        mk = r['max_k'] if r['max_k'] is not None else '∞'
        print(f"  {r['label']:20s} threshold={r['threshold']}, max_k={mk}, nDCG={r['ndcg']:.4f}")


if __name__ == "__main__":
    main()
