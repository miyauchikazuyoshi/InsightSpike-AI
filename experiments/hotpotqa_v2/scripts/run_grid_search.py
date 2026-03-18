#!/usr/bin/env python3
"""Run targeted grid search for AG parameters (Spec Q.1).

7 configs × 10q, 2 parallel workers.
"""
import subprocess
import json
import glob
import os
import time
import concurrent.futures

PYTHON = ".venv/bin/python3"
SCRIPT = "experiments/hotpotqa_v2/scripts/run_bright.py"
BASE = "experiments/hotpotqa_v2/results/v21_specq1_grid"

COMMON = [
    "--mode", "cot_retrieval",
    "--domains", "biology",
    "--limit", "10",
    "--scoring-mode", "gedig_refine",
    "--rerank-alpha", "0.1",
    "--graph-top-k", "50",
    "--token-graph", "--token-graph-walk-score",
    "--ria-loop", "--ria-max-rounds", "3",
    "--entity-feval", "--entity-feval-version", "v2",
]

CONFIGS = [
    ("t0.1_kinf",  ["--ag-threshold", "0.1"]),
    ("t0.15_kinf", ["--ag-threshold", "0.15"]),
    ("t0.3_kinf",  ["--ag-threshold", "0.3"]),
    ("t0.4_kinf",  ["--ag-threshold", "0.4"]),
    ("t0.2_k10",   ["--ag-threshold", "0.2", "--ag-max-k", "10"]),
    ("t0.2_k20",   ["--ag-threshold", "0.2", "--ag-max-k", "20"]),
    ("t0.2_k50",   ["--ag-threshold", "0.2", "--ag-max-k", "50"]),
]


def run_one(label, extra_args):
    outdir = os.path.join(BASE, label, "results.jsonl")
    result_file = os.path.join(outdir, "biology_results.jsonl")

    if os.path.exists(result_file):
        n = sum(1 for _ in open(result_file))
        if n >= 10:
            return label, f"SKIP (already {n}q)", n

    cmd = [PYTHON, SCRIPT] + COMMON + extra_args + ["--output", outdir]
    env = os.environ.copy()
    env["PYTHONPATH"] = "experiments/hotpotqa_v2/src"

    t0 = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    elapsed = time.time() - t0

    if proc.returncode != 0:
        err = proc.stderr[-300:] if proc.stderr else "no stderr"
        return label, f"ERROR rc={proc.returncode}: {err}", 0

    if os.path.exists(result_file):
        n = sum(1 for _ in open(result_file))
        return label, f"OK ({elapsed:.0f}s)", n
    return label, "NO OUTPUT", 0


def analyze():
    print()
    print("=" * 75)
    header = f"{'Config':20s} {'nDCG@10':>8s} {'Recall':>8s} {'MRR':>8s} {'n_edges':>8s} {'Δβ1':>8s}"
    print(header)
    print("-" * 75)

    for d in sorted(glob.glob(os.path.join(BASE, "*/results.jsonl/biology_results.jsonl"))):
        label = d.split("/")[-3]
        ndcgs, recalls, mrrs, n_edges_list, db1_list = [], [], [], [], []
        with open(d) as f:
            for line in f:
                r = json.loads(line)
                ndcgs.append(r.get("ndcg_10", 0))
                recalls.append(r.get("recall_10", 0))
                mrrs.append(r.get("mrr", 0))
                n_edges_list.append(r.get("entity_feval_n_bridge", 0))
                db1_list.append(r.get("entity_feval_beta1_global", 0))
        if not ndcgs:
            continue
        avg = lambda x: sum(x) / len(x)
        print(
            f"{label:20s} {avg(ndcgs):8.4f} {avg(recalls):8.4f} "
            f"{avg(mrrs):8.4f} {avg(n_edges_list):8.1f} {avg(db1_list):8.1f} "
            f"(n={len(ndcgs)})"
        )


def main():
    print(f"=== Grid Search: {len(CONFIGS)} configs × 10q (2 parallel) ===")
    print()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(run_one, label, args): label
            for label, args in CONFIGS
        }
        for future in concurrent.futures.as_completed(futures):
            label, status, n = future.result()
            print(f"  {label:15s}: {status} (n={n})")

    print()
    print("=== ALL CONFIGS COMPLETE ===")
    analyze()


if __name__ == "__main__":
    main()
