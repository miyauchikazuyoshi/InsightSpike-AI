#!/usr/bin/env python3
"""
Export paper-ready metrics from Query-Hub maze runs.

Reads one run (summary + steps) and emits a compact JSON/CSV for LaTeX v4.

Example:
  python experiments/maze-query-hub-prototype/tools/export_paper_maze.py \
    --summary experiments/maze-query-hub-prototype/results/_51x51_s200_l3_summary.json \
    --steps   experiments/maze-query-hub-prototype/results/_51x51_s200_l3_steps.json \
    --out-json docs/paper/data/maze_51x51_l3_s200.json \
    --out-csv  docs/paper/data/maze_51x51_l3_s200.csv \
    --compression-base mem

Compression bases:
  mem  -> use step['ecand_count'] as denominator for edge compression
  link -> use len(S_link) at hop0 as denominator (requires candidate_selection.links)
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(vals: List[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else 0.0


def _p95(vals: List[float]) -> float:
    if not vals:
        return 0.0
    v = sorted(vals)
    idx = int(round(0.95 * (len(v) - 1)))
    return float(v[idx])


def compute_metrics(summary: Dict[str, Any], steps: List[Dict[str, Any]], *, base: str = "mem", oracle: Dict[str, Any] | None = None) -> Dict[str, Any]:
    # Core summary
    out: Dict[str, Any] = {}
    ssum = summary.get("summary") or {}
    out.update({
        "success_rate": float(ssum.get("success_rate", 0.0)),
        "avg_steps": float(ssum.get("avg_steps", 0.0)),
        "avg_edges": float(ssum.get("avg_edges", 0.0)),
        "g0_mean": float(ssum.get("g0_mean", 0.0)),
        "gmin_mean": float(ssum.get("gmin_mean", 0.0)),
        "avg_k_star": float(ssum.get("avg_k_star", 0.0)),
        "avg_delta_sp": float(ssum.get("avg_delta_sp", 0.0)),
        "avg_delta_sp_min": float(ssum.get("avg_delta_sp_min", 0.0)),
        "best_hop_mean": float(ssum.get("best_hop_mean", 0.0)),
        "best_hop_hist_0": float(ssum.get("best_hop_hist_0", 0.0)),
        "best_hop_hist_1": float(ssum.get("best_hop_hist_1", 0.0)),
        "best_hop_hist_2": float(ssum.get("best_hop_hist_2", 0.0)),
        "best_hop_hist_3": float(ssum.get("best_hop_hist_3", 0.0)),
        "avg_time_ms_eval": float(ssum.get("avg_time_ms_eval", 0.0)),
        "p95_time_ms_eval": float(ssum.get("p95_time_ms_eval", 0.0)),
    })
    # Rates from steps
    ag = sum(1 for r in steps if r.get("ag_fire"))
    dg = sum(1 for r in steps if r.get("dg_fire"))
    n = len(steps) or 1
    out.update({
        "ag_rate": float(ag / n),
        "dg_rate": float(dg / n),
    })
    # Compression ratio (1 - committed/candidate)
    committed_edges = 0
    denom = 0
    for r in steps:
        committed_edges += len(r.get("committed_only_edges", []) or [])
        if base == "mem":
            denom += int(r.get("ecand_count", 0))
        elif base == "link":
            try:
                links = (r.get("selected_links") or [])
                denom += len(links)
            except Exception:
                pass
    out["edge_compression_base"] = base
    if denom > 0:
        out["edge_compression"] = float(max(0.0, 1.0 - (committed_edges / denom)))
    else:
        out["edge_compression"] = 0.0
    # Optional series stats
    g0 = [float(r.get("g0", 0.0)) for r in steps]
    out["g0_p95"] = _p95(g0)
    out["g0_min"] = float(min(g0) if g0 else 0.0)
    out["g0_max"] = float(max(g0) if g0 else 0.0)
    # Optional: merge oracle baseline
    if oracle:
        try:
            osr = float(oracle.get("baseline_success_rate", 0.0))
            oas = float(oracle.get("baseline_avg_steps", 0.0))
            omt = str(oracle.get("method") or "")
        except Exception:
            osr = 0.0; oas = 0.0; omt = ""
        out["oracle_success_rate"] = osr
        out["oracle_avg_steps"] = oas
        if omt:
            out["oracle_method"] = omt
        # Relative efficiency vs oracle (avg_steps / oracle_avg_steps)
        try:
            agent_avg_steps = float(summary.get("summary", {}).get("avg_steps", 0.0))
            out["avg_steps_over_oracle"] = (agent_avg_steps / oas) if oas > 0 else 0.0
            out["avg_steps_minus_oracle"] = (agent_avg_steps - oas) if oas > 0 else agent_avg_steps
        except Exception:
            out["avg_steps_over_oracle"] = 0.0
            out["avg_steps_minus_oracle"] = 0.0
    return out


def write_csv(path: Path, d: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        for k, v in d.items():
            w.writerow([k, v])


def main() -> None:
    ap = argparse.ArgumentParser(description="Export paper-ready maze metrics")
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--steps", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, default=None)
    ap.add_argument("--oracle-json", type=Path, default=None, help="Optional oracle baseline JSON to merge (bfs/dijkstra/astar)")
    ap.add_argument("--compression-base", type=str, default="mem", choices=["mem", "link"])
    args = ap.parse_args()

    summary = load_json(args.summary)
    steps = load_json(args.steps)
    oracle = None
    if args.oracle_json is not None and args.oracle_json.exists():
        try:
            oracle = load_json(args.oracle_json)
        except Exception:
            oracle = None
    metrics = compute_metrics(summary, steps, base=args.compression_base, oracle=oracle)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.out_csv:
        write_csv(args.out_csv, metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
