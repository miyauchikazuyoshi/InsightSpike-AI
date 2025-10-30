#!/usr/bin/env python3
"""
Export bridge detection scores for M‑ROC.

Writes: data/maze_eval/bridge_scores.csv with columns:
  run_id, step, aggregator, score, y_true

Input sources (auto-detected by default):
  experiments/maze-navigation-enhanced/results/**

You can also pass --results <DIR> to point at a specific run root.

What to log during runs (minimum):
  - For each candidate (at each step), aggregator scores (min / softmin:τ / sum)
  - If possible, average shortest path length before/after an accepted edge:
      SPL_before, SPL_after  (on the induced subgraph for the query region)
  - y_true definition: ΔSPL_rel = (SPL_after - SPL_before) / SPL_before <= -0.02
    (Change threshold with --spl_rel_threshold)

If no parseable artifacts are found, this script writes a schematic CSV
with header and guidance rows, so downstream plotting can be tested.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import csv
import json


def scan_runs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    runs = []
    for p in root.rglob("*"):
        if p.is_dir() and any((p / name).exists() for name in ("events.json", "scores.csv")):
            runs.append(p)
    return runs


def parse_run(run_dir: Path, spl_rel_threshold: float) -> list[dict]:
    rows: list[dict] = []
    # Preferred: a CSV file with per-candidate scores
    scores_csv = run_dir / "scores.csv"
    if scores_csv.exists():
        with scores_csv.open("r", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    step = int(row.get("step") or 0)
                except Exception:
                    step = 0
                agg = (row.get("aggregator") or "min").strip()
                score = float(row.get("score") or 0.0)
                # y_true may be present; otherwise try SPL fields
                if row.get("y_true") is not None:
                    y_true = int(float(row["y_true"]) > 0.5)
                else:
                    try:
                        spl_b = float(row.get("SPL_before"))
                        spl_a = float(row.get("SPL_after"))
                        y_true = 1 if spl_b > 0 and (spl_a - spl_b) / spl_b <= spl_rel_threshold else 0
                    except Exception:
                        y_true = 0
                rows.append({
                    "run_id": run_dir.name,
                    "step": step,
                    "aggregator": agg,
                    "score": score,
                    "y_true": y_true,
                })
        return rows
    # Alternative: JSON events with SPL_before/after snapshots
    events_json = run_dir / "events.json"
    if events_json.exists():
        try:
            ev = json.loads(events_json.read_text())
        except Exception:
            ev = []
        for e in ev:
            step = int(e.get("step") or 0)
            aggs = e.get("aggregators") or {"min": e.get("score", 0.0)}
            spl_b = e.get("SPL_before")
            spl_a = e.get("SPL_after")
            try:
                y_true = 1 if (spl_b and spl_a) and (spl_a - spl_b) / spl_b <= spl_rel_threshold else 0
            except Exception:
                y_true = 0
            for agg, score in aggs.items():
                rows.append({
                    "run_id": run_dir.name,
                    "step": step,
                    "aggregator": str(agg),
                    "score": float(score),
                    "y_true": int(y_true),
                })
        return rows
    return rows


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "step", "aggregator", "score", "y_true"])
        if rows:
            for r in rows:
                w.writerow([r["run_id"], r["step"], r["aggregator"], r["score"], r["y_true"]])
        else:
            # Guidance rows
            w.writerow(["demo_run", 12, "min", -0.013, 1])
            w.writerow(["demo_run", 12, "softmin:0.5", -0.010, 1])
            w.writerow(["demo_run", 12, "sum", -0.006, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=str, default="experiments/maze-navigation-enhanced/results", help="root folder of runs")
    ap.add_argument("--spl_rel_threshold", type=float, default=-0.02, help="ΔSPL_rel threshold for GT (≤ threshold → y_true=1)")
    ap.add_argument("--out", type=str, default="data/maze_eval/bridge_scores.csv")
    args = ap.parse_args()

    root = Path(args.results)
    runs = scan_runs(root)
    all_rows: list[dict] = []
    for run in runs:
        all_rows.extend(parse_run(run, args.spl_rel_threshold))

    out = Path(args.out)
    write_csv(out, all_rows)
    print(f"✅ Wrote {out} (rows={len(all_rows)})")


if __name__ == "__main__":
    raise SystemExit(main())

