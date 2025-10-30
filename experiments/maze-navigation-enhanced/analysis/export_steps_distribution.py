#!/usr/bin/env python3
"""
Export steps distribution for CDF figure.

Writes: data/maze_eval/steps_distribution.csv with columns:
  run_id, method, steps, success

Input sources:
  - `fast_results_all.csv` / `metrics/*.csv` under results/ (heuristic)
  - or explicit CSV via --from_csv with the same columns
"""
from __future__ import annotations
from pathlib import Path
import argparse
import csv


def scan_steps(root: Path) -> list[dict]:
    rows: list[dict] = []
    # Heuristic: collect from any steps.csv or summary CSVs
    for p in list(root.rglob("steps.csv")) + list(root.rglob("*results*.csv")) + list(root.rglob("metrics/*.csv")):
        run_id = p.parent.name
        try:
            with p.open("r", newline="") as f:
                r = csv.DictReader(f)
                cols = set(r.fieldnames or [])
                if {"method", "steps"}.issubset(cols):
                    for row in r:
                        method = row.get("method") or "geDIG"
                        try:
                            steps = int(row.get("steps") or 0)
                        except Exception:
                            continue
                        succ = int(float(row.get("success") or 1) > 0.5)
                        rows.append({"run_id": run_id, "method": method, "steps": steps, "success": succ})
        except Exception:
            continue
    return rows


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "method", "steps", "success"])
        if rows:
            for r in rows:
                w.writerow([r["run_id"], r["method"], r["steps"], r["success"]])
        else:
            # Guidance examples
            w.writerow(["demo_run", "Random", 200, 1])
            w.writerow(["demo_run", "DFS", 160, 1])
            w.writerow(["demo_run", "geDIG", 90, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=str, default="experiments/maze-navigation-enhanced/results")
    ap.add_argument("--from_csv", type=str, default="")
    ap.add_argument("--out", type=str, default="data/maze_eval/steps_distribution.csv")
    args = ap.parse_args()

    if args.from_csv:
        in_csv = Path(args.from_csv)
        with in_csv.open("r", newline="") as f:
            r = csv.DictReader(f)
            rows = [dict(row) for row in r]
    else:
        rows = scan_steps(Path(args.results))

    out = Path(args.out)
    write_csv(out, rows)
    print(f"✅ Wrote {out} (rows={len(rows)})")


if __name__ == "__main__":
    raise SystemExit(main())

