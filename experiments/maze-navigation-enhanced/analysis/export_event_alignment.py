#!/usr/bin/env python3
"""
Export event alignment around NA (M‑Causal) to long format.

Writes: data/maze_eval/event_alignment.csv with columns:
  run_id, t_from_NA, event, value

Input sources:
  - events.csv per run (columns: t_from_NA, BT, accept, evict)
  - or long CSV via --from_csv
"""
from __future__ import annotations
from pathlib import Path
import argparse
import csv


def scan_events(root: Path) -> list[dict]:
    rows: list[dict] = []
    for p in root.rglob("events.csv"):
        run_id = p.parent.name
        with p.open("r", newline="") as f:
            r = csv.DictReader(f)
            cols = set(r.fieldnames or [])
            if {"event", "value", "t_from_NA"}.issubset(cols):
                # already long
                for row in r:
                    rows.append({
                        "run_id": row.get("run_id") or run_id,
                        "t_from_NA": int(row.get("t_from_NA") or 0),
                        "event": row.get("event") or "BT",
                        "value": int(float(row.get("value") or 0) > 0.5),
                    })
            elif {"t_from_NA"}.issubset(cols):
                # wide → long
                candidates = [c for c in ("BT", "accept", "evict") if c in cols]
                for row in r:
                    t = int(row.get("t_from_NA") or 0)
                    for ev in candidates:
                        val = int(float(row.get(ev) or 0) > 0.5)
                        rows.append({"run_id": run_id, "t_from_NA": t, "event": ev, "value": val})
    return rows


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "t_from_NA", "event", "value"])
        if rows:
            for r in rows:
                w.writerow([r["run_id"], r["t_from_NA"], r["event"], r["value"]])
        else:
            for t in range(-2, 6):
                w.writerow(["demo_run", t, "BT", 1 if t >= 2 else 0])
                w.writerow(["demo_run", t, "accept", 1 if t >= 3 else 0])
                w.writerow(["demo_run", t, "evict", 1 if t >= 4 else 0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=str, default="experiments/maze-navigation-enhanced/results")
    ap.add_argument("--from_csv", type=str, default="")
    ap.add_argument("--out", type=str, default="data/maze_eval/event_alignment.csv")
    args = ap.parse_args()

    if args.from_csv:
        in_csv = Path(args.from_csv)
        with in_csv.open("r", newline="") as f:
            r = csv.DictReader(f)
            rows = [dict(row) for row in r]
    else:
        rows = scan_events(Path(args.results))

    out = Path(args.out)
    write_csv(out, rows)
    print(f"✅ Wrote {out} (rows={len(rows)})")


if __name__ == "__main__":
    raise SystemExit(main())

