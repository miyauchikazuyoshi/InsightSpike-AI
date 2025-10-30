#!/usr/bin/env python3
"""
Export RAG latency measurements (additional latency) per (H,k).

Writes: data/latency/rag_latency.csv with columns:
  run_id, H, k, latency_ms

Default source: parse insight_eval JSON meta or per‑query records if present.
Otherwise, accept explicit CSV via --from_csv.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import csv
import json


def scan_json(folder: Path) -> list[dict]:
    rows: list[dict] = []
    for p in folder.glob("*.json"):
        try:
            js = json.loads(p.read_text())
        except Exception:
            continue
        H = js.get("H")
        topk = js.get("topk")
        recs = js.get("records") or []
        for r in recs:
            lat = r.get("latency_ms")
            if lat is None:
                continue
            rows.append({
                "run_id": p.stem,
                "H": r.get("H", H),
                "k": topk,
                "latency_ms": lat,
            })
    return rows


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "H", "k", "latency_ms"])
        for r in rows:
            w.writerow([r.get("run_id"), r.get("H"), r.get("k"), r.get("latency_ms")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", type=str, default="experiments/rag-dynamic-db-v3/insight_eval/results/outputs")
    ap.add_argument("--from_csv", type=str, default="")
    ap.add_argument("--out", type=str, default="data/latency/rag_latency.csv")
    args = ap.parse_args()

    if args.from_csv:
        with Path(args.from_csv).open("r", newline="") as f:
            r = csv.DictReader(f)
            rows = [dict(row) for row in r]
    else:
        rows = scan_json(Path(args.scan))

    write_csv(Path(args.out), rows)
    print(f"✅ Wrote {args.out} (rows={len(rows)})")


if __name__ == "__main__":
    raise SystemExit(main())

