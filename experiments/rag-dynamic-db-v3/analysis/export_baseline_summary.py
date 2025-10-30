#!/usr/bin/env python3
"""
Export baseline summary for RAG.

Writes: data/rag_eval/baseline_summary.csv with columns:
  method, acceptance, FMR, latency_ms

Two input modes:
  - Replicate rows (method, seed, acceptance, FMR, latency_ms) via --from_csv, we aggregate means
  - Simple already-aggregated CSV with the same columns
"""
from __future__ import annotations
from pathlib import Path
import argparse
import csv
import statistics as stats


def aggregate_rows(rows: list[dict]) -> list[dict]:
    # Detect replicate mode if seed present
    has_seed = any("seed" in r for r in rows)
    if not has_seed:
        return rows
    by_method: dict[str, list[dict]] = {}
    for r in rows:
        by_method.setdefault(r.get("method", "unknown"), []).append(r)
    out: list[dict] = []
    for m, lst in by_method.items():
        def fget(k):
            vals = [float(x.get(k)) for x in lst if x.get(k) not in (None, "")]
            return stats.fmean(vals) if vals else None
        out.append({
            "method": m,
            "acceptance": fget("acceptance"),
            "FMR": fget("FMR"),
            "latency_ms": fget("latency_ms"),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from_csv", type=str, required=True, help="Baseline CSV (replicates or aggregates)")
    ap.add_argument("--out", type=str, default="data/rag_eval/baseline_summary.csv")
    args = ap.parse_args()

    with Path(args.from_csv).open("r", newline="") as f:
        r = csv.DictReader(f)
        rows = [dict(row) for row in r]
    rows = aggregate_rows(rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "acceptance", "FMR", "latency_ms"])
        for r in rows:
            w.writerow([r.get("method"), r.get("acceptance"), r.get("FMR"), r.get("latency_ms")])
    print(f"✅ Wrote {out} (rows={len(rows)})")


if __name__ == "__main__":
    raise SystemExit(main())

