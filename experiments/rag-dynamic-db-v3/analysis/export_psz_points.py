#!/usr/bin/env python3
"""
Export PSZ scatter points for RAG.

Writes: data/rag_eval/psz_points.csv with columns:
  run_id, query_id, H, k, PER, acceptance, FMR, latency_ms

Default source: parse insight_eval outputs under
  experiments/rag-dynamic-db-v3/insight_eval/results/outputs/*.json
or accept explicit JSON via --from_json (same structure as run_alignment_ragv3.py).
"""
from __future__ import annotations
from pathlib import Path
import argparse
import csv
import json


def extract_from_json(p: Path) -> list[dict]:
    js = json.loads(p.read_text())
    rows = []
    recs = js.get("records") or []
    # The alignment JSON does not have PER/accept/FMR; keep this function generic
    # If you have a RAG results JSON per query with these fields, map them here.
    for i, r in enumerate(recs):
        row = {
            "run_id": p.stem,
            "query_id": i,
            "H": r.get("H", js.get("H", 3)),
            "k": js.get("topk", 8),
            "PER": r.get("PER"),
            "acceptance": r.get("acceptance"),
            "FMR": r.get("FMR"),
            "latency_ms": r.get("latency_ms"),
        }
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "query_id", "H", "k", "PER", "acceptance", "FMR", "latency_ms"])
        for r in rows:
            w.writerow([r.get("run_id"), r.get("query_id"), r.get("H"), r.get("k"), r.get("PER"), r.get("acceptance"), r.get("FMR"), r.get("latency_ms")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from_json", type=str, default="", help="Explicit RAG results JSON per‑query")
    ap.add_argument("--scan", type=str, default="experiments/rag-dynamic-db-v3/insight_eval/results/outputs", help="Folder to scan for JSON")
    ap.add_argument("--out", type=str, default="data/rag_eval/psz_points.csv")
    args = ap.parse_args()

    rows: list[dict] = []
    if args.from_json:
        rows.extend(extract_from_json(Path(args.from_json)))
    else:
        for p in Path(args.scan).glob("*.json"):
            try:
                rows.extend(extract_from_json(p))
            except Exception:
                continue

    write_csv(Path(args.out), rows)
    print(f"✅ Wrote {args.out} (rows={len(rows)})")


if __name__ == "__main__":
    raise SystemExit(main())

