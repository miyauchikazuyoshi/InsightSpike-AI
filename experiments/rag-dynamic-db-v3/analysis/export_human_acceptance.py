#!/usr/bin/env python3
"""
Export human acceptance annotations to a standard format for κ.

Writes: data/rag_eval/human_acceptance.csv with columns:
  query_id, annotator_id, accept

Input: a CSV you already use (pass via --from_csv) with at least
  some notion of (query, annotator, decision). Adjust --qid_col / --aid_col / --acc_col
to map column names.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from_csv", type=str, required=True)
    ap.add_argument("--qid_col", type=str, default="query_id")
    ap.add_argument("--aid_col", type=str, default="annotator_id")
    ap.add_argument("--acc_col", type=str, default="accept")
    ap.add_argument("--out", type=str, default="data/rag_eval/human_acceptance.csv")
    args = ap.parse_args()

    in_csv = Path(args.from_csv)
    with in_csv.open("r", newline="") as f:
        r = csv.DictReader(f)
        rows = []
        for row in r:
            try:
                qid = row[args.qid_col]
                aid = row[args.aid_col]
                acc = int(float(row[args.acc_col]) > 0.5)
                rows.append({"query_id": qid, "annotator_id": aid, "accept": acc})
            except Exception:
                continue

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "annotator_id", "accept"])
        for r in rows:
            w.writerow([r["query_id"], r["annotator_id"], r["accept"]])
    print(f"✅ Wrote {out} (rows={len(rows)})")


if __name__ == "__main__":
    raise SystemExit(main())

