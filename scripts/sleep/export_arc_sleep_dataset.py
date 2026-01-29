#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


"""
ARC trace exporter (template)

This repository does not yet have a full ARC solver implementation in code, but we can
still define the trace format that Wake should emit, so Sleep can:
- keep a ledger of candidate programs (positive/negative)
- mine near-miss failures as hard negatives
- build (anchor, positive, negatives) triplets per task

Expected input: JSONL, one event per line, with at least:
  {
    "event": "candidate_evaluated",
    "domain": "arc",
    "task_id": "<task_id>",
    "candidate_id": "<stable id>",
    "program": "<dsl source or AST json>",
    "metrics": {
      "train_exact": true/false,
      "train_loss": <float>,
      "epc": <float>,              # optional: program complexity proxy
      "ig": <float>                # optional: improvement proxy
    },
    "decision": "commit"|"reject", # optional (DG)
    "reason": "<string>"           # optional
  }
"""


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Sleep datasets from ARC solver traces (template).")
    parser.add_argument("--trace", type=Path, required=True, help="ARC trace JSONL (Wake output).")
    parser.add_argument("--out-dir", type=Path, default=Path("results/sleep/arc"), help="Output directory.")
    parser.add_argument("--hard-negatives", type=int, default=5, help="Hard negatives per task.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    per_task: Dict[str, List[Dict[str, Any]]] = {}
    for ev in _iter_jsonl(args.trace):
        if ev.get("event") != "candidate_evaluated":
            continue
        task_id = str(ev.get("task_id") or "unknown")
        per_task.setdefault(task_id, []).append(ev)

    candidates_out: List[Dict[str, Any]] = []
    triplets_out: List[Dict[str, Any]] = []

    for task_id, events in per_task.items():
        # Collect candidates
        for ev in events:
            metrics = ev.get("metrics") or {}
            candidates_out.append(
                {
                    "domain": "arc",
                    "type": "candidate",
                    "task_id": task_id,
                    "candidate_id": ev.get("candidate_id"),
                    "train_exact": bool(metrics.get("train_exact", False)),
                    "train_loss": metrics.get("train_loss"),
                    "epc": metrics.get("epc"),
                    "ig": metrics.get("ig"),
                    "decision": ev.get("decision"),
                    "reason": ev.get("reason"),
                    "program": ev.get("program"),
                }
            )

        # Mine one "positive" (best exact) and some near-miss negatives
        exact = [ev for ev in events if bool((ev.get("metrics") or {}).get("train_exact", False))]
        if not exact:
            continue
        # simplest: pick the first exact as positive prototype
        pos = exact[0]
        pos_id = str(pos.get("candidate_id") or "")
        if not pos_id:
            continue

        # hard negatives: lowest train_loss among non-exact
        non_exact = [ev for ev in events if not bool((ev.get("metrics") or {}).get("train_exact", False))]
        non_exact.sort(key=lambda ev: float((ev.get("metrics") or {}).get("train_loss", 1e9)))
        hard_negs = [str((ev.get("candidate_id") or "")) for ev in non_exact[: max(0, int(args.hard_negatives))] if ev.get("candidate_id")]

        if hard_negs:
            triplets_out.append(
                {
                    "domain": "arc",
                    "type": "task_triplet",
                    "task_id": task_id,
                    "anchor_id": pos_id,
                    "positive_id": pos_id,
                    "hard_negative_ids": hard_negs,
                }
            )

    out_dir = args.out_dir
    n_cand = _write_jsonl(out_dir / "candidates.jsonl", candidates_out)
    n_trip = _write_jsonl(out_dir / "triplets.jsonl", triplets_out)
    stats = {"input": str(args.trace), "counts": {"tasks": len(per_task), "candidates": n_cand, "triplets": n_trip}}
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

