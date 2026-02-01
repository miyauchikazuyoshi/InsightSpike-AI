#!/usr/bin/env python3
"""Aggregate HotpotQA benchmark results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METRICS = [
    "em",
    "f1",
    "precision",
    "recall",
    "sf_em",
    "sf_f1",
    "sf_precision",
    "sf_recall",
    "latency_p50_ms",
    "latency_p95_ms",
    "ag_fire_rate",
    "dg_fire_rate",
    "final_ag_fire_rate",
    "final_dg_fire_rate",
    "avg_gedig_score",
    "avg_graph_edges",
]


def load_summary(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def aggregate_by_method(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for record in records:
        method = record.get("method", "unknown")
        count = record.get("count", 0) or 0
        if method not in grouped:
            grouped[method] = {"method": method, "count": 0, "_weights": {}}
        grouped[method]["count"] += count
        for key in METRICS:
            val = record.get(key)
            if isinstance(val, (int, float)) and count:
                grouped[method]["_weights"][key] = grouped[method]["_weights"].get(key, 0.0) + val * count

    aggregated = []
    for method, data in grouped.items():
        total = data["count"] or 0
        row = {"method": method, "count": total}
        for key in METRICS:
            if total and key in data.get("_weights", {}):
                row[key] = data["_weights"][key] / total
        aggregated.append(row)
    return aggregated


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = []
    for key in ["method", "count"] + METRICS:
        if any(key in row for row in rows):
            fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate HotpotQA results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent.parent / "results",
        help="Directory containing *_summary.json files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: results-dir)",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    output_dir = args.output_dir or results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_files = sorted(results_dir.glob("*_summary.json"))
    records = [load_summary(path) for path in summary_files]

    runs_path = output_dir / "summary_runs.jsonl"
    with open(runs_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    aggregated = aggregate_by_method(records)
    agg_json = output_dir / "summary_by_method.json"
    agg_csv = output_dir / "summary_by_method.csv"

    with open(agg_json, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2)
    write_csv(agg_csv, aggregated)

    print(f"[done] Loaded {len(records)} summaries from {results_dir}")
    print(f"[done] Wrote {runs_path}")
    print(f"[done] Wrote {agg_json}")
    print(f"[done] Wrote {agg_csv}")


if __name__ == "__main__":
    main()
