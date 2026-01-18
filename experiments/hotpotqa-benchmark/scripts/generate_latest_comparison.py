#!/usr/bin/env python3
"""Generate latest comparison tables by dataset and method."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_METHODS = [
    "closed_book",
    "bm25",
    "contriever",
    "dpr",
    "colbert",
    "static_graphrag",
    "gedig",
]


def normalize_data_name(path: str | None) -> str:
    if not path:
        return "unknown"
    return Path(path).name


def timestamp_key(ts: str | None) -> int:
    if not ts:
        return -1
    try:
        return int(ts.replace("_", ""))
    except Exception:
        return -1


def load_records(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def latest_runs(records: list[dict], data_name: str, methods: list[str]) -> dict[str, dict]:
    subset = [r for r in records if normalize_data_name(r.get("data")) == data_name]
    out: dict[str, dict] = {}
    for method in methods:
        cand = [r for r in subset if r.get("method") == method]
        if not cand:
            continue
        out[method] = max(cand, key=lambda r: timestamp_key(r.get("timestamp")))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate latest comparison tables")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent.parent / "results",
        help="Directory containing summary_runs.jsonl",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Output markdown path (default: results/latest_comparison.md)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output csv path (default: results/latest_comparison.csv)",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=DEFAULT_METHODS,
        help="Method order to include",
    )
    parser.add_argument(
        "--data",
        nargs="*",
        default=None,
        help="Filter dataset filenames (e.g., hotpotqa_sample_100.jsonl)",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    summary_runs = results_dir / "summary_runs.jsonl"
    if not summary_runs.exists():
        raise FileNotFoundError(f"Missing {summary_runs}")

    output_md = args.output_md or results_dir / "latest_comparison.md"
    output_csv = args.output_csv or results_dir / "latest_comparison.csv"

    records = load_records(summary_runs)
    data_names = sorted({normalize_data_name(r.get("data")) for r in records})
    if args.data:
        data_filter = set(args.data)
        data_names = [name for name in data_names if name in data_filter]

    rows = []
    md_lines = ["# Latest comparison (generated)", ""]
    for data_name in data_names:
        latest = latest_runs(records, data_name, args.methods)
        if not latest:
            continue
        md_lines.append(f"## {data_name}")
        md_lines.append("| method | em | f1 | sf_f1 | count | timestamp | incomplete |")
        md_lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for method in args.methods:
            record = latest.get(method)
            if not record:
                continue
            row = {
                "data": data_name,
                "method": method,
                "em": record.get("em"),
                "f1": record.get("f1"),
                "sf_f1": record.get("sf_f1"),
                "count": record.get("count"),
                "timestamp": record.get("timestamp"),
                "incomplete": bool(record.get("incomplete", False)),
            }
            rows.append(row)
            md_lines.append(
                f"| {row['method']} | {row['em']} | {row['f1']} | {row['sf_f1']} | "
                f"{row['count']} | {row['timestamp']} | {row['incomplete']} |"
            )
        md_lines.append("")

    output_md.write_text("\n".join(md_lines), encoding="utf-8")

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "data",
                "method",
                "em",
                "f1",
                "sf_f1",
                "count",
                "timestamp",
                "incomplete",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[done] Wrote {output_md}")
    print(f"[done] Wrote {output_csv}")


if __name__ == "__main__":
    main()
