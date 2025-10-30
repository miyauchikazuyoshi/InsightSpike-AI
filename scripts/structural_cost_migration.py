#!/usr/bin/env python3
"""Convert legacy geDIG logs that still use structural_improvement.

Usage:
    python scripts/structural_cost_migration.py --input old.json --output new.json
    python scripts/structural_cost_migration.py --input step_log.csv --overwrite

Supports JSON (nested dict/list) and CSV (step logs). For JSON the script copies
structural_improvement → structural_cost (if missing) and negates the value so the
result matches the new positive-cost convention. The legacy key is preserved unless
--drop-legacy is specified. For CSV the script writes a new column and optionally
drops the legacy column.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Union

JSONType = Union[Dict[str, Any], list, str, int, float, bool, None]


def _convert_json(obj: JSONType, drop_legacy: bool) -> JSONType:
    if isinstance(obj, dict):
        new_obj: Dict[str, Any] = {}
        for key, value in obj.items():
            if key == "structural_improvement" and isinstance(value, (int, float)):
                if "structural_cost" not in obj:
                    new_obj["structural_cost"] = -float(value)
                elif isinstance(obj["structural_cost"], (int, float)):
                    new_obj["structural_cost"] = obj["structural_cost"]
                if not drop_legacy:
                    new_obj[key] = value
                continue
            converted = _convert_json(value, drop_legacy)
            new_obj[key] = converted
        # ensure cost present if nested dict added it
        if "structural_cost" in obj and "structural_cost" not in new_obj:
            new_obj["structural_cost"] = obj["structural_cost"]
        return new_obj
    if isinstance(obj, list):
        return [_convert_json(item, drop_legacy) for item in obj]
    return obj


def _convert_csv(input_path: Path, output_path: Path, drop_legacy: bool) -> None:
    with input_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        if "structural_cost" not in fieldnames:
            fieldnames.append("structural_cost")
        if drop_legacy and "structural_improvement" in fieldnames:
            fieldnames = [f for f in fieldnames if f != "structural_improvement"]
        rows = []
        for row in reader:
            if "structural_cost" not in row or not row["structural_cost"]:
                try:
                    si_val = float(row.get("structural_improvement", ""))
                    row["structural_cost"] = f"{-si_val:.10f}"
                except (TypeError, ValueError):
                    row["structural_cost"] = row.get("structural_cost", "")
            if drop_legacy and "structural_improvement" in row:
                row.pop("structural_improvement", None)
            rows.append(row)
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def convert_file(input_path: Path, output_path: Path, drop_legacy: bool) -> None:
    suffix = input_path.suffix.lower()
    if suffix == ".json":
        with input_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        converted = _convert_json(data, drop_legacy)
        with output_path.open("w", encoding="utf-8") as out_fh:
            json.dump(converted, out_fh, ensure_ascii=False, indent=2)
    elif suffix == ".csv":
        _convert_csv(input_path, output_path, drop_legacy)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert structural_improvement → structural_cost in legacy logs.")
    parser.add_argument("--input", required=True, type=Path, help="Input JSON/CSV file.")
    parser.add_argument("--output", type=Path, help="Output path. Defaults to <input>_converted.ext unless --overwrite.")
    parser.add_argument("--overwrite", action="store_true", help="Replace the input file in-place.")
    parser.add_argument("--drop-legacy", action="store_true", help="Remove structural_improvement after conversion.")
    args = parser.parse_args()

    input_path: Path = args.input
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    if args.overwrite:
        output_path = input_path
    else:
        if args.output:
            output_path = args.output
        else:
            output_path = input_path.with_name(f"{input_path.stem}_converted{input_path.suffix}")

    convert_file(input_path, output_path, drop_legacy=args.drop_legacy)


if __name__ == "__main__":
    main()
