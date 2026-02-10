#!/usr/bin/env python3
"""Summarize multi-model geDIG v2 runs with delta-R2 metrics.

Outputs:
  - CSV table (one row per model run)
  - Markdown report with key rankings and validity checks
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _latest_run_json(model_dir: Path) -> Optional[Path]:
    runs = sorted(model_dir.glob("run_*.json"))
    return runs[-1] if runs else None


def _as_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _family_of(model_name: str) -> str:
    lowered = model_name.lower()
    if "bert" in lowered and "gpt" not in lowered:
        return "bert_family"
    if "gpt" in lowered or "pythia" in lowered:
        return "gpt_family"
    return "other"


def _normalize_model_name(raw_name: str) -> str:
    # Convert local HF cache snapshot paths:
    # .../models--org--name/snapshots/<rev> -> org/name
    marker = "/models--"
    snapshot_marker = "/snapshots/"
    if marker in raw_name and snapshot_marker in raw_name:
        fragment = raw_name.split(marker, 1)[1]
        repo_fragment = fragment.split(snapshot_marker, 1)[0]
        if repo_fragment:
            return repo_fragment.replace("--", "/")
    return raw_name


def _track_of(model_name: str) -> str:
    lowered = model_name.lower()
    if lowered.startswith("sentence-transformers/"):
        return "sentence_embedding"
    return "token_lm"


def _extract_metrics(payload: Dict[str, object], model_dir_name: str, run_path: Path) -> Dict[str, object]:
    metadata = payload.get("metadata") or {}
    conditions = payload.get("conditions") or {}
    baseline = conditions.get("baseline") or {}
    shuffle = conditions.get("shuffle_input") or {}
    random_init = conditions.get("random_init") or {}

    baseline_fit = baseline.get("mean_fit") or {}
    shuffle_fit = shuffle.get("mean_fit") or {}
    random_fit = random_init.get("mean_fit") or {}
    baseline_grid = baseline.get("grid_search_best") or {}
    baseline_grid_fit = baseline_grid.get("fit") or {}

    model_name = _normalize_model_name(str(metadata.get("model") or model_dir_name))

    baseline_r2 = _as_float(baseline_fit.get("r2"))
    shuffle_r2 = _as_float(shuffle_fit.get("r2"))
    random_r2 = _as_float(random_fit.get("r2"))

    delta_r2_learn: Optional[float] = None
    if baseline_r2 is not None and random_r2 is not None:
        delta_r2_learn = baseline_r2 - random_r2

    delta_r2_struct: Optional[float] = None
    controls = [x for x in (shuffle_r2, random_r2) if x is not None]
    if baseline_r2 is not None and controls:
        delta_r2_struct = baseline_r2 - max(controls)

    row: Dict[str, object] = {
        "model_dir": model_dir_name,
        "model_name": model_name,
        "track": _track_of(model_name),
        "family": _family_of(model_name),
        "run_file": str(run_path),
        "valid": baseline_r2 is not None,
        "num_texts": metadata.get("num_texts"),
        "baseline_r2": baseline_r2,
        "baseline_slope": _as_float(baseline_fit.get("slope")),
        "shuffle_r2": shuffle_r2,
        "random_r2": random_r2,
        "delta_r2_learn": delta_r2_learn,
        "delta_r2_struct": delta_r2_struct,
        "best_lambda": _as_float(baseline_grid.get("lambda")),
        "best_gamma": _as_float(baseline_grid.get("gamma")),
        "baseline_grid_r2": _as_float(baseline_grid_fit.get("r2")),
        "baseline_grid_slope": _as_float(baseline_grid_fit.get("slope")),
    }
    return row


def _fmt(value: object, digits: int = 6) -> str:
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    fieldnames = [
        "model_dir",
        "model_name",
        "track",
        "family",
        "run_file",
        "valid",
        "num_texts",
        "baseline_r2",
        "baseline_slope",
        "shuffle_r2",
        "random_r2",
        "delta_r2_learn",
        "delta_r2_struct",
        "best_lambda",
        "best_gamma",
        "baseline_grid_r2",
        "baseline_grid_slope",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _top_k(
    rows: List[Dict[str, object]],
    metric_key: str,
    descending: bool,
    k: int,
) -> List[Tuple[str, Optional[float]]]:
    values: List[Tuple[str, float]] = []
    for row in rows:
        metric = row.get(metric_key)
        if not isinstance(metric, float):
            continue
        values.append((str(row.get("model_name")), metric))
    values.sort(key=lambda x: x[1], reverse=descending)
    return values[:k]


def _write_markdown(rows: List[Dict[str, object]], path: Path) -> None:
    valid_rows = [r for r in rows if r.get("valid") is True]
    invalid_rows = [r for r in rows if r.get("valid") is not True]

    best_struct = _top_k(valid_rows, "delta_r2_struct", descending=True, k=5)
    best_learn = _top_k(valid_rows, "delta_r2_learn", descending=True, k=5)

    lines: List[str] = []
    lines.append("# Multi-model Summary")
    lines.append("")
    lines.append(f"- total_models: {len(rows)}")
    lines.append(f"- valid_models: {len(valid_rows)}")
    lines.append(f"- invalid_models: {len(invalid_rows)}")
    tracks = sorted({str(row.get("track")) for row in rows})
    lines.append(f"- tracks: {', '.join(tracks)}")
    lines.append("")

    if invalid_rows:
        lines.append("## Invalid Models")
        for row in invalid_rows:
            lines.append(f"- {row['model_name']} ({row['model_dir']}): baseline_r2=None")
        lines.append("")

    lines.append("## Metrics Table")
    lines.append(
        "| model | track | family | valid | baseline_r2 | shuffle_r2 | random_r2 | delta_r2_learn | delta_r2_struct | best_lambda | best_gamma | baseline_grid_r2 |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["model_name"]),
                    str(row["track"]),
                    str(row["family"]),
                    "1" if bool(row["valid"]) else "0",
                    _fmt(row["baseline_r2"]),
                    _fmt(row["shuffle_r2"]),
                    _fmt(row["random_r2"]),
                    _fmt(row["delta_r2_learn"]),
                    _fmt(row["delta_r2_struct"]),
                    _fmt(row["best_lambda"]),
                    _fmt(row["best_gamma"]),
                    _fmt(row["baseline_grid_r2"]),
                ]
            )
            + " |"
        )
    lines.append("")

    lines.append("## Top delta_r2_struct")
    if best_struct:
        for model_name, metric in best_struct:
            lines.append(f"- {model_name}: {_fmt(metric)}")
    else:
        lines.append("- None")
    lines.append("")

    lines.append("## Top delta_r2_learn")
    if best_learn:
        for model_name, metric in best_learn:
            lines.append(f"- {model_name}: {_fmt(metric)}")
    else:
        lines.append("- None")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize multi-model geDIG v2 results.")
    parser.add_argument("--results-dir", type=Path, required=True, help="Directory containing model subdirs with run_*.json.")
    parser.add_argument("--output-csv", type=Path, default=None, help="CSV output path (default: <results-dir>/multi_model_metrics.csv)")
    parser.add_argument("--output-md", type=Path, default=None, help="Markdown output path (default: <results-dir>/multi_model_metrics.md)")
    parser.add_argument(
        "--track",
        type=str,
        choices=["all", "token_lm", "sentence_embedding"],
        default="all",
        help="Filter rows by model track before writing outputs.",
    )
    args = parser.parse_args()

    results_dir: Path = args.results_dir
    if not results_dir.is_dir():
        raise FileNotFoundError(f"results dir not found: {results_dir}")

    if args.track == "all":
        default_csv = results_dir / "multi_model_metrics.csv"
        default_md = results_dir / "multi_model_metrics.md"
    else:
        default_csv = results_dir / f"multi_model_metrics_{args.track}.csv"
        default_md = results_dir / f"multi_model_metrics_{args.track}.md"
    output_csv = args.output_csv or default_csv
    output_md = args.output_md or default_md

    rows: List[Dict[str, object]] = []
    for model_dir in sorted(p for p in results_dir.iterdir() if p.is_dir() and not p.name.startswith(".")):
        run_path = _latest_run_json(model_dir)
        if run_path is None:
            continue
        payload = json.loads(run_path.read_text(encoding="utf-8"))
        rows.append(_extract_metrics(payload=payload, model_dir_name=model_dir.name, run_path=run_path))

    if args.track != "all":
        rows = [row for row in rows if row.get("track") == args.track]

    _write_csv(rows=rows, path=output_csv)
    _write_markdown(rows=rows, path=output_md)

    valid_count = sum(1 for row in rows if row.get("valid") is True)
    print(f"[done] rows={len(rows)} valid={valid_count}")
    print(f"[out] {output_csv}")
    print(f"[out] {output_md}")


if __name__ == "__main__":
    main()
