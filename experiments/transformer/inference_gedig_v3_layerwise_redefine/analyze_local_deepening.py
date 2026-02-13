#!/usr/bin/env python3
"""Local deepening analysis for transformer geDIG inference runs.

What this script does:
1. Reads latest run_*.json per model directory.
2. Re-estimates best (lambda, gamma) on z-score normalized component curves.
3. Computes trend checks for predefined model sequences.
4. Writes CSV + Markdown summary.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from metrics import grid_search_f


def _as_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _latest_run_json(model_dir: Path) -> Optional[Path]:
    runs = sorted(model_dir.glob("run_*.json"))
    return runs[-1] if runs else None


def _normalize_model_name(raw_name: str) -> str:
    marker = "/models--"
    snapshot_marker = "/snapshots/"
    if marker in raw_name and snapshot_marker in raw_name:
        fragment = raw_name.split(marker, 1)[1]
        repo_fragment = fragment.split(snapshot_marker, 1)[0]
        if repo_fragment:
            return repo_fragment.replace("--", "/")
    return raw_name


def _zscore(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / len(values)
    std = var ** 0.5
    if std <= 1e-12:
        return [0.0 for _ in values]
    return [(x - mean) / std for x in values]


def _valid_triplets(
    delta_epc: Sequence[Optional[float]],
    delta_h: Sequence[Optional[float]],
    delta_struct: Sequence[Optional[float]],
) -> Tuple[List[float], List[float], List[float]]:
    epc: List[float] = []
    dh: List[float] = []
    ds: List[float] = []
    for x, y, z in zip(delta_epc, delta_h, delta_struct):
        if x is None or y is None or z is None:
            continue
        epc.append(float(x))
        dh.append(float(y))
        ds.append(float(z))
    return epc, dh, ds


def _fit_zscore_best(payload: Dict[str, object]) -> Dict[str, Optional[float]]:
    config = payload.get("config") or {}
    conditions = payload.get("conditions") or {}
    baseline = conditions.get("baseline") or {}
    mean_curves = baseline.get("mean_curves") or {}

    structural_term = str(baseline.get("f_structural_term") or config.get("f_structural_term") or "betti1")
    struct_key = "delta_SP" if structural_term == "sp" else "delta_B1"

    delta_epc = mean_curves.get("delta_EPC") or []
    delta_h = mean_curves.get("delta_H") or []
    delta_struct = mean_curves.get(struct_key) or []
    epc, dh, ds = _valid_triplets(delta_epc=delta_epc, delta_h=delta_h, delta_struct=delta_struct)
    if len(epc) < 2:
        return {
            "z_best_lambda": None,
            "z_best_gamma": None,
            "z_best_r2": None,
            "z_best_slope": None,
            "z_points": float(len(epc)),
        }

    z_epc = _zscore(epc)
    z_dh = _zscore(dh)
    z_ds = _zscore(ds)

    lambda_values = [float(x.strip()) for x in str(config.get("grid_lambda", "0.01,0.1,0.5,1,2,5,10")).split(",") if x.strip()]
    gamma_values = [float(x.strip()) for x in str(config.get("grid_gamma", "0.01,0.1,0.5,1,2,5,10")).split(",") if x.strip()]
    best = grid_search_f(
        delta_epc=z_epc,
        delta_h=z_dh,
        delta_sp=z_ds,
        lambda_values=lambda_values,
        gamma_values=gamma_values,
    )
    fit = best.get("fit") if isinstance(best, dict) else None
    return {
        "z_best_lambda": _as_float(best.get("lambda") if isinstance(best, dict) else None),
        "z_best_gamma": _as_float(best.get("gamma") if isinstance(best, dict) else None),
        "z_best_r2": _as_float(fit.get("r2") if isinstance(fit, dict) else None),
        "z_best_slope": _as_float(fit.get("slope") if isinstance(fit, dict) else None),
        "z_points": float(len(epc)),
    }


def _track_of(model_name: str) -> str:
    lowered = model_name.lower()
    if lowered.startswith("sentence-transformers/"):
        return "sentence_embedding"
    return "token_lm"


def _extract_row(payload: Dict[str, object], model_dir_name: str, run_file: Path) -> Dict[str, object]:
    metadata = payload.get("metadata") or {}
    conditions = payload.get("conditions") or {}
    baseline = conditions.get("baseline") or {}
    shuffle = conditions.get("shuffle_input") or {}
    random_init = conditions.get("random_init") or {}

    baseline_fit = baseline.get("mean_fit") or {}
    shuffle_fit = shuffle.get("mean_fit") or {}
    random_fit = random_init.get("mean_fit") or {}
    grid_best = baseline.get("grid_search_best") or {}
    grid_fit = grid_best.get("fit") or {}

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

    z_best = _fit_zscore_best(payload=payload)

    return {
        "model_dir": model_dir_name,
        "model_name": model_name,
        "track": _track_of(model_name),
        "run_file": str(run_file),
        "valid": baseline_r2 is not None,
        "num_texts": metadata.get("num_texts"),
        "baseline_r2": baseline_r2,
        "shuffle_r2": shuffle_r2,
        "random_r2": random_r2,
        "delta_r2_learn": delta_r2_learn,
        "delta_r2_struct": delta_r2_struct,
        "raw_best_lambda": _as_float(grid_best.get("lambda")),
        "raw_best_gamma": _as_float(grid_best.get("gamma")),
        "raw_best_r2": _as_float(grid_fit.get("r2")),
        "raw_best_slope": _as_float(grid_fit.get("slope")),
        **z_best,
    }


def _fmt(value: object, digits: int = 6) -> str:
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _sequence_check(rows: List[Dict[str, object]], names: List[str], metric_key: str) -> Tuple[bool, List[float]]:
    by_name = {str(r.get("model_name")): r for r in rows}
    values: List[float] = []
    for name in names:
        row = by_name.get(name)
        if row is None:
            return False, []
        metric = row.get(metric_key)
        if not isinstance(metric, float):
            return False, []
        values.append(metric)
    nondecreasing = all(values[i] <= values[i + 1] for i in range(len(values) - 1))
    return nondecreasing, values


def _write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    fieldnames = [
        "model_dir",
        "model_name",
        "track",
        "run_file",
        "valid",
        "num_texts",
        "baseline_r2",
        "shuffle_r2",
        "random_r2",
        "delta_r2_learn",
        "delta_r2_struct",
        "raw_best_lambda",
        "raw_best_gamma",
        "raw_best_r2",
        "raw_best_slope",
        "z_best_lambda",
        "z_best_gamma",
        "z_best_r2",
        "z_best_slope",
        "z_points",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_md(rows: List[Dict[str, object]], path: Path) -> None:
    valid_rows = [r for r in rows if r.get("valid") is True]
    invalid_rows = [r for r in rows if r.get("valid") is not True]

    gpt_names = ["distilgpt2", "gpt2", "gpt2-medium"]
    tinyllama_names = [
        "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ]
    gpt_ok, gpt_vals = _sequence_check(valid_rows, names=gpt_names, metric_key="delta_r2_struct")
    tiny_ok, tiny_vals = _sequence_check(valid_rows, names=tinyllama_names, metric_key="delta_r2_struct")

    lines: List[str] = []
    lines.append("# Local Deepening Summary")
    lines.append("")
    lines.append(f"- total_models: {len(rows)}")
    lines.append(f"- valid_models: {len(valid_rows)}")
    lines.append(f"- invalid_models: {len(invalid_rows)}")
    lines.append("")

    lines.append("## Sequence Checks")
    lines.append(f"- GPT delta_r2_struct nondecreasing: {gpt_ok} ({', '.join(_fmt(v, 4) for v in gpt_vals) if gpt_vals else 'N/A'})")
    lines.append(f"- TinyLlama delta_r2_struct nondecreasing: {tiny_ok} ({', '.join(_fmt(v, 4) for v in tiny_vals) if tiny_vals else 'N/A'})")
    lines.append("")

    lines.append("## Metrics Table")
    lines.append(
        "| model | track | valid | baseline_r2 | delta_r2_struct | raw_best_lambda | raw_best_gamma | raw_best_r2 | z_best_lambda | z_best_gamma | z_best_r2 |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["model_name"]),
                    str(row["track"]),
                    "1" if bool(row["valid"]) else "0",
                    _fmt(row["baseline_r2"]),
                    _fmt(row["delta_r2_struct"]),
                    _fmt(row["raw_best_lambda"]),
                    _fmt(row["raw_best_gamma"]),
                    _fmt(row["raw_best_r2"]),
                    _fmt(row["z_best_lambda"]),
                    _fmt(row["z_best_gamma"]),
                    _fmt(row["z_best_r2"]),
                ]
            )
            + " |"
        )
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze local deepening trends for geDIG transformer runs.")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument(
        "--track",
        type=str,
        choices=["all", "token_lm", "sentence_embedding"],
        default="token_lm",
        help="Filter rows by track before output.",
    )
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args()

    results_dir = args.results_dir
    if not results_dir.is_dir():
        raise FileNotFoundError(f"results dir not found: {results_dir}")

    if args.track == "all":
        default_csv = results_dir / "local_deepening_metrics.csv"
        default_md = results_dir / "local_deepening_summary.md"
    else:
        default_csv = results_dir / f"local_deepening_metrics_{args.track}.csv"
        default_md = results_dir / f"local_deepening_summary_{args.track}.md"
    out_csv = args.output_csv or default_csv
    out_md = args.output_md or default_md

    rows: List[Dict[str, object]] = []
    for model_dir in sorted(p for p in results_dir.iterdir() if p.is_dir() and not p.name.startswith(".")):
        run_path = _latest_run_json(model_dir)
        if run_path is None:
            continue
        payload = json.loads(run_path.read_text(encoding="utf-8"))
        rows.append(_extract_row(payload=payload, model_dir_name=model_dir.name, run_file=run_path))

    if args.track != "all":
        rows = [r for r in rows if r.get("track") == args.track]

    _write_csv(rows=rows, path=out_csv)
    _write_md(rows=rows, path=out_md)

    valid_count = sum(1 for r in rows if r.get("valid") is True)
    print(f"[done] rows={len(rows)} valid={valid_count}")
    print(f"[out] {out_csv}")
    print(f"[out] {out_md}")


if __name__ == "__main__":
    main()
