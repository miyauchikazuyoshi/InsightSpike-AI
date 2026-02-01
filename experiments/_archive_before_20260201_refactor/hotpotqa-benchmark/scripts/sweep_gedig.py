#!/usr/bin/env python3
"""Sweep lambda/thresholds for geDIG on HotpotQA."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).parent
EXPERIMENT_DIR = SCRIPT_DIR.parent

SUMMARY_RE = re.compile(r"Summary saved to (.+_summary\.json)")
OUTPUT_RE = re.compile(r"\[run\] Output: (.+\.jsonl)")


def _parse_float_list(values: str) -> list[float]:
    result: list[float] = []
    for raw in values.split(","):
        raw = raw.strip()
        if not raw:
            continue
        result.append(float(raw))
    return result


def _run_once(
    python: str,
    data_path: Path,
    limit: int | None,
    seed: int | None,
    lambda_weight: float,
    ag_pct: float,
    dg_pct: float,
    tune_size: int | None,
    max_expansions: int | None,
    max_retries: int | None,
    retry_wait: float | None,
    retry_backoff: float | None,
    retry_max_wait: float | None,
    env: dict[str, str],
    dry_run: bool,
) -> Path | None:
    cmd = [
        python,
        str(SCRIPT_DIR / "run_gedig.py"),
        "--data",
        str(data_path),
        "--lambda-weight",
        str(lambda_weight),
        "--tune-thresholds",
        "--tune-ag-percentile",
        str(ag_pct),
        "--tune-dg-percentile",
        str(dg_pct),
    ]
    if limit:
        cmd.extend(["--limit", str(limit)])
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    if tune_size:
        cmd.extend(["--tune-size", str(tune_size)])
    if max_expansions is not None:
        cmd.extend(["--max-expansions", str(max_expansions)])
    if max_retries is not None:
        cmd.extend(["--max-retries", str(max_retries)])
    if retry_wait is not None:
        cmd.extend(["--retry-wait", str(retry_wait)])
    if retry_backoff is not None:
        cmd.extend(["--retry-backoff", str(retry_backoff)])
    if retry_max_wait is not None:
        cmd.extend(["--retry-max-wait", str(retry_max_wait)])

    if dry_run:
        print("[dry-run]", " ".join(cmd))
        return None

    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 0:
        print(output)
        raise RuntimeError(f"run_gedig failed (lambda={lambda_weight})")

    match = SUMMARY_RE.search(output)
    if match:
        return Path(match.group(1)).expanduser()

    output_match = OUTPUT_RE.search(output)
    if output_match:
        output_path = Path(output_match.group(1)).expanduser()
        summary_path = output_path.with_name(
            output_path.name.replace(".jsonl", "_summary.json")
        )
        if summary_path.exists():
            return summary_path

    print(output)
    raise RuntimeError("summary path not found in run_gedig output")


def _load_completed(
    results_dir: Path, limit: int | None, max_expansions: int | None
) -> set[tuple[float, float, float, int]]:
    completed: set[tuple[float, float, float, int]] = set()
    if limit is None:
        return completed
    for path in results_dir.glob("gedig_*_summary.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("method") != "gedig":
            continue
        if int(data.get("count", -1)) != int(limit):
            continue
        params = data.get("parameters") or {}
        tuning = data.get("tuning") or {}
        if not tuning.get("enabled", False):
            continue
        if max_expansions is not None and params.get("max_expansions") != max_expansions:
            continue
        try:
            key = (
                float(params.get("lambda_weight")),
                float(tuning.get("ag_percentile")),
                float(tuning.get("dg_percentile")),
                int(params.get("max_expansions", 0)),
            )
        except (TypeError, ValueError):
            continue
        completed.add(key)
    return completed


def _write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep lambda/thresholds for geDIG.")
    parser.add_argument(
        "--data",
        type=Path,
        default=EXPERIMENT_DIR / "data" / "hotpotqa_distractor_dev.jsonl",
        help="Path to data file",
    )
    parser.add_argument("--limit", type=int, default=100, help="Limit examples per run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--lambda-values",
        type=str,
        default="0.7,1.0,1.3",
        help="Comma-separated lambda weights",
    )
    parser.add_argument(
        "--ag-percentiles",
        type=str,
        default="50",
        help="Comma-separated AG percentiles (e.g. 40,50,60)",
    )
    parser.add_argument(
        "--dg-percentiles",
        type=str,
        default="30",
        help="Comma-separated DG percentiles (e.g. 20,30,40)",
    )
    parser.add_argument(
        "--tune-size",
        type=int,
        default=None,
        help="Sample size for threshold tuning (default: limit)",
    )
    parser.add_argument(
        "--max-expansions", type=int, default=None, help="Override max expansions"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="Retry count for LLM rate limit/connection errors",
    )
    parser.add_argument(
        "--retry-wait",
        type=float,
        default=None,
        help="Base seconds to wait before retrying LLM call",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=None,
        help="Backoff multiplier for LLM retries",
    )
    parser.add_argument(
        "--retry-max-wait",
        type=float,
        default=None,
        help="Maximum seconds to wait between LLM retries",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python interpreter to use",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip runs that already exist in results",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock LLM provider (no API calls)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    args = parser.parse_args()

    lambda_values = _parse_float_list(args.lambda_values)
    ag_values = _parse_float_list(args.ag_percentiles)
    dg_values = _parse_float_list(args.dg_percentiles)
    if not lambda_values or not ag_values or not dg_values:
        raise ValueError("lambda-values/ag-percentiles/dg-percentiles must be non-empty")

    tune_size = args.tune_size or args.limit

    env = os.environ.copy()
    if args.mock:
        env["LLM_PROVIDER"] = "mock"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = EXPERIMENT_DIR / "results" / f"sweep_gedig_{timestamp}.csv"

    rows = []
    completed = _load_completed(
        EXPERIMENT_DIR / "results", args.limit, args.max_expansions
    )
    total = len(lambda_values) * len(ag_values) * len(dg_values)
    idx = 0
    for lambda_weight, ag_pct, dg_pct in product(lambda_values, ag_values, dg_values):
        idx += 1
        key = (float(lambda_weight), float(ag_pct), float(dg_pct), int(args.max_expansions or 0))
        if args.resume and key in completed:
            print(
                f"[sweep] {idx}/{total} lambda={lambda_weight} ag_pct={ag_pct} dg_pct={dg_pct} (skip)"
            )
            continue
        print(
            f"[sweep] {idx}/{total} lambda={lambda_weight} ag_pct={ag_pct} dg_pct={dg_pct}"
        )
        summary_path = _run_once(
            args.python,
            args.data,
            args.limit,
            args.seed,
            lambda_weight,
            ag_pct,
            dg_pct,
            tune_size,
            args.max_expansions,
            args.max_retries,
            args.retry_wait,
            args.retry_backoff,
            args.retry_max_wait,
            env,
            args.dry_run,
        )
        if args.dry_run or summary_path is None:
            continue
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        rows.append(
            {
                "lambda_weight": lambda_weight,
                "ag_percentile": ag_pct,
                "dg_percentile": dg_pct,
                "em": summary.get("em"),
                "f1": summary.get("f1"),
                "sf_f1": summary.get("sf_f1"),
                "count": summary.get("count"),
                "summary_path": str(summary_path),
            }
        )

    if rows:
        _write_csv(out_csv, rows)
        print(f"[done] Wrote {out_csv}")


if __name__ == "__main__":
    main()
