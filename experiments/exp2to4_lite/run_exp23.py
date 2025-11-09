"""Run Experiments II–III (lite) and summarize results.

Wraps experiments/rag-dynamic-db-v3-lite pipeline, prints a compact table,
and saves JSON/CSV summaries under results/.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Dict


# Ensure repository root + local src on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_SRC = Path(__file__).resolve().parent / "src"
for p in (REPO_ROOT, REPO_ROOT / "src", LOCAL_SRC):
    ps = str(p)
    if ps not in sys.path:
        sys.path.append(ps)
from experiments.exp2to4_lite.src.config_loader import load_config  # type: ignore
from experiments.exp2to4_lite.src.pipeline import run_experiment  # type: ignore


def _env_cloud_safe() -> None:
    os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    os.environ.setdefault("INSIGHTSPIKE_LITE_MODE", "1")
    os.environ.setdefault("INSIGHTSPIKE_MIN_IMPORT", "1")
    # cloud-safe logging/cache
    os.environ.setdefault("INSIGHTSPIKE_LOG_DIR", str(REPO_ROOT / "results" / "logs"))
    os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "results" / "mpl"))


def _load_results(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _print_and_save_summary(result_json: Dict[str, Any], outdir: Path) -> tuple[Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    json_path = outdir / f"exp23_summary_{ts}.json"
    csv_path = outdir / f"exp23_summary_{ts}.csv"

    config = result_json.get("config", {})
    results = result_json.get("results", {})

    # Build compact summary
    rows: list[dict[str, Any]] = []
    for name, payload in results.items():
        row = {
            "baseline": name,
            "per_mean": round(float(payload.get("per_mean", 0.0)), 4),
            "acceptance_rate": round(float(payload.get("acceptance_rate", 0.0)), 4),
            "zsr": round(float(payload.get("zsr", 0.0)), 4),
            "fmr": round(float(payload.get("fmr", 0.0)), 4),
            "latency_p50": round(float(payload.get("latency_p50", 0.0)), 2),
            "latency_p95": round(float(payload.get("latency_p95", 0.0)), 2),
            "psz_inside": bool(payload.get("psz_inside", False)),
            "ag_rate": round(float(payload.get("ag_rate", 0.0)), 4),
            "dg_rate": round(float(payload.get("dg_rate", 0.0)), 4),
            "avg_steps": round(float(payload.get("avg_steps", 0.0)), 3),
        }
        rows.append(row)

    # Print compact table
    print("\n=== Experiments II–III Summary ===")
    print(f"dataset={config.get('dataset')} cases={config.get('num_queries')}\n")
    header = (
        "baseline",
        "per_mean",
        "acceptance_rate",
        "zsr",
        "fmr",
        "latency_p50",
        "latency_p95",
        "psz_inside",
        "ag_rate",
        "dg_rate",
        "avg_steps",
    )
    print("	".join(header))
    for r in rows:
        print(
            "\t".join(
                [
                    str(r[k])
                    for k in header
                ]
            )
        )

    # Save JSON
    payload = {
        "config": config,
        "summary": rows,
    }
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    # Save CSV
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(header))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in header})

    print(f"\nSaved summaries: {json_path} and {csv_path}")
    return json_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experiments II–III (lite) and summarize")
    parser.add_argument("--config", type=Path, required=True, help="Path to v3-lite YAML config")
    default_out = REPO_ROOT / "experiments" / "exp2to4_lite" / "results"
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=default_out,
        help="Directory to save exp23 summaries (JSON/CSV)",
    )
    args = parser.parse_args()

    _env_cloud_safe()

    cfg = load_config(args.config)
    result_path = run_experiment(cfg)
    print(f"\nPipeline result: {result_path}")

    result_json = _load_results(result_path)
    _print_and_save_summary(result_json, args.summary_dir)


if __name__ == "__main__":
    main()
