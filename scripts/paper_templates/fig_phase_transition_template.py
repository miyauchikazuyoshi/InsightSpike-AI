#!/usr/bin/env python3
"""
Placeholder phase-transition figure generator.

Intended to visualize a lambda (or tau×lambda) sweep showing signs of a phase
transition (e.g., success_rate, FMR, PSZ shortfall, ged_min_proxy).

Inputs (CSV or JSON):
- If CSV: columns lambda, success_rate, fmr, psz_shortfall, ged_min_proxy (optional), metric_* (optional)
- If JSON: an array of objects with the same keys as above
- If no file is provided or parsing fails, synthetic data are generated.

Examples:
  python scripts/paper_templates/fig_phase_transition_template.py \
    --input results/maze-lambda/phase_grid.csv \
    --out docs/paper/figures/fig_phase_transition.png
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=None, help="CSV or JSON with lambda/metrics")
    ap.add_argument("--out", type=Path, required=True, help="Output PNG path")
    return ap.parse_args()


def load_data(path: Path | None) -> List[Dict[str, float]]:
    if path is None or not path.exists():
        return []
    try:
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text())
            return [dict(item) for item in data] if isinstance(data, list) else []
        # CSV path
        rows: List[Dict[str, float]] = []
        import csv

        with path.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows.append({k: float(v) for k, v in row.items() if v not in ("", None)})
        return rows
    except Exception:
        return []


def synthetic_data() -> List[Dict[str, float]]:
    lambdas = [round(0.1 * i, 2) for i in range(1, 11)]
    data: List[Dict[str, float]] = []
    for lam in lambdas:
        success = max(0.0, min(1.0, 0.2 + 0.6 * (lam > 0.4) + random.uniform(-0.05, 0.05)))
        fmr = max(0.0, min(1.0, 0.5 - 0.3 * (lam > 0.4) + random.uniform(-0.05, 0.05)))
        psz = max(0.0, 1.0 - success + fmr)
        gedmin = max(0.0, 0.1 * (lam > 0.5) + random.uniform(0.0, 0.05))
        data.append({"lambda": lam, "success_rate": success, "fmr": fmr, "psz_shortfall": psz, "ged_min_proxy": gedmin})
    return data


def plot(data: List[Dict[str, float]], out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise SystemExit(f"matplotlib is required: {e}")

    if not data:
        data = synthetic_data()

    lam = [d.get("lambda") for d in data]
    success = [d.get("success_rate") for d in data]
    fmr = [d.get("fmr") for d in data]
    psz = [d.get("psz_shortfall") for d in data]
    gedmin = [d.get("ged_min_proxy") for d in data]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].plot(lam, success, marker="o", label="success")
    axes[0].plot(lam, fmr, marker="x", label="fmr")
    axes[0].set_xlabel("lambda")
    axes[0].set_title("Success / FMR")
    axes[0].legend()

    axes[1].plot(lam, psz, marker="o", color="C2", label="psz shortfall")
    axes[1].set_xlabel("lambda")
    axes[1].set_title("PSZ shortfall")
    axes[1].legend()

    axes[2].plot(lam, gedmin, marker="o", color="C3", label="ged_min_proxy")
    axes[2].set_xlabel("lambda")
    axes[2].set_title("GED_min proxy")
    axes[2].legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"[plot] wrote {out_path}")


def main() -> None:
    args = parse_args()
    data = load_data(args.input) if args.input else []
    plot(data, args.out)


if __name__ == "__main__":
    main()
