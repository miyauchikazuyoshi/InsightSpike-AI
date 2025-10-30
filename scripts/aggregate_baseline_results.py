#!/usr/bin/env python3
"""Aggregate geDIG and baseline metrics for quick comparison."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass
class Metrics:
    per_mean: float
    acceptance_rate: float
    fmr: float
    latency_p50: float


def load_metrics(path: Path, key: str) -> Metrics:
    data = json.loads(path.read_text())
    result = data["results"][key]
    return Metrics(
        per_mean=float(result["per_mean"]),
        acceptance_rate=float(result["acceptance_rate"]),
        fmr=float(result["fmr"]),
        latency_p50=float(result["latency_p50"]),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gedig", type=Path, required=True, help="geDIG results JSON")
    ap.add_argument("--baseline", type=Path, required=True, help="baseline results JSON")
    ap.add_argument("--baseline-key", type=str, default="graphrag_baseline")
    args = ap.parse_args()

    gedig = load_metrics(args.gedig, "gedig_ag_dg")
    baseline = load_metrics(args.baseline, args.baseline_key)

    table: Dict[str, Dict[str, float]] = {
        "geDIG": gedig.__dict__,
        "baseline": baseline.__dict__,
    }
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
