#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List
import importlib.util
import sys

_ROOT = Path(__file__).resolve().parents[1] / "experiments" / "maze-online-phase1-querylog" / "src"
_SPEC = importlib.util.spec_from_file_location("maze_phase1_log_analysis", _ROOT / "log_analysis.py")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules["maze_phase1_log_analysis"] = _MODULE
_SPEC.loader.exec_module(_MODULE)  # type: ignore[arg-type]

summarize_step_log_file = _MODULE.summarize_step_log_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize per-step maze logs (AG/DG stats).")
    parser.add_argument("paths", nargs="+", type=Path, help="CSV step log file(s)")
    parser.add_argument("--pset", action="store_true", help="print summary as key=value lines")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = []
    for path in args.paths:
        summary = summarize_step_log_file(path)
        summaries.append((path, summary))

    for path, summary in summaries:
        print(f"=== {path} ===")
        if args.pset:
            for key, value in summary.to_dict().items():
                print(f"{key}={value}")
        else:
            data = summary.to_dict()
            ag_pct = data['ag_rate'] * 100
            dg_pct = data['dg_rate'] * 100
            print(f"steps: {data['total_steps']} | AG {data['ag_count']} ({ag_pct:.2f}%) | DG {data['dg_count']} ({dg_pct:.2f}%)")
            if data['delta_g_min'] is not None:
                print(f"delta_g_min: {data['delta_g_min']:.4f}")
            if data['sp_relative_min'] is not None:
                print(f"sp_relative_min: {data['sp_relative_min']:.4f}")
            if data['mean_latency_ms'] is not None:
                print(f"latency mean/p50: {data['mean_latency_ms']:.2f} / {data['p50_latency_ms']:.2f} ms")
            print(f"fail_streak_max: {data['fail_streak_max']}")


if __name__ == "__main__":
    main()
