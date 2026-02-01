#!/usr/bin/env python3
"""Run a series of RAG v3-lite experiments over multiple datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

# Ensure repository root on sys.path
REPO_ROOT = Path(__file__).resolve().parents[3]
LITE_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT, REPO_ROOT / "src", LITE_ROOT, LITE_ROOT / "src"):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.append(candidate_str)

from src.config_loader import load_config  # type: ignore
from src.pipeline import run_experiment  # type: ignore


def run_suite(config_path: Path, dataset_paths: Iterable[Path]) -> None:
    for ds_path in dataset_paths:
        cfg = load_config(config_path)
        cfg.dataset_path = ds_path
        suffix = ds_path.stem.replace("sample_queries", "").strip("_") or ds_path.stem
        cfg.name = f"{cfg.name}_{suffix}"
        print(f"[suite] Running {cfg.name} on dataset={ds_path}")
        outpath = run_experiment(cfg)
        print(f"[suite] Finished. Results → {outpath}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch runner for RAG v3-lite baselines and geDIG strategy.")
    parser.add_argument("--config", type=Path, required=True, help="Base YAML config (e.g., configs/experiment_geDIG_vs_baselines.yaml)")
    parser.add_argument("--datasets", type=Path, nargs="+", required=True, help="List of dataset JSONL paths.")
    args = parser.parse_args()

    run_suite(args.config, args.datasets)


if __name__ == "__main__":
    main()
