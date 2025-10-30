#!/usr/bin/env python3
"""Utility runner for Maze Phase-1 replays with optional feature profiles."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def build_command(
    *,
    python_bin: str,
    runner: Path,
    size: int,
    seeds: int,
    seed_offset: int,
    max_steps: int,
    feature_profile: str,
    output_path: Path,
    summary_path: Path,
    log_pattern: Path,
    compare_baseline: bool,
    extra_args: List[str],
) -> List[str]:
    cmd: List[str] = [
        python_bin,
        str(runner),
        "--size",
        str(size),
        "--seeds",
        str(seeds),
        "--seed-offset",
        str(seed_offset),
        "--max-steps",
        str(max_steps),
        "--feature-profile",
        feature_profile,
        "--summary",
        str(summary_path),
        "--output",
        str(output_path),
        "--log-steps",
        str(log_pattern),
    ]
    if compare_baseline:
        cmd.append("--compare-baseline")
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Maze Phase-1 replay orchestrator")
    parser.add_argument("--size", type=int, default=15, help="maze size (odd)")
    parser.add_argument("--seeds", type=int, default=5, help="number of seeds per profile")
    parser.add_argument("--seed-offset", type=int, default=0, help="seed offset")
    parser.add_argument("--max-steps", type=int, default=200, help="maximum steps per episode")
    parser.add_argument(
        "--feature-profiles",
        nargs="+",
        default=["default"],
        choices=["default", "option_a", "option_b"],
        help="feature profiles to evaluate",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/maze-online-phase1-querylog/results/replays"),
        help="base directory for outputs",
    )
    parser.add_argument("--compare-baseline", action="store_true", help="run baseline heuristic")
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        help="additional arguments forwarded to run_experiment.py",
    )
    parser.add_argument("--dry-run", action="store_true", help="print commands without executing")
    args = parser.parse_args()

    runner = Path(__file__).resolve().parents[1] / "experiments" / "maze-online-phase1-querylog" / "run_experiment.py"
    if not runner.exists():
        raise FileNotFoundError(f"run_experiment.py not found at {runner}")

    output_dir = args.output_dir.resolve()
    log_dir = output_dir / "step_logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    python_bin = sys.executable
    extra_args = args.extra_args or []

    for profile in args.feature_profiles:
        summary_path = output_dir / f"{profile}_summary.csv"
        output_path = output_dir / f"{profile}_raw.json"
        log_pattern = log_dir / f"{profile}_seed{{seed}}.csv"
        cmd = build_command(
            python_bin=python_bin,
            runner=runner,
            size=args.size,
            seeds=args.seeds,
            seed_offset=args.seed_offset,
            max_steps=args.max_steps,
            feature_profile=profile,
            output_path=output_path,
            summary_path=summary_path,
            log_pattern=log_pattern,
            compare_baseline=args.compare_baseline,
            extra_args=extra_args,
        )
        if args.dry_run:
            print("DRY:", " ".join(cmd))
            continue
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Command failed for profile {profile} (exit {result.returncode})")


if __name__ == "__main__":
    main()
