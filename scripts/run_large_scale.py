#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def build_command(
    *,
    python_bin: str,
    size: int,
    seeds: int,
    feature_profile: str,
    output_dir: Path,
    extra_args: List[str],
) -> List[str]:
    return [
        python_bin,
        "scripts/maze_phase1_replay.py",
        "--feature-profiles",
        feature_profile,
        "--size",
        str(size),
        "--seeds",
        str(seeds),
        "--max-steps",
        "400",
        "--output-dir",
        str(output_dir),
        "--extra-args",
        *extra_args,
    ]


def run_command(cmd: List[str]) -> None:
    print(">>>", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run large-scale maze experiments.")
    parser.add_argument("--feature-profile", default="option_b", choices=["default", "option_a", "option_b"])
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()

    python_bin = sys.executable

    targets = [
        (25, 100, Path("experiments/maze-online-phase1-querylog/results/25x25_100seed")),
        (51, 50, Path("experiments/maze-online-phase1-querylog/results/51x51_50seed")),
    ]

    for size, seeds, out_dir in targets:
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_command(
            python_bin=python_bin,
            size=size,
            seeds=seeds,
            feature_profile=args.feature_profile,
            output_dir=out_dir,
            extra_args=args.extra_args,
        )
        run_command(cmd)

    print("All experiments completed.")


if __name__ == "__main__":
    main()
