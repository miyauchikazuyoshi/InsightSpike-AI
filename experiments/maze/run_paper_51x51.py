#!/usr/bin/env python3
"""
51x51 Maze Paper Experiment Runner

論文用51x51迷路実験を実行するスクリプト。
- 40 seeds
- max_steps=1500
- インクリメンタル保存でクラッシュ対策
- 再開機能付き

Usage:
    # 新規実行
    python experiments/maze/run_paper_51x51.py

    # 特定のseedから再開
    python experiments/maze/run_paper_51x51.py --resume --seed-offset 13

    # workers数を指定
    python experiments/maze/run_paper_51x51.py --workers 4

    # 出力先を指定（Colab等）
    python experiments/maze/run_paper_51x51.py --output-dir /content/drive/MyDrive/maze_results

    # ドライラン（コマンド確認のみ）
    python experiments/maze/run_paper_51x51.py --dry-run
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# === Configuration ===
CONFIG = {
    "maze_size": 51,
    "max_steps": 1500,
    "total_seeds": 40,
    "workers": 8,

    # geDIG parameters (optimized)
    "max_hops": 10,
    "sp_pair_samples": 128,
    "sp_cand_topk": 12,
    "lambda_weight": 1.0,
    "sp_beta": 1.0,
    "decay_factor": 0.7,
    "sp_hop_expand": 3,
    "theta_ag": -1.0,
    "theta_dg": 0.15,
    "action_temp": 0.1,

    # Modes
    "gh_mode": "greedy",
    "sp_scope": "union",
    "linkset_mode": True,
    "eval_all_hops": True,
}

# Default output directory
DEFAULT_RESULTS_DIR = Path("experiments/maze/results/paper/51x51_40seeds")


def get_python_path() -> str:
    """Get the Python interpreter path."""
    # Try miniconda first
    miniconda_python = Path.home() / "miniconda3" / "bin" / "python"
    if miniconda_python.exists():
        return str(miniconda_python)
    return sys.executable


def count_completed_seeds(results_dir: Path) -> int:
    """Count completed seeds from incremental file."""
    incremental_file = results_dir / "summary.incremental.jsonl"
    if not incremental_file.exists():
        return 0

    with open(incremental_file) as f:
        return sum(1 for _ in f)


def build_command(seed_offset: int, num_seeds: int, workers: int, results_dir: Path, output_suffix: str = "") -> list[str]:
    """Build the experiment command."""
    python = get_python_path()

    output_name = f"summary{output_suffix}.json"
    step_log_name = f"steps{output_suffix}.json"

    cmd = [
        python, "experiments/maze/run_experiment_query.py",
        f"--maze-size", str(CONFIG["maze_size"]),
        f"--max-steps", str(CONFIG["max_steps"]),
        f"--seeds", str(num_seeds),
        f"--seed-start", str(seed_offset),
        f"--workers", str(workers),

        # geDIG params
        f"--max-hops", str(CONFIG["max_hops"]),
        f"--sp-pair-samples", str(CONFIG["sp_pair_samples"]),
        f"--sp-cand-topk", str(CONFIG["sp_cand_topk"]),
        f"--lambda-weight", str(CONFIG["lambda_weight"]),
        f"--sp-beta", str(CONFIG["sp_beta"]),
        f"--decay-factor", str(CONFIG["decay_factor"]),
        f"--sp-hop-expand", str(CONFIG["sp_hop_expand"]),
        f"--theta-ag", str(CONFIG["theta_ag"]),
        f"--theta-dg", str(CONFIG["theta_dg"]),
        f"--action-temp", str(CONFIG["action_temp"]),
        f"--action-policy", "softmax",
        f"--norm-base", "link",

        # Modes
        f"--gh-mode", CONFIG["gh_mode"],
        f"--sp-scope", CONFIG["sp_scope"],

        # Output
        f"--output", str(results_dir / output_name),
        f"--step-log", str(results_dir / step_log_name),
    ]

    if CONFIG["linkset_mode"]:
        cmd.append("--linkset-mode")
    if CONFIG["eval_all_hops"]:
        cmd.append("--eval-all-hops")

    return cmd


def print_status(results_dir: Path):
    """Print current experiment status."""
    incremental_file = results_dir / "summary.incremental.jsonl"

    if not incremental_file.exists():
        print("Status: Not started")
        return

    completed = 0
    success = 0

    with open(incremental_file) as f:
        for line in f:
            data = json.loads(line)
            completed += 1
            if data.get("summary", {}).get("success", False):
                success += 1

    total = CONFIG["total_seeds"]
    success_rate = 100 * success / completed if completed > 0 else 0

    print(f"Status: {completed}/{total} seeds completed ({100*completed/total:.1f}%)")
    print(f"Success: {success}/{completed} ({success_rate:.1f}%)")

    if completed < total:
        remaining = total - completed
        print(f"Remaining: {remaining} seeds")
        print(f"\nTo resume: python {__file__} --resume --seed-offset {completed}")


def main():
    parser = argparse.ArgumentParser(description="Run 51x51 paper experiment")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--seed-offset", type=int, default=0, help="Starting seed offset")
    parser.add_argument("--workers", type=int, default=CONFIG["workers"], help="Number of parallel workers")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: experiments/maze/results/paper/51x51_40seeds)")
    parser.add_argument("--dry-run", action="store_true", help="Print command without executing")
    parser.add_argument("--status", action="store_true", help="Show current status only")
    args = parser.parse_args()

    # Determine output directory
    results_dir = Path(args.output_dir) if args.output_dir else DEFAULT_RESULTS_DIR

    # Create output directory
    results_dir.mkdir(parents=True, exist_ok=True)

    # Status only
    if args.status:
        print_status(results_dir)
        return

    # Determine seed offset
    if args.resume:
        completed = count_completed_seeds(results_dir)
        seed_offset = max(completed, args.seed_offset)
        print(f"Resuming from seed {seed_offset}")
    else:
        seed_offset = args.seed_offset

    # Calculate remaining seeds
    remaining_seeds = CONFIG["total_seeds"] - seed_offset
    if remaining_seeds <= 0:
        print(f"All {CONFIG['total_seeds']} seeds already completed!")
        print_status(results_dir)
        return

    # Build command
    output_suffix = f"_from{seed_offset}" if seed_offset > 0 else ""
    cmd = build_command(seed_offset, remaining_seeds, args.workers, results_dir, output_suffix)

    # Print info
    print("=" * 60)
    print("51x51 Paper Experiment")
    print("=" * 60)
    print(f"Maze size: {CONFIG['maze_size']}x{CONFIG['maze_size']}")
    print(f"Max steps: {CONFIG['max_steps']}")
    print(f"Seeds: {seed_offset} -> {CONFIG['total_seeds']} ({remaining_seeds} to run)")
    print(f"Workers: {args.workers}")
    print(f"Output: {results_dir}")
    print("=" * 60)

    if args.dry_run:
        print("\nCommand (dry-run):")
        print(" \\\n  ".join(cmd))
        return

    # Run
    print(f"\nStarting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)

    env = os.environ.copy()
    env["MAZE_GEDIG_SP_BOUNDARY"] = "trim"

    try:
        subprocess.run(cmd, env=env, check=True)
        print("\n" + "=" * 60)
        print("Experiment completed!")
        print_status(results_dir)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        print_status(results_dir)
    except subprocess.CalledProcessError as e:
        print(f"\nError: {e}")
        print_status(results_dir)
        sys.exit(1)


if __name__ == "__main__":
    main()
