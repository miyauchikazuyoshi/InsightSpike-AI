#!/usr/bin/env python3
"""
Run 25x25 / s500 maze batch (eval or L3) across multiple seeds, then aggregate and
optionally update the TeX table for the paper.

Examples:
  # Run 60 seeds for Eval path (use_main_l3 = false), aggregate, update TeX
  python scripts/run_maze_batch_and_update.py --mode eval --seeds 60 --workers 4 --update-tex

  # Run 60 seeds for L3 path (use_main_l3 = true) and aggregate only
  python scripts/run_maze_batch_and_update.py --mode l3 --seeds 60 --workers 4

Notes:
  - This script calls the experiment driver:
      experiments/maze-query-hub-prototype/run_experiment_query.py
  - Outputs are written under batch_25x25 with the prefix:
      Eval: paper25_25x25_s500_eval_seed{seed}
      L3:   paper25_25x25_s500_seed{seed}
  - Aggregation uses scripts/aggregate_maze_batch.py and writes to:
      docs/paper/data/maze_25x25_eval_s500.json (Eval)
      docs/paper/data/maze_25x25_l3_s500.json   (L3)
  - When --update-tex is set, the script updates the Eval or L3 column in
      docs/paper/figures/maze_25x25_s500_table.tex accordingly.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple


RE_LINE = re.compile(r"^(?P<label>\s*[^&]+?)\s*&\s*(?P<eval>[^&]+?)\s*&\s*(?P<l3>[^\\]+)\\\\\s*$")

METRIC_MAP = [
    ("Success rate", "success_rate", "rate"),
    ("Avg. steps", "avg_steps", "float1"),
    ("Avg. edges", "avg_edges", "float1"),
    ("AG rate", "ag_rate", "rate"),
    ("DG rate", "dg_rate", "rate"),
    ("Mean $g_0$", "g0_mean", "float4"),
    ("Mean $g_{\\min}$", "gmin_mean", "float4"),
    ("Avg. eval time (ms)", "avg_time_ms_eval", "float1"),
    ("P95 eval time (ms)", "p95_time_ms_eval", "float1"),
]


def fmt_value(value: float, kind: str) -> str:
    if value is None:
        return "--"
    if kind == "rate":
        return f"{value:.3f}"
    if kind == "float4":
        return f"{value:.4f}"
    # default 1 decimal
    return f"{value:.1f}"


def update_tex_table(tex_path: Path, agg_json: Path, mode: str) -> None:
    """Update eval or L3 column in the TeX table from aggregated JSON.

    Args:
      tex_path: docs/paper/figures/maze_25x25_s500_table.tex
      agg_json: aggregated JSON path (eval or l3)
      mode: 'eval' or 'l3' (which column to update)
    """
    data = json.loads(agg_json.read_text())
    lines = tex_path.read_text().splitlines()
    out_lines: List[str] = []

    for line in lines:
        m = RE_LINE.match(line)
        if not m:
            out_lines.append(line)
            continue
        label = m.group("label").strip()
        eval_str = m.group("eval").rstrip()
        l3_str = m.group("l3").rstrip()

        replaced = False
        for metric_label, key, kind in METRIC_MAP:
            if label.startswith(metric_label):
                if key not in data:
                    # if metric absent (e.g., AG/DG not present for eval) keep as-is
                    break
                new_val = fmt_value(data[key], kind)
                if mode == "eval":
                    eval_str = new_val
                else:
                    l3_str = l3_str  # preserve suffix/footnotes after number if any
                    # Try to replace the leading number in l3_str while preserving any suffixes
                    mnum = re.match(r"^\s*([0-9.+-Ee]+)(.*)$", l3_str)
                    if mnum:
                        l3_str = f"{new_val}{mnum.group(2)}".strip()
                    else:
                        l3_str = new_val
                newline = f"{m.group('label')} & {eval_str} & {l3_str} \\\\"  # preserve trailing \\
                out_lines.append(newline)
                replaced = True
                break
        if not replaced:
            out_lines.append(line)

    tex_path.write_text("\n".join(out_lines) + "\n")


def run_one(seed: int, mode: str, base_dir: Path, py: str, size: int, steps: int, extra: List[str]) -> Tuple[int, int]:
    """Run a single seed. Returns (seed, returncode)."""
    prefix = f"paper25_25x25_s{steps}_seed" if mode == "l3" else f"paper25_25x25_s{steps}_eval_seed"
    out = base_dir / f"{prefix}{seed}_summary.json"
    step = base_dir / f"{prefix}{seed}_steps.json"

    cmd = [
        py, str(Path("experiments/maze-query-hub-prototype/run_experiment_query.py")),
        "--preset", "paper",
        "--maze-size", str(size),
        "--max-steps", str(steps),
        "--theta-ag", "0.4",
        "--theta-dg", "0.0",
        "--top-m", "32",
        "--candidate-cap", "32",
        "--cand-radius", "1.0",
        "--link-radius", "1.0",
        "--seed-start", str(seed),
        "--seeds", "1",
        "--output", str(out),
        "--step-log", str(step),
    ] + extra

    env = os.environ.copy()
    env.setdefault("INSIGHTSPIKE_MIN_IMPORT", "1")
    try:
        proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return (seed, proc.returncode)
    except Exception:
        return (seed, 1)


def aggregate(dir_path: Path, mode: str, steps: int) -> Path:
    prefix = f"paper25_25x25_s{steps}_seed" if mode == "l3" else f"paper25_25x25_s{steps}_eval_seed"
    out = Path("docs/paper/data/maze_25x25_l3_s500.json" if mode == "l3" else "docs/paper/data/maze_25x25_eval_s500.json")
    cmd = [
        "python", "scripts/aggregate_maze_batch.py",
        "--dir", str(dir_path),
        "--prefix", prefix,
        "--out", str(out),
    ]
    subprocess.run(cmd, check=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["eval", "l3"], default="eval")
    ap.add_argument("--seeds", type=int, default=60)
    ap.add_argument("--start-seed", type=int, default=0)
    ap.add_argument("--maze-size", type=int, default=25)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--update-tex", action="store_true")
    ap.add_argument("--tex-path", type=Path, default=Path("docs/paper/figures/maze_25x25_s500_table.tex"))
    ap.add_argument("--batch-dir", type=Path, default=Path("experiments/maze-query-hub-prototype/results/batch_25x25"))
    ap.add_argument("--python", type=str, default="python")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    extra: List[str] = []
    if args.mode == "l3":
        extra.append("--use-main-l3")

    # Ensure output dir exists
    args.batch_dir.mkdir(parents=True, exist_ok=True)

    # Run seeds
    if not args.dry_run:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futs = []
            for s in range(args.start_seed, args.start_seed + args.seeds):
                futs.append(ex.submit(run_one, s, args.mode, args.batch_dir, args.python, args.maze_size, args.max_steps, extra))
            failures: List[int] = []
            for fut in as_completed(futs):
                seed, rc = fut.result()
                print(f"[run] seed={seed} rc={rc}")
                if rc != 0:
                    failures.append(seed)
            if failures:
                print(f"[warn] {len(failures)} seeds failed: {failures}")

    # Aggregate
    agg_path = aggregate(args.batch_dir, args.mode, args.max_steps)
    print(f"[agg] wrote {agg_path}")

    # Update TeX table
    if args.update_tex:
        update_tex_table(args.tex_path, agg_path, mode=args.mode)
        print(f"[tex] updated {args.tex_path} ({args.mode} column)")


if __name__ == "__main__":
    main()

