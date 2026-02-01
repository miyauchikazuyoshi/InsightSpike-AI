#!/usr/bin/env python3
"""
Run maze query-hub experiments with safety step caps for large mazes.

Rationale:
  - Paper v4 used 250-step limits, but rerunning 25x25/51x51 with the same cap
    underestimates long-path behaviour.
  - Directly bumping to 500/1500 steps via ad-hoc commands is risky (runs can
    explode in wall-clock time if mis-typed).

This helper enforces per-size maximum steps, defaults to:
    25x25 -> 500 steps
    51x51 -> 1500 steps
and orchestrates multiple runs sequentially with the paper L3-only settings.

Example:
  python experiments/maze-query-hub-prototype/tools/run_adjusted_step_caps.py \
    --targets 25:600:60,51:2000:40 \
    --safety-caps 25:500,51:1500 \
    --out-root experiments/maze-query-hub-prototype/results/adjusted_steps \
    --namespace-prefix adj_v4 --ultra-light
"""
from __future__ import annotations

import argparse
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


HERE = Path(__file__).resolve().parent
MAZE_ROOT = HERE.parent
EXPERIMENTS_ROOT = MAZE_ROOT.parent
REPO_ROOT = EXPERIMENTS_ROOT.parent
RUNNER = MAZE_ROOT / "run_experiment_query.py"


@dataclass
class RunSpec:
    maze_size: int
    max_steps: int
    seeds: int


def parse_targets(text: str, default_seeds: int) -> List[RunSpec]:
    specs: List[RunSpec] = []
    if not text:
        return specs
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(":")
        if len(parts) < 2:
            raise ValueError(f"Invalid target spec '{chunk}' (expected size:max_steps[:seeds])")
        size = int(parts[0])
        steps = int(parts[1])
        seeds = int(parts[2]) if len(parts) >= 3 and parts[2] else default_seeds
        specs.append(RunSpec(size, steps, seeds))
    return specs


def parse_caps(text: str) -> Dict[int, int]:
    caps: Dict[int, int] = {}
    if not text:
        return caps
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid cap spec '{chunk}' (expected size:cap)")
        caps[int(parts[0])] = int(parts[1])
    return caps


def run_cmd(cmd: List[str], env: Dict[str, str], dry_run: bool) -> None:
    print("[run]", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True, env=env)


def build_common_args(
    args,
    spec: RunSpec,
    capped_steps: int,
    extra_args: List[str],
    *,
    sp_ds_path: Optional[Path],
    sp_ds_namespace: Optional[str],
    seeds_override: Optional[int] = None,
    seed_start_override: Optional[int] = None,
) -> List[str]:
    seeds_val = seeds_override if seeds_override is not None else spec.seeds
    seed_start_val = seed_start_override if seed_start_override is not None else args.seed_start
    common = [
        "--maze-size", str(spec.maze_size),
        "--maze-type", str(args.maze_type),
        "--max-steps", str(capped_steps),
        "--seeds", str(seeds_val),
        "--seed-start", str(seed_start_val),
        "--linkset-mode",
        "--norm-base", "link",
        "--theta-ag", str(args.ag),
        "--top-link", "1",
        "--l1-cap", "128",
        "--link-forced-as-base",
        "--snapshot-level", "minimal",
        "--use-main-l3",
    ]
    if args.sp_cache:
        common.append("--sp-cache")
    common += [
        "--sp-cache-mode", str(args.sp_cache_mode),
        "--sp-pair-samples", str(args.sp_pair_samples),
        "--sp-cand-topk", str(args.sp_cand_topk),
        "--sp-report-best-hop",
    ]
    common += [
        "--link-radius", str(args.link_radius),
        "--cand-radius", str(args.cand_radius),
        "--theta-link", str(args.theta_link),
        "--theta-cand", str(args.theta_cand),
    ]
    if args.enable_sp_ds and sp_ds_path and sp_ds_namespace:
        common += [
            "--sp-ds-sqlite", str(sp_ds_path),
            "--sp-ds-namespace", sp_ds_namespace,
        ]
    if args.enable_sp_allpairs_exact:
        common += ["--sp-allpairs-exact", "--sp-exact-stable-nodes"]
    if args.ultra_light:
        common += ["--steps-ultra-light", "--no-post-sp-diagnostics"]
    if args.maze_snapshot_out:
        snap = Path(args.maze_snapshot_out)
        if snap.is_dir():
            snap = snap / f"maze_{spec.maze_size}x{spec.maze_size}.json"
        common += ["--maze-snapshot-out", str(snap)]
    if args.timeline_to_graph:
        common.append("--timeline-to-graph")
    if args.add_next_q:
        common.append("--add-next-q")
    if args.persist_timeline_edges:
        common.append("--persist-timeline-edges")
    if extra_args:
        common += extra_args
    return common


def main() -> None:
    ap = argparse.ArgumentParser(description="Run query-hub experiments with enforced step caps.")
    ap.add_argument("--targets", type=str, default="25:500:60,51:1500:40",
                    help="Comma-separated size:max_steps[:seeds] entries.")
    ap.add_argument("--safety-caps", type=str, default="25:500,51:1500",
                    help="Comma-separated size:cap entries applied before execution.")
    ap.add_argument("--default-seeds", type=int, default=40)
    ap.add_argument("--maze-type", type=str, default="dfs")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--ag", type=float, default=0.4)
    ap.add_argument("--ultra-light", action="store_true")
    ap.add_argument("--out-root", type=Path, default=Path("experiments/maze-query-hub-prototype/results/adjusted_steps"))
    ap.add_argument("--namespace-prefix", type=str, default="adjusted")
    ap.add_argument("--sqlite-dir", type=Path, default=None,
                    help="Optional directory for SQLite dumps (defaults to out-root).")
    ap.add_argument("--maze-snapshot-out", type=Path, default=None,
                    help="Path or directory for maze layout dumps (optional).")
    ap.add_argument("--preset", type=str, default="paper")
    ap.add_argument("--extra-args", nargs=argparse.REMAINDER, default=None,
                    help="Additional arguments forwarded to run_experiment_query.py after '--'.")
    ap.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    ap.add_argument("--force-over-cap", action="store_true",
                    help="Allow requested steps to exceed safety caps (not recommended).")
    ap.add_argument("--per-seed-output", dest="per_seed_output", action="store_true",
                    help="Emit per-seed outputs (default).")
    ap.add_argument("--aggregate-output", dest="per_seed_output", action="store_false",
                    help="Aggregate all seeds into a single summary/steps/sqlite.")
    ap.add_argument("--sp-cache", action="store_true", help="Enable SP cache for greedy ΔSP evaluation.")
    ap.add_argument("--sp-cache-mode", type=str, default="cached_incr",
                    choices=["core", "cached", "cached_incr"], help="SP cache mode to use.")
    ap.add_argument("--sp-pair-samples", type=int, default=600,
                    help="Number of pair samples when capturing L_before.")
    ap.add_argument("--sp-cand-topk", type=int, default=0,
                    help="Top-K cap for greedy SP candidate evaluation (0 = unlimited).")
    ap.add_argument("--enable-sp-ds", dest="enable_sp_ds", action="store_true",
                    help="Persist SP pairsets to SQLite for reuse.")
    ap.add_argument("--disable-sp-ds", dest="enable_sp_ds", action="store_false",
                    help="Disable SP pairset persistence.")
    ap.add_argument("--sp-ds-sqlite", type=Path, default=None,
                    help="SQLite path for SP pairset DS (defaults to <out_root>/sp_pairsets.sqlite).")
    ap.add_argument("--sp-ds-namespace-prefix", type=str, default="spds",
                    help="Namespace prefix for SP pairset DS entries.")
    ap.add_argument("--enable-sp-allpairs-exact", action="store_true",
                    help="Enable ALL-PAIRS exact SP evaluation with APSP reuse.")
    ap.add_argument("--build-html", action="store_true",
                    help="Generate HTML via build_reports.py for each run.")
    ap.add_argument("--link-radius", type=float, default=1.0,
                    help="Weighted L2 radius for link candidates (default 1.0 to expose distant memory).")
    ap.add_argument("--cand-radius", type=float, default=1.0,
                    help="Weighted L2 radius for candidate prefilter (default 1.0).")
    ap.add_argument("--theta-link", type=float, default=0.0,
                    help="Similarity threshold for link candidates.")
    ap.add_argument("--theta-cand", type=float, default=0.0,
                    help="Similarity threshold for candidate pool.")
    ap.add_argument("--timeline-to-graph", dest="timeline_to_graph", action="store_true",
                    help="Inject Q_prev→dir and dir→Q_now edges into graph/timeline.")
    ap.add_argument("--no-timeline-to-graph", dest="timeline_to_graph", action="store_false")
    ap.add_argument("--add-next-q", dest="add_next_q", action="store_true",
                    help="Ensure next query node is materialized for timeline edges.")
    ap.add_argument("--no-add-next-q", dest="add_next_q", action="store_false")
    ap.add_argument("--persist-timeline-edges", dest="persist_timeline_edges", action="store_true",
                    help="Persist timeline edges into SQLite for Strict DS replay.")
    ap.add_argument("--no-persist-timeline-edges", dest="persist_timeline_edges", action="store_false")
    ap.set_defaults(
        per_seed_output=True,
        enable_sp_ds=True,
        sp_cache=True,
        timeline_to_graph=True,
        add_next_q=True,
        persist_timeline_edges=True,
    )
    args = ap.parse_args()

    extra_args: List[str] = []
    if args.extra_args:
        extra_args = args.extra_args
        if extra_args and extra_args[0] == "--":
            extra_args = extra_args[1:]

    specs = parse_targets(args.targets, args.default_seeds)
    if not specs:
        raise SystemExit("No targets specified. Use --targets size:max_steps[:seeds].")
    caps = parse_caps(args.safety_caps)

    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    sqlite_root = args.sqlite_dir or out_root
    sqlite_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str((REPO_ROOT / "src").resolve()))
    env.setdefault("INSIGHTSPIKE_LOG_DIR", "results/logs")
    env.setdefault("MPLCONFIGDIR", "results/mpl")
    env.setdefault("INSIGHTSPIKE_PRESET", args.preset)

    sp_ds_path: Optional[Path] = None
    if args.enable_sp_ds:
        sp_ds_path = args.sp_ds_sqlite or (out_root / "sp_pairsets.sqlite")
        sp_ds_path.parent.mkdir(parents=True, exist_ok=True)

    for spec in specs:
        cap = caps.get(spec.maze_size)
        capped_steps = spec.max_steps
        if cap is not None and capped_steps > cap and not args.force_over_cap:
            print(f"[info] capping {spec.maze_size}x{spec.maze_size} from {capped_steps} to {cap} steps")
            capped_steps = cap
        elif cap is not None and args.force_over_cap and capped_steps > cap:
            print(f"[warn] force-over-cap enabled for {spec.maze_size} (requested {capped_steps} > cap {cap})")

        base_ns = f"{args.namespace_prefix}_{spec.maze_size}x{spec.maze_size}_s{capped_steps}"
        seed_values = [args.seed_start + i for i in range(spec.seeds)]

        if args.per_seed_output:
            for seed in seed_values:
                ns = f"{base_ns}_seed{seed}"
                summary = out_root / f"{ns}_summary.json"
                steps = out_root / f"{ns}_steps.json"
                sqlite_path = sqlite_root / f"{ns}.sqlite"
                sp_namespace = f"{args.sp_ds_namespace_prefix}_{ns}"
                cmd = [
                    "python",
                    str(RUNNER),
                    *build_common_args(
                        args,
                        spec,
                        capped_steps,
                        extra_args,
                        sp_ds_path=sp_ds_path,
                        sp_ds_namespace=sp_namespace,
                        seeds_override=1,
                        seed_start_override=seed,
                    ),
                    "--persist-graph-sqlite", str(sqlite_path),
                    "--persist-namespace", ns,
                    "--persist-forced-candidates",
                    "--output", str(summary),
                    "--step-log", str(steps),
                ]
                run_cmd(cmd, env, args.dry_run)
                if not args.dry_run:
                    print(f"[done] {spec.maze_size}x{spec.maze_size} seed={seed} -> {summary.name}")
                    if args.build_html:
                        html_out = summary.with_name(summary.stem.replace("_summary", "_interactive") + ".html")
                        html_cmd = [
                            "python",
                            str((MAZE_ROOT / "build_reports.py")),
                            "--summary", str(summary),
                            "--steps", str(steps),
                            "--sqlite", str(sqlite_path),
                            "--namespace", ns,
                            "--out", str(html_out),
                            "--strict",
                            "--light-steps",
                            "--present-mode", "strict",
                        ]
                        run_cmd(html_cmd, env, args.dry_run)
        else:
            ns = base_ns
            summary = out_root / f"{ns}_summary.json"
            steps = out_root / f"{ns}_steps.json"
            sqlite_path = sqlite_root / f"{ns}.sqlite"
            sp_namespace = f"{args.sp_ds_namespace_prefix}_{ns}"
            cmd = [
                "python",
                str(RUNNER),
                *build_common_args(
                    args,
                    spec,
                    capped_steps,
                    extra_args,
                    sp_ds_path=sp_ds_path,
                    sp_ds_namespace=sp_namespace,
                ),
                "--persist-graph-sqlite", str(sqlite_path),
                "--persist-namespace", ns,
                "--persist-forced-candidates",
                "--output", str(summary),
                "--step-log", str(steps),
            ]
            run_cmd(cmd, env, args.dry_run)
            if not args.dry_run:
                print(f"[done] {spec.maze_size}x{spec.maze_size} capped run -> {summary.name}")
                if args.build_html:
                    html_out = summary.with_name(summary.stem.replace("_summary", "_interactive") + ".html")
                    html_cmd = [
                        "python",
                        str((MAZE_ROOT / "build_reports.py")),
                        "--summary", str(summary),
                        "--steps", str(steps),
                        "--sqlite", str(sqlite_path),
                        "--namespace", ns,
                        "--out", str(html_out),
                        "--strict",
                        "--light-steps",
                        "--present-mode", "strict",
                    ]
                    run_cmd(html_cmd, env, args.dry_run)


if __name__ == "__main__":
    main()
