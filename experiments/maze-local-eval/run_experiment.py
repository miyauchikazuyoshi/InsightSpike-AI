#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
import csv
import sys
import importlib.util

_ROOT = Path(__file__).resolve().parent
local_src = (_ROOT / "src").resolve()
if str(local_src) not in sys.path:
    sys.path.insert(0, str(local_src))


def _load_base_module(name: str, filename: str):
    module_name = f"maze_online_phase1_querylog.src.{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    base_src = (_ROOT / "../maze-online-phase1-querylog/src").resolve()
    if str(base_src) not in sys.path:
        sys.path.insert(0, str(base_src))
    import types
    base_package = "maze_online_phase1_querylog"
    if base_package not in sys.modules:
        pkg = types.ModuleType(base_package)
        pkg.__path__ = [str(base_src.parent)]  # type: ignore[attr-defined]
        sys.modules[base_package] = pkg
    src_package = f"{base_package}.src"
    if src_package not in sys.modules:
        subpkg = types.ModuleType(src_package)
        subpkg.__path__ = [str(base_src)]  # type: ignore[attr-defined]
        sys.modules[src_package] = subpkg
    spec = importlib.util.spec_from_file_location(module_name, base_src / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


navigator_mod = importlib.util.spec_from_file_location(
    "maze_local_eval.src.navigator", local_src / "navigator.py"
)
if navigator_mod and navigator_mod.loader:
    import types

    if "maze_local_eval" not in sys.modules:
        pkg = types.ModuleType("maze_local_eval")
        pkg.__path__ = [str(local_src)]  # type: ignore[attr-defined]
        sys.modules["maze_local_eval"] = pkg
    if "maze_local_eval.src" not in sys.modules:
        subpkg = types.ModuleType("maze_local_eval.src")
        subpkg.__path__ = [str(local_src)]  # type: ignore[attr-defined]
        sys.modules["maze_local_eval.src"] = subpkg

    # Ensure local_adapter is loaded under package name for relative import
    adapter_spec = importlib.util.spec_from_file_location(
        "maze_local_eval.src.local_adapter", local_src / "local_adapter.py"
    )
    if adapter_spec and adapter_spec.loader:
        adapter_module = importlib.util.module_from_spec(adapter_spec)
        sys.modules["maze_local_eval.src.local_adapter"] = adapter_module
        adapter_spec.loader.exec_module(adapter_module)  # type: ignore[attr-defined]
    _navigator_module = importlib.util.module_from_spec(navigator_mod)
    sys.modules["maze_local_eval.src.navigator"] = _navigator_module
    navigator_mod.loader.exec_module(_navigator_module)  # type: ignore[attr-defined]
else:  # pragma: no cover
    raise ImportError("Failed to load localized navigator module.")

LocalizedGeDIGNavigator = _navigator_module.LocalizedGeDIGNavigator  # type: ignore
NavigatorConfig = _navigator_module.NavigatorConfig  # type: ignore

maze_env_mod = _load_base_module("maze_env", "maze_env.py")
MazeEnvironment = maze_env_mod.MazeEnvironment  # type: ignore

metrics_mod = _load_base_module("metrics", "metrics.py")
summarize = metrics_mod.summarize  # type: ignore

baselines_mod = _load_base_module("baselines", "baselines.py")
run_simple_heuristic = baselines_mod.run_simple_heuristic  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Maze Phase-1 localized geDIG runner")

    def _bool_arg(value: str) -> bool:
        if isinstance(value, bool):
            return value
        lowered = str(value).strip().lower()
        if lowered in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "f", "no", "n", "off"}:
            return False
        raise argparse.ArgumentTypeError(f"expected boolean value, got '{value}'")

    parser.add_argument("--size", type=int, default=15, help="maze size (odd)")
    parser.add_argument("--seeds", type=int, default=20, help="number of seeds")
    parser.add_argument("--seed-offset", type=int, default=0, help="start seed offset")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--theta-na", type=float, default=0.35)
    parser.add_argument("--theta-bt", type=float, default=0.30)
    parser.add_argument("--lambda-weight", type=float, default=0.5)
    parser.add_argument("--sp-beta", type=float, default=0.25)
    parser.add_argument("--max-hops", type=int, default=3, help="maximum hop depth for geDIG multi-hop evaluation")
    parser.add_argument("--decay-factor", type=float, default=0.7, help="decay factor applied to hop weights")
    parser.add_argument("--adaptive-hops", dest="adaptive_hops", action="store_true", help="enable GeDIGCore adaptive hop truncation")
    parser.add_argument("--no-adaptive-hops", dest="adaptive_hops", action="store_false", help="disable adaptive hop truncation")
    parser.set_defaults(adaptive_hops=True)
    parser.add_argument(
        "--ig-norm",
        choices=["before", "max", "two_max", "two_before", "logn"],
        default="before",
        help="information gain normalization strategy (before|max|two_max|two_before|logn)",
    )
    parser.add_argument("--dir-weight", type=float, default=0.15, help="directional bias weight for query vectors")
    parser.add_argument("--wall-weight", type=float, default=6.0, help="wall feature weight used in query/episode vectors")
    parser.add_argument("--visit-weight", type=float, default=5.0, help="visit feature weight used in query/episode vectors")
    parser.add_argument("--use-dynamic-na", type=_bool_arg, default=True, help="enable quantile-based NA threshold adaptation")
    parser.add_argument("--na-quantile", type=float, default=0.85, help="quantile alpha used for dynamic NA threshold (θ_AG)")
    parser.add_argument("--na-hysteresis", type=float, default=1e-3, help="hysteresis margin added to NA threshold")
    parser.add_argument("--na-warmup-steps", type=int, default=10, help="minimum steps before NA threshold evaluation")
    parser.add_argument("--na-cooldown", type=int, default=0, help="minimum steps between NA triggers")
    parser.add_argument("--observed-tau", type=float, default=0.38, help="norm threshold for wiring observed candidates")
    parser.add_argument(
        "--observed-tau-unvisited",
        type=float,
        default=0.32,
        help="norm threshold for wiring observed unvisited candidates",
    )
    parser.add_argument(
        "--no-observed-virtual",
        dest="observed_virtual_enabled",
        action="store_false",
        help="disable virtual wiring of observed candidates",
    )
    parser.set_defaults(observed_virtual_enabled=True)
    parser.add_argument("--compare-baseline", action="store_true")
    parser.add_argument("--output", type=Path, help="save raw results JSON")
    parser.add_argument("--summary", type=Path, help="save summary CSV")
    parser.add_argument(
        "--log-steps",
        type=str,
        help="per-step CSV log path (supports {seed} placeholder)",
    )
    parser.add_argument(
        "--feature-profile",
        choices=["default", "option_a", "option_b"],
        default="default",
        help="feature configuration profile",
    )
    parser.add_argument("--dg-sp-threshold", type=float, default=-0.10, help="ΔSP relative threshold (≤) required to trigger DG")
    parser.add_argument("--dg-delta-threshold", type=float, default=-0.05, help="Δg = gmin - g0 threshold (≤) fallback for DG")
    parser.add_argument("--local-radius", type=int, default=1, help="hop radius used for localized g0 evaluation")
    return parser.parse_args()


def _resolve_log_path(pattern: Optional[str], seed: int) -> Optional[Path]:
    if not pattern:
        return None
    path_str = pattern.format(seed=seed)
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def main() -> None:
    args = parse_args()
    summary_records: List[Dict[str, float]] = []
    detail_records: List[Dict[str, any]] = []
    baseline_records: List[Dict[str, float]] = []

    for idx in range(args.seed_offset, args.seed_offset + args.seeds):
        env = MazeEnvironment(size=args.size, seed=idx)
        step_log_path = _resolve_log_path(args.log_steps, idx)
        cfg = NavigatorConfig(
            theta_na=args.theta_na,
            theta_bt=args.theta_bt,
            lambda_weight=args.lambda_weight,
            max_hops=args.max_hops,
            sp_beta=args.sp_beta,
            max_steps=args.max_steps,
            decay_factor=args.decay_factor,
            adaptive_hops=args.adaptive_hops,
            use_dynamic_na=args.use_dynamic_na,
            na_hysteresis=args.na_hysteresis,
            na_cooldown=args.na_cooldown,
            na_warmup_steps=args.na_warmup_steps,
            observed_wiring_tau=args.observed_tau,
            observed_wiring_tau_unvisited=args.observed_tau_unvisited,
            observed_virtual_enabled=args.observed_virtual_enabled,
            dir_weight=args.dir_weight,
            wall_weight=args.wall_weight,
            visit_weight=args.visit_weight,
            step_log_path=step_log_path,
            feature_profile=args.feature_profile,
            dg_sp_gain_threshold=args.dg_sp_threshold,
            dg_delta_threshold=args.dg_delta_threshold,
            local_radius=args.local_radius,
            ig_norm_strategy=args.ig_norm,
        )
        navigator = LocalizedGeDIGNavigator(env, cfg)
        stats = navigator.run()
        summary = {"seed": float(idx), **stats.to_dict()}
        summary_records.append(summary)
        detail_records.append({"seed": int(idx), **stats.to_detail()})

        if args.compare_baseline:
            env_baseline = MazeEnvironment(size=args.size, seed=idx)
            base_stats = run_simple_heuristic(env_baseline, max_steps=args.max_steps)
            baseline_records.append(
                {
                    "seed": float(idx),
                    "success": float(base_stats.success),
                    "steps": float(base_stats.steps),
                    "edges": float(base_stats.edges_added),
                }
            )

    aggregate = summarize(summary_records)
    baseline_steps: Optional[float] = None
    baseline_success: Optional[float] = None
    if baseline_records:
        baseline_steps = sum(r["steps"] for r in baseline_records) / len(baseline_records)
        baseline_success = sum(r["success"] for r in baseline_records) / len(baseline_records)

    print("=== Maze Phase-1 Localized geDIG ===")
    print(f"Size={args.size}, Seeds={args.seeds}")
    for k, v in aggregate.to_dict().items():
        print(f"{k:16s}: {v:.4f}")
    if baseline_records:
        print("--- Simple Heuristic ---")
        print(f"steps_avg       : {baseline_steps:.2f}")
        print(f"success_rate    : {baseline_success:.4f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "summary": summary_records,
                    "details": detail_records,
                    "baseline": baseline_records,
                    "config": {
                        "size": args.size,
                        "seeds": args.seeds,
                        "max_steps": args.max_steps,
                        "theta_na": args.theta_na,
                        "theta_bt": args.theta_bt,
                        "lambda_weight": args.lambda_weight,
                        "sp_beta": args.sp_beta,
                        "max_hops": args.max_hops,
                        "decay_factor": args.decay_factor,
                        "adaptive_hops": args.adaptive_hops,
                        "use_dynamic_na": args.use_dynamic_na,
                        "na_hysteresis": args.na_hysteresis,
                        "na_cooldown": args.na_cooldown,
                        "na_warmup_steps": args.na_warmup_steps,
                        "observed_tau": args.observed_tau,
                        "observed_tau_unvisited": args.observed_tau_unvisited,
                        "observed_virtual_enabled": args.observed_virtual_enabled,
                        "dir_weight": args.dir_weight,
                        "wall_weight": args.wall_weight,
                        "visit_weight": args.visit_weight,
                        "na_quantile": cfg.na_quantile,
                        "fail_bias_eta": cfg.fail_bias_eta,
                        "fail_bias_cap": cfg.fail_bias_cap,
                        "dg_sp_gain_threshold": cfg.dg_sp_gain_threshold,
                        "ag_failsafe_threshold": cfg.ag_failsafe_threshold,
                        "step_log_path": str(step_log_path) if step_log_path is not None else "",
                        "feature_profile": cfg.feature_profile,
                        "branch_feature_enabled": cfg.branch_feature_enabled,
                        "dg_delta_threshold": cfg.dg_delta_threshold,
                        "custom_weight_vector": list(cfg.custom_weight_vector) if cfg.custom_weight_vector else [],
                        "local_radius": cfg.local_radius,
                        "ig_norm_strategy": cfg.ig_norm_strategy,
                    },
                },
                fh,
                indent=2,
            )

    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        with args.summary.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "size",
                    "seeds",
                    "success_rate",
                    "avg_steps",
                    "avg_edges",
                    "na_rate",
                    "integration_rate",
                    "g0_mean",
                    "gmin_mean",
                    "sp_gain_mean",
                    "multihop_usage",
                    "best_hop_mean",
                    "baseline_steps",
                    "baseline_success",
                ]
            )
            agg = aggregate.to_dict()
            writer.writerow(
                [
                    args.size,
                    args.seeds,
                    f"{agg['success_rate']:.4f}",
                    f"{agg['avg_steps']:.2f}",
                    f"{agg['avg_edges']:.2f}",
                    f"{agg['na_rate']:.4f}",
                    f"{agg['integration_rate']:.4f}",
                    f"{agg['g0_mean']:.6f}",
                    f"{agg['gmin_mean']:.6f}",
                    f"{agg['sp_gain_mean']:.6f}",
                    f"{agg['multihop_usage']:.4f}",
                    f"{agg['best_hop_mean']:.2f}",
                    (f"{baseline_steps:.2f}" if baseline_steps is not None else ""),
                    (f"{baseline_success:.4f}" if baseline_success is not None else ""),
                ]
            )


if __name__ == "__main__":
    main()
