#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import importlib
import sys
import types

ROOT = Path(__file__).resolve().parents[1] / "experiments" / "maze-online-phase1-querylog"


def _ensure_package(package_name: str, path: Path) -> None:
    if package_name in sys.modules:
        return
    module = types.ModuleType(package_name)
    module.__path__ = [str(path)]
    sys.modules[package_name] = module


_ensure_package("maze_phase1_pkg", ROOT / "src")
_ensure_package("maze_phase1_fixtures_pkg", ROOT / "fixtures")

fixtures_mod = importlib.import_module("maze_phase1_fixtures_pkg.mazes")
maze_utils_mod = importlib.import_module("maze_phase1_pkg.maze_utils")
navigator_mod = importlib.import_module("maze_phase1_pkg.navigator")
log_analysis_mod = importlib.import_module("maze_phase1_pkg.log_analysis")

SCENARIOS = fixtures_mod.SCENARIOS
MazeScenario = fixtures_mod.MazeScenario
create_environment_from_ascii = maze_utils_mod.create_environment_from_ascii
GeDIGNavigator = navigator_mod.GeDIGNavigator
NavigatorConfig = navigator_mod.NavigatorConfig
summarize_step_log_file = log_analysis_mod.summarize_step_log_file


def ensure_step_log_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_config(overrides: Dict[str, float] | None) -> NavigatorConfig:
    cfg = NavigatorConfig(
        theta_na=-0.012,
        theta_bt=-0.012,
        lambda_weight=0.5,
        max_hops=3,
        decay_factor=0.7,
        sp_beta=0.25,
        max_steps=400,
        prefer_unexplored=True,
        use_dynamic_na=True,
        na_quantile=0.76,
        na_hysteresis=1e-3,
        na_warmup_steps=5,
        na_cooldown=0,
        observed_wiring_tau=0.38,
        observed_wiring_tau_unvisited=0.32,
        observed_virtual_enabled=True,
        dir_weight=0.15,
        wall_weight=6.0,
        visit_weight=5.0,
        step_log_path=None,
        dg_sp_gain_threshold=-0.10,
        dg_delta_threshold=-0.03,
    )
    if overrides:
        for key, value in overrides.items():
            setattr(cfg, key, value)
    return cfg


def run_scenario(
    scenario: MazeScenario,
    *,
    log_dir: Path,
    output_json: Path | None,
) -> Tuple[Path, Dict[str, float]]:
    env = create_environment_from_ascii(scenario.ascii_map)
    ensure_step_log_dir(log_dir)
    step_log_path = log_dir / f"{scenario.name}_steps.csv"
    cfg = build_config(scenario.config_overrides)
    cfg.step_log_path = step_log_path

    navigator = GeDIGNavigator(env, cfg)
    stats = navigator.run()

    summary = stats.to_dict()
    summary["scenario"] = scenario.name
    summary["description"] = scenario.description

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        config_dump = {}
        for key, value in cfg.__dict__.items():
            if isinstance(value, Path):
                config_dump[key] = str(value)
            else:
                config_dump[key] = value
        with output_json.open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "scenario": scenario.name,
                    "description": scenario.description,
                    "summary": summary,
                    "config": config_dump,
                    "step_log": str(step_log_path),
                },
                fh,
                indent=2,
            )

    log_summary = summarize_step_log_file(step_log_path)
    return step_log_path, {
        "ag_rate": log_summary.ag_rate,
        "dg_rate": log_summary.dg_rate,
        "delta_g_min": log_summary.delta_g_min or 0.0,
        "sp_relative_min": log_summary.sp_relative_min or 0.0,
    }


def validate_rates(
    scenario: MazeScenario,
    ag_rate: float,
    dg_rate: float,
) -> List[str]:
    msgs: List[str] = []
    if scenario.expected_ag_min is not None and ag_rate < scenario.expected_ag_min:
        msgs.append(f"AG rate {ag_rate:.3f} below expected minimum {scenario.expected_ag_min:.3f}")
    if scenario.expected_ag_max is not None and ag_rate > scenario.expected_ag_max:
        msgs.append(f"AG rate {ag_rate:.3f} above expected maximum {scenario.expected_ag_max:.3f}")
    if scenario.expected_dg_min is not None and dg_rate < scenario.expected_dg_min:
        msgs.append(f"DG rate {dg_rate:.3f} below expected minimum {scenario.expected_dg_min:.3f}")
    if scenario.expected_dg_max is not None and dg_rate > scenario.expected_dg_max:
        msgs.append(f"DG rate {dg_rate:.3f} above expected maximum {scenario.expected_dg_max:.3f}")
    return msgs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Navigator on fixed ASCII maze scenarios.")
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=list(SCENARIOS.keys()),
        help="Scenario names to execute (default: all)",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("experiments/maze-online-phase1-querylog/results/fixed_scenarios/logs"),
        help="Directory to store per-step logs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/maze-online-phase1-querylog/results/fixed_scenarios"),
        help="Directory to store JSON summaries",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MPLCONFIGDIR", str(Path("tmp/mplconfig")))
    log_dir = args.log_dir
    output_dir = args.output_dir

    for name in args.scenarios:
        if name not in SCENARIOS:
            print(f"[WARN] Scenario '{name}' not found. Available: {', '.join(SCENARIOS)}")
            continue
        scenario = SCENARIOS[name]
        step_log_path, rates = run_scenario(
            scenario,
            log_dir=log_dir,
            output_json=output_dir / f"{scenario.name}.json",
        )
        ag_rate = rates["ag_rate"]
        dg_rate = rates["dg_rate"]
        print(f"[{scenario.name}] step_log={step_log_path} | AG={ag_rate:.4f} | DG={dg_rate:.4f}")
        issues = validate_rates(scenario, ag_rate, dg_rate)
        for issue in issues:
            print(f"  - {issue}")


if __name__ == "__main__":
    main()
