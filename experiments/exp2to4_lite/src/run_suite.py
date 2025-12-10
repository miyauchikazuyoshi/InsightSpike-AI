"""Paper-scale suite runner: generate -> (optional) split -> calibrate -> test -> summarize.

Uses only the local exp2to4_lite package and main geDIG core.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple
import sys

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
for p in (REPO_ROOT, REPO_ROOT / "src", HERE):
    ps = str(p)
    if ps not in sys.path:
        sys.path.append(ps)

from .config_loader import load_config, ExperimentConfig
from .pipeline import run_experiment


def _env_cloud_safe() -> None:
    os.environ.setdefault("INSIGHTSPIKE_LITE_MODE", "1")
    os.environ.setdefault("INSIGHTSPIKE_MIN_IMPORT", "1")


def _compute_rates(result_json: Dict[str, Any], baseline_name: str = "gedig_ag_dg") -> Tuple[float, float]:
    payload = result_json.get("results", {}).get(baseline_name, {})
    ag_rate = float(payload.get("ag_rate", 0.0))
    dg_rate = float(payload.get("dg_rate", 0.0))
    return ag_rate, dg_rate


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _grid(values: Iterable[float]) -> Iterable[float]:
    return list(values)


def calibrate_theta(cfg: ExperimentConfig, val_path: Path, ag_target: float, dg_target: float) -> Tuple[float, float, Path]:
    """Grid search on (theta_ag, theta_dg) to match target rates on val set.

    Returns best (theta_ag, theta_dg) and the path to the best result JSON.
    """
    candidates_ag = _grid([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0])
    candidates_dg = _grid([0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8])

    best_score = float("inf")
    best_pair = (cfg.gedig.theta_ag, cfg.gedig.theta_dg)
    best_path: Path | None = None

    base = replace(cfg)
    base.dataset_path = val_path
    base.max_queries = None

    for ag in candidates_ag:
        for dg in candidates_dg:
            trial = replace(base)
            trial.gedig = replace(base.gedig, theta_ag=ag, theta_dg=dg)
            res_path = run_experiment(trial)
            js = _load_json(res_path)
            ag_rate, dg_rate = _compute_rates(js)
            score = abs(ag_rate - ag_target) + abs(dg_rate - dg_target)
            if score < best_score:
                best_score = score
                best_pair = (ag, dg)
                best_path = res_path

    assert best_path is not None
    return best_pair[0], best_pair[1], best_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Run paper-scale suite with calibration and test")
    ap.add_argument("--config", type=Path, required=True, help="Suite config (uses dataset train/val/test if present)")
    ap.add_argument("--calibrate", action="store_true", help="Run theta_ag/theta_dg calibration on val set")
    ap.add_argument("--lambda-weight", type=float, default=None, help="Override geDIG lambda_weight for a single run")
    ap.add_argument("--lambda-sweep", type=float, nargs="+", default=None, help="List of lambda values to sweep; overrides config when set")
    args = ap.parse_args()

    _env_cloud_safe()
    cfg = load_config(args.config)

    # Build lambda list (single override or sweep, else config value)
    if args.lambda_sweep:
        lambda_values = [float(v) for v in args.lambda_sweep]
    elif args.lambda_weight is not None:
        lambda_values = [float(args.lambda_weight)]
    else:
        lambda_values = [float(cfg.gedig.lambda_weight)]

    def run_once(cfg_in: ExperimentConfig, lambda_weight: float) -> None:
        cfg_l = replace(cfg_in, gedig=replace(cfg_in.gedig, lambda_weight=lambda_weight))
        name_suffix = f"_lambda{str(lambda_weight).replace('.', 'p')}" if len(lambda_values) > 1 else ""
        cfg_named = replace(cfg_l, name=cfg_l.name + name_suffix)

        # If split is provided, we respect it; else run on single dataset.
        cfg_cal = cfg_named
        if args.calibrate and cfg_named.dataset_val_path:
            ag_tgt = cfg_named.target_ag_rate or 0.08
            dg_tgt = cfg_named.target_dg_rate or 0.04
            best_ag, best_dg, best_val_path = calibrate_theta(cfg_named, cfg_named.dataset_val_path, ag_tgt, dg_tgt)
            print(f"[lambda={lambda_weight}] Calibrated theta on val: theta_ag={best_ag}, theta_dg={best_dg}\nVal result: {best_val_path}")
            cfg_cal = replace(cfg_named, gedig=replace(cfg_named.gedig, theta_ag=best_ag, theta_dg=best_dg))

        # Choose test or default dataset
        run_cfg = cfg_cal
        if cfg_cal.dataset_test_path:
            run_cfg = replace(cfg_cal, dataset_path=cfg_cal.dataset_test_path)

        out = run_experiment(run_cfg)
        print(f"[lambda={lambda_weight}] Test run complete: {out}")

    for lam in lambda_values:
        run_once(cfg, lam)


if __name__ == "__main__":
    main()
