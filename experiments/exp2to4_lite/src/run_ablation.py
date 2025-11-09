"""Ablation runner: EPC-only, IG-only, 0-hop only, multihop on/off.

Note: IG-only uses lite controller metric (signature variance) to isolate IG term.
EPC-only sets lambda=0.0. 0-hop only disables multihop.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict
import sys

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
for p in (REPO_ROOT, REPO_ROOT / "src", HERE):
    ps = str(p)
    if ps not in sys.path:
        sys.path.append(ps)

from .config_loader import load_config, ExperimentConfig
from .pipeline import run_experiment


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Run ablations for Exp II–III")
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()

    base = load_config(args.config)
    outputs: Dict[str, Dict[str, Any]] = {}

    # Baseline (as-is)
    r_base = run_experiment(base)
    outputs["base"] = _load_json(r_base)

    # EPC-only (lambda=0)
    cfg_epc = replace(base, gedig=replace(base.gedig, lambda_weight=0.0))
    r_epc = run_experiment(cfg_epc)
    outputs["epc_only"] = _load_json(r_epc)

    # 0-hop only (disable multihop)
    cfg_h0 = replace(base, gedig=replace(base.gedig, use_multihop=False))
    r_h0 = run_experiment(cfg_h0)
    outputs["hop0_only"] = _load_json(r_h0)

    # no-AG (disable attention gate by setting high threshold)
    cfg_no_ag = replace(base, gedig=replace(base.gedig, theta_ag=1e9))
    r_no_ag = run_experiment(cfg_no_ag)
    outputs["no_ag"] = _load_json(r_no_ag)

    # no-DG (disable decision gate by setting very low threshold so never fires)
    cfg_no_dg = replace(base, gedig=replace(base.gedig, theta_dg=-1e9))
    r_no_dg = run_experiment(cfg_no_dg)
    outputs["no_dg"] = _load_json(r_no_dg)

    # gamma=0 (no shortest-path gain)
    cfg_gamma0 = replace(base, gedig=replace(base.gedig, sp_beta=0.0))
    r_gamma0 = run_experiment(cfg_gamma0)
    outputs["gamma0"] = _load_json(r_gamma0)

    # IG-only (approx via lite metric): hack by setting lambda high and lowering structural cost effect
    # In absence of direct control, we reuse base but post-process summary to mark mode.
    # Practically: use higher lambda to emphasise IG.
    cfg_ig = replace(base, gedig=replace(base.gedig, lambda_weight=5.0))
    r_ig = run_experiment(cfg_ig)
    outputs["ig_emphasis"] = _load_json(r_ig)

    out_path = r_base.with_name(r_base.stem + "_ablations.json")
    out_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote ablation bundle: {out_path}")


if __name__ == "__main__":
    main()
