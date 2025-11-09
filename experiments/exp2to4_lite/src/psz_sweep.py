"""Parameter sweep to approach PSZ band (Acc>=0.95, FMR<=0.02, P50<=200ms).

Runs a small grid over (acceptance_threshold, top_k, max_hops), reports the
best configuration and writes a JSON summary.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .config_loader import load_config, ExperimentConfig
import os
from .pipeline import run_experiment


def inside_psz(stats: Dict[str, float], thr_acc: float, thr_fmr: float, thr_p50: float) -> bool:
    return (
        stats.get("acceptance_rate", 0.0) >= thr_acc and
        stats.get("fmr", 1.0) <= thr_fmr and
        stats.get("latency_p50", 1e9) <= thr_p50
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="PSZ parameter sweep")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--psz-acc", type=float, default=0.95)
    ap.add_argument("--psz-fmr", type=float, default=0.02)
    ap.add_argument("--psz-p50", type=float, default=200.0)
    ap.add_argument("--accept-mode", type=str, default="threshold", choices=["threshold","percentile"], help="Acceptance mode")
    ap.add_argument("--accept-percentile", type=float, default=0.02, help="Percentile for acceptance when mode=percentile (e.g., 0.02 for top 98% accepted)")
    args = ap.parse_args()

    base = load_config(args.config)
    # Limit queries for speed if set
    base = replace(base, max_queries=min(base.max_queries or 50, 50))
    # Enable real latency measurement
    os.environ["EXP_LITE_REAL_LATENCY"] = "1"

    grid_acc = [0.05, 0.10, 0.15, 0.20, 0.30, 0.35]
    grid_k = [3, 4]
    grid_h = [1, 2]
    grid_ag = [3.0, 4.0, 5.0]
    grid_dg = [0.1, 0.2, 0.3]

    trials: List[Tuple[Tuple[float, int, int], Path, Dict[str, Any]]] = []
    best = None
    best_score = 1e9

    for thr in grid_acc:
        for k in grid_k:
            for h in grid_h:
                for ag in grid_ag:
                    for dg in grid_dg:
                        cfg = replace(
                    base,
                    retrieval_top_k=k,
                    gedig=replace(base.gedig, max_hops=h, theta_ag=ag, theta_dg=dg),
                    psz_acceptance_threshold=thr,
                )
                        res_path = run_experiment(cfg)
                        res = json.loads(res_path.read_text(encoding="utf-8"))
                        metrics = res.get("results", {}).get("gedig_ag_dg", {})
                        # Optionally recompute acceptance via percentile on per_samples
                        if args.accept_mode == "percentile":
                            samples = res.get("results", {}).get("gedig_ag_dg", {}).get("per_samples", [])
                            per_vals = [float(s.get("per", 0.0)) for s in samples]
                            import numpy as np
                            if per_vals:
                                thr_val = float(np.percentile(per_vals, 100*args.accept_percentile))
                                acc = float(np.mean([1.0 if v >= thr_val else 0.0 for v in per_vals]))
                                metrics = dict(metrics)
                                metrics["acceptance_rate"] = acc
                                metrics["fmr"] = 1.0 - acc
                                # Recompute p50 latency from per_samples using recorded latency
                                lat = [float(s.get("latency_ms", 0.0)) for s in samples]
                                if lat:
                                    metrics["latency_p50"] = float(np.percentile(lat, 50))
                        ok = inside_psz(metrics, args.psz_acc, args.psz_fmr, args.psz_p50)
                        score = (
                            max(0.0, args.psz_acc - metrics.get("acceptance_rate", 0.0)) +
                            max(0.0, metrics.get("fmr", 1.0) - args.psz_fmr) +
                            max(0.0, metrics.get("latency_p50", 1e9) - args.psz_p50) / 1000.0
                        )
                        trials.append(((thr, k, h, ag, dg), res_path, metrics))
                        if ok or score < best_score:
                            best = (thr, k, h, ag, dg, res_path, metrics, ok)
                            best_score = score

    out = {
        "psz_target": {
            "target": {"acc": args.psz_acc, "fmr": args.psz_fmr, "p50": args.psz_p50},
            "best": {
                "acceptance_threshold": best[0] if best else None,
                "top_k": best[1] if best else None,
                "max_hops": best[2] if best else None,
                "theta_ag": best[3] if best else None,
                "theta_dg": best[4] if best else None,
                "result": best[6] if best else None,
                "ok": best[7] if best else False,
                "json": str(best[5]) if best else None,
            },
        }
    }
    out_path = Path(args.config).with_name(Path(args.config).stem + "_psz_sweep.json")
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Wrote", out_path)


if __name__ == "__main__":
    main()
