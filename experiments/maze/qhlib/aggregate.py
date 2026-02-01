"""Aggregation functions for maze experiment results."""
from __future__ import annotations

from typing import Any, Dict, List

from .models import MazeSummary


def aggregate(runs: List[MazeSummary]) -> Dict[str, float]:
    """Aggregate multiple run summaries into a single result dictionary."""
    if not runs:
        return {
            "success_rate": 0.0,
            "avg_steps": 0.0,
            "avg_edges": 0.0,
            "g0_mean": 0.0,
            "gmin_mean": 0.0,
            "avg_k_star": 0.0,
            "avg_delta_sp": 0.0,
            "avg_delta_sp_min": 0.0,
            "best_hop_mean": 0.0,
            "best_hop_hist_0": 0.0,
            "best_hop_hist_1": 0.0,
            "best_hop_hist_2": 0.0,
            "best_hop_hist_3": 0.0,
            "avg_time_ms_eval": 0.0,
            "p95_time_ms_eval": 0.0,
        }

    def _mean(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    success_rate = sum(1.0 if run.get("success") else 0.0 for run in runs) / len(runs)
    avg_steps = _mean([run.get("steps", 0) for run in runs])
    avg_edges = _mean([run.get("edges", 0) for run in runs])

    g0_values: List[float] = []
    gmin_values: List[float] = []
    k_values: List[float] = []
    sp_values: List[float] = []
    sp_min_values: List[float] = []
    best_hops: List[int] = []
    eval_times: List[float] = []
    ged_min_values: List[float] = []
    psz_samples: List[Dict[str, float]] = []
    for run in runs:
        g0_values.extend(float(v) for v in run.get("g0_series", []))
        gmin_values.extend(float(v) for v in run.get("gmin_series", []))
        k_values.extend(float(v) for v in run.get("k_star_series", []))
        sp_values.extend(float(v) for v in run.get("delta_sp_series", []))
        sp_min_values.extend(float(v) for v in run.get("delta_sp_min_series", []))
        ged_min_values.extend(float(v) for v in run.get("ged_min_series", []))
        best_hops.extend(int(h) for h in run.get("multihop_best_hop", []))
        eval_times.extend(float(t) for t in run.get("eval_time_ms_series", []))
        # Build PSZ samples from accepted flags and eval times (per-step)
        acc_flags = run.get("accepted_series", [])
        lat_values = run.get("eval_time_ms_series", [])
        for a, l in zip(acc_flags, lat_values):
            try:
                psz_samples.append({"accepted": bool(a), "latency_ms": float(l)})
            except Exception:
                continue
    # best hop histogram (0..3; others bucketed to 3+)
    hop_hist = {0: 0, 1: 0, 2: 0, 3: 0}
    for h in best_hops:
        if h <= 0:
            hop_hist[0] += 1
        elif h == 1:
            hop_hist[1] += 1
        elif h == 2:
            hop_hist[2] += 1
        else:
            hop_hist[3] += 1

    def _p95(vals: List[float]) -> float:
        if not vals:
            return 0.0
        vals_sorted = sorted(vals)
        idx = int(max(0, min(len(vals_sorted) - 1, round(0.95 * (len(vals_sorted) - 1)))))
        return float(vals_sorted[idx])

    def _pctl(vals: List[float], q: float) -> float:
        if not vals:
            return 0.0
        v = sorted(vals)
        q = max(0.0, min(1.0, float(q)))
        i = int(round(q * (len(v) - 1)))
        return float(v[i])

    # PSZ summary (acceptance/FMR/P50)
    try:
        from insightspike.metrics.psz import summarize_accept_latency as _psz
        psz = _psz(psz_samples) if psz_samples else None
    except Exception:
        psz = None

    out = {
        "success_rate": success_rate,
        "avg_steps": avg_steps,
        "avg_edges": avg_edges,
        "g0_mean": _mean(g0_values),
        "gmin_mean": _mean(gmin_values),
        "g0_p50": _pctl(g0_values, 0.50),
        "g0_p90": _pctl(g0_values, 0.90),
        "g0_p95": _pctl(g0_values, 0.95),
        "gmin_p50": _pctl(gmin_values, 0.50),
        "gmin_p90": _pctl(gmin_values, 0.90),
        "gmin_p95": _pctl(gmin_values, 0.95),
        "avg_k_star": _mean(k_values),
        "avg_delta_sp": _mean(sp_values),
        "avg_delta_sp_min": _mean(sp_min_values),
        "avg_ged_min_proxy": _mean(ged_min_values),
        "best_hop_mean": _mean([float(h) for h in best_hops]) if best_hops else 0.0,
        "best_hop_hist_0": float(hop_hist[0]),
        "best_hop_hist_1": float(hop_hist[1]),
        "best_hop_hist_2": float(hop_hist[2]),
        "best_hop_hist_3": float(hop_hist[3]),
        "avg_time_ms_eval": _mean(eval_times),
        "p95_time_ms_eval": _p95(eval_times),
    }
    if psz is not None:
        try:
            out.update({
                "psz_acceptance_rate": float(psz.acceptance_rate),
                "psz_fmr": float(psz.fmr),
                "psz_latency_p50_ms": float(psz.latency_p50_ms),
                "psz_inside": bool(psz.inside_psz),
                # Heuristic recommendations for gating thresholds
                # - θ_DG ≈ gminの95パーセンタイル（gmin < θ_DG を ~95% にする）
                "rec_theta_dg_p95": _pctl(gmin_values, 0.95),
            })
        except Exception:
            pass
    return out
