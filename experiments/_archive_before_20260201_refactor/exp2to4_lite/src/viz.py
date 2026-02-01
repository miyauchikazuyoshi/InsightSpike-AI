"""Visualization utilities: PSZ curves and gating time series."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def plot_psz_curves(results_path: Path, out_dir: Path) -> Tuple[Path, Path]:
    data = _load_json(results_path)
    per_baseline: Dict[str, List[float]] = {}
    lat_per_sample: Dict[str, List[float]] = {}
    # reconstruct per from per_samples; latency from recorded latency if available, else simulate
    for name, payload in data.get("results", {}).items():
        samples = payload.get("per_samples", [])
        per_baseline[name] = [float(s.get("per", 0.0)) for s in samples]
        lat_list: List[float] = []
        for s in samples:
            if "latency_ms" in s:
                try:
                    lat_list.append(float(s.get("latency_ms", 0.0)))
                except Exception:
                    lat_list.append(0.0)
            else:
                st = int(s.get("steps", 1))
                lat_list.append(120.0 + 40.0 * st)
        lat_per_sample[name] = lat_list

    thresholds = np.linspace(0.0, 1.0, 21)
    _ensure_dir(out_dir)
    psz_png = out_dir / (results_path.stem + "_psz_curve.png")
    lat_png = out_dir / (results_path.stem + "_latency_curve.png")

    plt.figure(figsize=(6, 4))
    for name, per_list in per_baseline.items():
        if not per_list:
            continue
        per_arr = np.asarray(per_list)
        acc = [float(np.mean(per_arr >= t)) for t in thresholds]
        fmr = [1.0 - a for a in acc]
        plt.plot(fmr, acc, label=name)
    # PSZ band overlay: acceptance >= 0.95 and FMR <= 0.02
    try:
        import matplotlib.patches as patches
        rect = patches.Rectangle((0.0, 0.95), 0.02, 0.05, linewidth=0, edgecolor=None, facecolor='green', alpha=0.12)
        plt.gca().add_patch(rect)
        plt.text(0.004, 0.955, 'PSZ band', color='green', fontsize=9)
    except Exception:
        pass
    plt.xlabel("FMR (1 - acceptance)")
    plt.ylabel("Acceptance rate")
    plt.title("Operating Curve: Acceptance vs FMR")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(psz_png, dpi=160)
    plt.close()

    # Latency vs acceptance at thresholds
    plt.figure(figsize=(6, 4))
    for name, per_list in per_baseline.items():
        if not per_list:
            continue
        per_arr = np.asarray(per_list)
        lat_arr = np.asarray(lat_per_sample.get(name, []))
        xs = []
        ys = []
        for t in thresholds:
            mask = per_arr >= t
            if not np.any(mask):
                continue
            xs.append(float(np.mean(per_arr >= t)))
            ys.append(float(np.percentile(lat_arr[mask], 50)))
        if xs:
            plt.plot(xs, ys, label=name)
    # PSZ guideline overlays: acceptance=0.95 vertical, latency=200ms horizontal
    try:
        plt.axvline(0.95, color='green', linestyle='--', alpha=0.4)
        plt.axhline(200.0, color='green', linestyle='--', alpha=0.4)
        plt.text(0.952, 205, 'Acc=0.95', color='green', fontsize=8, rotation=90)
        plt.text(0.2, 205, 'P50=200ms', color='green', fontsize=8)
    except Exception:
        pass
    plt.xlabel("Acceptance rate")
    plt.ylabel("Latency P50 (ms)")
    plt.title("Latency vs Acceptance")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(lat_png, dpi=160)
    plt.close()

    return psz_png, lat_png


def plot_gating_timeseries(results_path: Path, out_dir: Path) -> Path:
    data = _load_json(results_path)
    samples = data.get("results", {}).get("gedig_ag_dg", {}).get("per_samples", [])
    g0_series: List[List[float]] = []
    ag_series: List[List[float]] = []
    dg_series: List[List[float]] = []
    for s in samples:
        meta = s.get("metadata", {})
        g0_series.append([float(x) for x in meta.get("gedig_g0_sequence", [])])
        ag_series.append([float(x) for x in meta.get("gedig_ag_sequence", [])])
        dg_series.append([float(x) for x in meta.get("gedig_dg_sequence", [])])

    # pad to max length
    def _pad_to(mat: List[List[float]], fill: float = 0.0) -> np.ndarray:
        L = max((len(r) for r in mat), default=0)
        arr = np.full((len(mat), L), fill, dtype=float)
        for i, r in enumerate(mat):
            arr[i, : len(r)] = r
        return arr

    g0_arr = _pad_to(g0_series, fill=np.nan)
    ag_arr = _pad_to(ag_series, fill=0.0)
    dg_arr = _pad_to(dg_series, fill=0.0)

    mean_g0 = np.nanmean(g0_arr, axis=0) if g0_arr.size else np.array([])
    mean_ag = np.mean(ag_arr, axis=0) if ag_arr.size else np.array([])
    mean_dg = np.mean(dg_arr, axis=0) if dg_arr.size else np.array([])

    _ensure_dir(out_dir)
    fig_path = out_dir / (results_path.stem + "_gating_timeseries.png")
    plt.figure(figsize=(7, 4))
    x = np.arange(len(mean_g0))
    if mean_g0.size:
        plt.plot(x, mean_g0, label="g0", color="C0")
    if mean_ag.size:
        plt.plot(x, mean_ag, label="AG rate", color="C1")
    if mean_dg.size:
        plt.plot(x, mean_dg, label="DG rate", color="C2")
    plt.xlabel("Iteration")
    plt.title("Gating sequences (mean across queries)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close()
    return fig_path
