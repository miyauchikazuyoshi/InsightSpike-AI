#!/usr/bin/env python3
"""Visualize fixed-model geDIG inference results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _finite_xy(values: Sequence[Optional[float]]) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[int] = []
    ys: List[float] = []
    for idx, value in enumerate(values):
        if value is None:
            continue
        if not np.isfinite(value):
            continue
        xs.append(idx)
        ys.append(float(value))
    return np.asarray(xs, dtype=np.int32), np.asarray(ys, dtype=np.float64)


def _latest_run_file(results_dir: Path) -> Path:
    candidates = sorted(results_dir.glob("run_*.json"))
    if not candidates:
        raise FileNotFoundError(f"no run_*.json found in {results_dir}")
    return candidates[-1]


def _load_payload(input_path: Path, results_dir: Optional[Path], latest: bool) -> Tuple[dict, Path]:
    if input_path is not None:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        return payload, input_path

    if results_dir is None:
        raise ValueError("either --input or --results-dir must be provided")

    target_path = _latest_run_file(results_dir) if latest else None
    if target_path is None:
        candidates = sorted(results_dir.glob("run_*.json"))
        if len(candidates) != 1:
            raise ValueError(
                "when --latest is not used, results dir must contain exactly one run_*.json "
                "or specify --input"
            )
        target_path = candidates[0]
    payload = json.loads(target_path.read_text(encoding="utf-8"))
    return payload, target_path


def _condition_items(payload: dict) -> List[Tuple[str, dict]]:
    conditions = payload.get("conditions", {})
    if not isinstance(conditions, dict) or not conditions:
        raise ValueError("payload does not include non-empty 'conditions'")
    return list(conditions.items())


def _structural_metric_key(payload: dict) -> str:
    config = payload.get("config", {})
    if not isinstance(config, dict):
        return "delta_SP"
    term = str(config.get("f_structural_term", "sp")).lower()
    return "delta_B1" if term == "betti1" else "delta_SP"


def _plot_f_curves(payload: dict, output_dir: Path, run_stem: str) -> Path:
    condition_items = _condition_items(payload)
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for idx, (name, condition) in enumerate(condition_items):
        curves = condition.get("mean_curves", {})
        f_curve = curves.get("F", [])
        x, y = _finite_xy(f_curve)
        if len(x) == 0:
            continue
        color = f"C{idx}"
        fit = condition.get("mean_fit", {})
        slope = fit.get("slope")
        r2 = fit.get("r2")
        label = f"{name}"
        if slope is not None and r2 is not None and np.isfinite(slope) and np.isfinite(r2):
            label += f" (slope={float(slope):.4f}, R2={float(r2):.3f})"

        ax.plot(x, y, marker="o", linewidth=2.0, markersize=4, color=color, label=label)
        if slope is not None and fit.get("intercept") is not None and np.isfinite(slope):
            intercept = float(fit["intercept"])
            y_fit = float(slope) * x + intercept
            ax.plot(x, y_fit, linestyle="--", linewidth=1.0, color=color, alpha=0.75)

    ax.axhline(y=0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.4)
    ax.set_title("Mean F Trajectory by Condition")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("F")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()

    output_path = output_dir / f"{run_stem}_f_curve_conditions.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _plot_component_deltas(payload: dict, output_dir: Path, run_stem: str) -> Path:
    condition_items = _condition_items(payload)
    structural_key = _structural_metric_key(payload)
    structural_label = "delta_B1" if structural_key == "delta_B1" else "delta_SP"
    metrics = ["delta_EPC", "delta_H", structural_key]
    ylabels = ["delta_EPC", "delta_H", structural_label]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for ax, metric, ylabel in zip(axes, metrics, ylabels):
        for idx, (name, condition) in enumerate(condition_items):
            curve = condition.get("mean_curves", {}).get(metric, [])
            x, y = _finite_xy(curve)
            if len(x) == 0:
                continue
            ax.plot(x, y, marker="o", linewidth=1.8, markersize=3.5, label=name, color=f"C{idx}")

        ax.axhline(y=0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.3)
        ax.set_xlabel("Layer index")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), frameon=False)
    fig.suptitle("Mean Delta Components by Condition", y=1.02)
    fig.tight_layout()

    output_path = output_dir / f"{run_stem}_delta_components.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_summary_bars(payload: dict, output_dir: Path, run_stem: str) -> Path:
    condition_items = _condition_items(payload)
    names = [name for name, _ in condition_items]

    r2_values = []
    slope_values = []
    mono_values = []
    for _, condition in condition_items:
        fit = condition.get("mean_fit", {})
        r2 = fit.get("r2")
        slope = fit.get("slope")
        mono = condition.get("monotonic_nonincreasing_rate")
        r2_values.append(np.nan if r2 is None else float(r2))
        slope_values.append(np.nan if slope is None else float(slope))
        mono_values.append(np.nan if mono is None else float(mono))

    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))

    axes[0].bar(x, r2_values, color="tab:blue", alpha=0.85)
    axes[0].set_title("Mean-fit R2")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=20, ha="right")
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].bar(x, slope_values, color="tab:orange", alpha=0.85)
    axes[1].axhline(y=0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.3)
    axes[1].set_title("Mean-fit slope")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=20, ha="right")
    axes[1].grid(True, axis="y", alpha=0.25)

    axes[2].bar(x, mono_values, color="tab:green", alpha=0.85)
    axes[2].set_title("Monotonic non-increasing rate")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=20, ha="right")
    axes[2].set_ylim(0, 1)
    axes[2].grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    output_path = output_dir / f"{run_stem}_summary_bars.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _plot_sample_trajectories(payload: dict, output_dir: Path, run_stem: str, max_samples: int) -> Optional[Path]:
    baseline = payload.get("conditions", {}).get("baseline", {})
    samples = baseline.get("samples")
    if not isinstance(samples, list) or not samples:
        return None

    fig, ax = plt.subplots(figsize=(10, 5.8))
    curves = []
    for idx, sample in enumerate(samples[:max_samples]):
        trajectory = sample.get("trajectory", {})
        f_curve = trajectory.get("F", [])
        x, y = _finite_xy(f_curve)
        if len(x) == 0:
            continue
        curves.append((x, y))
        ax.plot(x, y, linewidth=1.1, alpha=0.35, color="tab:gray")

    if not curves:
        plt.close(fig)
        return None

    max_len = max(int(x.max()) + 1 for x, _ in curves)
    stacked = []
    for x, y in curves:
        arr = np.full((max_len,), np.nan, dtype=np.float64)
        arr[x] = y
        stacked.append(arr)
    mean_curve = np.nanmean(np.stack(stacked, axis=0), axis=0)
    finite_idx = np.where(np.isfinite(mean_curve))[0]
    ax.plot(finite_idx, mean_curve[finite_idx], color="tab:red", linewidth=2.4, label="mean F (baseline)")

    ax.axhline(y=0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.35)
    ax.set_title("Baseline Sample F Trajectories")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("F")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    output_path = output_dir / f"{run_stem}_baseline_samples.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _write_text_summary(payload: dict, output_dir: Path, run_stem: str, source_path: Path) -> Path:
    lines = []
    meta = payload.get("metadata", {})
    config = payload.get("config", {})
    model = meta.get("model", "unknown")
    lines.append(f"source: {source_path}")
    lines.append(f"model: {model}")
    if isinstance(config, dict):
        lines.append(f"sp_mode: {config.get('sp_mode', 'n/a')}")
        lines.append(f"f_structural_term: {config.get('f_structural_term', 'n/a')}")
    lines.append("")

    for name, condition in _condition_items(payload):
        fit = condition.get("mean_fit", {})
        grid = condition.get("grid_search_best")
        mono = condition.get("monotonic_nonincreasing_rate")
        lines.append(f"[{name}]")
        lines.append(
            f"  slope={fit.get('slope')}  r2={fit.get('r2')}  monotonic_rate={mono}"
        )
        if isinstance(grid, dict):
            fit_grid = grid.get("fit", {})
            lines.append(
                "  grid_best: "
                f"lambda={grid.get('lambda')} gamma={grid.get('gamma')} "
                f"slope={fit_grid.get('slope')} r2={fit_grid.get('r2')}"
            )
        lines.append("")

    output_path = output_dir / f"{run_stem}_summary.txt"
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize geDIG inference run output")
    parser.add_argument("--input", type=Path, default=None, help="Path to run_*.json")
    parser.add_argument("--results-dir", type=Path, default=None, help="Directory containing run_*.json")
    parser.add_argument("--latest", action="store_true", help="Use latest run_*.json in results dir")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for output figures")
    parser.add_argument("--max-sample-curves", type=int, default=24)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    payload, source_path = _load_payload(args.input, args.results_dir, args.latest)
    run_stem = source_path.stem
    output_dir = args.output_dir or source_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = []
    generated.append(_plot_f_curves(payload=payload, output_dir=output_dir, run_stem=run_stem))
    generated.append(_plot_component_deltas(payload=payload, output_dir=output_dir, run_stem=run_stem))
    generated.append(_plot_summary_bars(payload=payload, output_dir=output_dir, run_stem=run_stem))

    sample_plot = _plot_sample_trajectories(
        payload=payload,
        output_dir=output_dir,
        run_stem=run_stem,
        max_samples=max(1, args.max_sample_curves),
    )
    if sample_plot is not None:
        generated.append(sample_plot)

    generated.append(
        _write_text_summary(
            payload=payload,
            output_dir=output_dir,
            run_stem=run_stem,
            source_path=source_path,
        )
    )

    print(f"[done] source={source_path}")
    for path in generated:
        print(f"[out] {path}")


if __name__ == "__main__":
    main()
