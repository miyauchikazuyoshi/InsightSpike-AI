#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib


def _configure_matplotlib() -> None:
    # Use a non-interactive backend and avoid global cache warnings.
    if "MPLCONFIGDIR" not in os.environ:
        cache_dir = Path("tmp/mplconfig")
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(cache_dir)
    matplotlib.use("Agg")


def load_step_records(path: Path) -> List[dict[str, str]]:
    import csv

    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return [dict(row) for row in reader]


def extract_values(
    records: Iterable[dict[str, str]],
    key: str,
) -> List[float]:
    values: List[float] = []
    for row in records:
        raw = row.get(key)
        if raw is None or raw == "":
            continue
        try:
            values.append(float(raw))
        except ValueError:
            continue
    return values


def plot_histogram(
    values: List[float],
    *,
    bins: int,
    title: str,
    xlabel: str,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=bins, color="#4F81BD", edgecolor="black", alpha=0.75)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def summarize_histograms(
    files: List[Path],
    output_dir: Path,
    bins: int,
) -> None:
    merged_delta: List[float] = []
    merged_sp: List[float] = []

    for file_path in files:
        records = load_step_records(file_path)
        delta_vals = extract_values(records, "delta_g")
        sp_vals = extract_values(records, "sp_relative")
        if delta_vals:
            plot_histogram(
                delta_vals,
                bins=bins,
                title=f"Δg distribution ({file_path.name})",
                xlabel="Δg = gmin - g0",
                output_path=output_dir / f"{file_path.stem}_delta_g.png",
            )
            merged_delta.extend(delta_vals)
        if sp_vals:
            plot_histogram(
                sp_vals,
                bins=bins,
                title=f"ΔSP_rel distribution ({file_path.name})",
                xlabel="ΔSP_rel",
                output_path=output_dir / f"{file_path.stem}_sp_rel.png",
            )
            merged_sp.extend(sp_vals)

    if merged_delta:
        plot_histogram(
            merged_delta,
            bins=bins,
            title="Δg distribution (merged)",
            xlabel="Δg = gmin - g0",
            output_path=output_dir / "merged_delta_g.png",
        )
    if merged_sp:
        plot_histogram(
            merged_sp,
            bins=bins,
            title="ΔSP_rel distribution (merged)",
            xlabel="ΔSP_rel",
            output_path=output_dir / "merged_sp_rel.png",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot histograms for per-step logs (Δg and ΔSP_rel)."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="step log CSV files or glob patterns",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory to store generated histograms",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=40,
        help="number of bins for histograms",
    )
    return parser.parse_args()


def expand_inputs(patterns: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if not matches:
            path = Path(pattern)
            if path.exists():
                matches = [str(path)]
        for match in matches:
            files.append(Path(match))
    unique_files = sorted(set(files))
    return unique_files


def main() -> None:
    args = parse_args()
    _configure_matplotlib()
    files = expand_inputs(args.inputs)
    if not files:
        raise SystemExit("No input step log files matched.")
    summarize_histograms(files, args.output_dir, bins=args.bins)


if __name__ == "__main__":
    main()
