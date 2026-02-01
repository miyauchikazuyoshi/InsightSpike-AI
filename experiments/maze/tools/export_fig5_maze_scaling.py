#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_summary(path: Path) -> Dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> None:
    import re

    # Prefer the aggregated maze_* JSONs under docs/paper/data (v5 paper data).
    data_root = Path("docs/paper/data").resolve()
    targets: List[Tuple[int, Path]] = []
    if data_root.exists():
        for size in (15, 25, 51):
            patt = re.compile(rf"^maze_{size}x{size}_l3_s(\d+)\.json$")
            candidates: List[Tuple[int, Path]] = []
            for p in data_root.glob(f"maze_{size}x{size}_l3_s*.json"):
                m = patt.match(p.name)
                if not m:
                    continue
                steps = int(m.group(1))
                candidates.append((steps, p))
            if candidates:
                # pick the largest step budget available for each size
                candidates.sort(key=lambda t: t[0])
                targets.append((size, candidates[-1][1]))

    # Fallback: use older paper_grid L3 summaries if aggregated data is missing.
    if not targets or len(targets) < 3:
        root = Path("experiments/maze-query-hub-prototype/results/paper_grid").resolve()
        targets = []
        for size in (15, 25, 51):
            patt = re.compile(rf"^_{size}x{size}_s(\d+)_l3_summary\.json$")
            candidates: List[Tuple[int, Path]] = []
            for p in root.glob(f"_{size}x{size}_s*_l3_summary.json"):
                m = patt.match(p.name)
                if not m:
                    continue
                steps = int(m.group(1))
                candidates.append((steps, p))
            if candidates:
                candidates.sort(key=lambda t: t[0])
                targets.append((size, candidates[-1][1]))

    if not targets:
        print("[warn] no maze L3 summaries found; expected maze_*_l3_s*.json under docs/paper/data or _{size}x{size}_s*_l3_summary.json under paper_grid")
        return

    sizes: List[int] = []
    succ: List[float] = []
    steps: List[float] = []
    for size, path in targets:
        data = load_summary(path)
        # Newer aggregated files store metrics at top-level; older ones nest under "summary".
        s = data.get("summary", data)
        sizes.append(size)
        succ.append(float(s.get("success_rate", 0.0)) * 100.0)
        steps.append(float(s.get("avg_steps", 0.0)))

    # Sort by size
    order = sorted(range(len(sizes)), key=lambda i: sizes[i])
    sizes = [sizes[i] for i in order]
    succ = [succ[i] for i in order]
    steps = [steps[i] for i in order]

    fig, ax1 = plt.subplots(figsize=(7.2, 4.2))
    ax2 = ax1.twinx()

    ax1.plot(sizes, succ, color="#2563eb", marker="o", linewidth=2.5, label="Success rate (%)")
    ax2.plot(sizes, steps, color="#ef4444", marker="s", linewidth=2.5, label="Avg. steps")

    ax1.set_xlabel("Maze size (N×N)")
    ax1.set_ylabel("Success rate (%)", color="#2563eb")
    ax2.set_ylabel("Avg. steps", color="#ef4444")
    ax1.set_xticks(sizes)
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.25)

    # Build a combined legend outside the main plot to avoid overlap
    lines = [
        plt.Line2D([0], [0], color="#2563eb", marker="o", linewidth=2.5, label="Success rate (%)"),
        plt.Line2D([0], [0], color="#ef4444", marker="s", linewidth=2.5, label="Avg. steps"),
    ]
    ax1.legend(handles=lines, loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2)

    fig.tight_layout()
    outdir = Path("docs/paper/figures").resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "fig5_maze_scaling.png").unlink(missing_ok=True)
    (outdir / "fig5_maze_scaling.pdf").unlink(missing_ok=True)
    fig.savefig(outdir / "fig5_maze_scaling.png", dpi=160, bbox_inches="tight")
    fig.savefig(outdir / "fig5_maze_scaling.pdf", bbox_inches="tight")
    print("[done] Wrote:", outdir / "fig5_maze_scaling.png")


if __name__ == "__main__":
    main()
