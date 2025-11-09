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
    root = Path("experiments/maze-query-hub-prototype/results/paper_grid").resolve()
    targets: List[Tuple[int, Path]] = []
    import re
    for size in (15, 25, 51):
        # pick the largest step budget available for L3
        # Note: Use a correctly-escaped dot and digit group in a raw f-string.
        # The previous pattern over-escaped ('\\\.json'), which looked for a
        # literal backslash before '.json' and never matched.
        patt = re.compile(rf"^_{size}x{size}_s(\d+)_l3_summary\.json$")
        candidates = []
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
        print("[warn] no L3 summaries found; expected _{size}x{size}_s250_l3_summary.json under paper_grid")
        return

    sizes: List[int] = []
    succ: List[float] = []
    steps: List[float] = []
    for size, path in targets:
        data = load_summary(path)
        s = data.get("summary", {})
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
    ax1.grid(True, alpha=0.25)

    # Build a combined legend
    lines = [
        plt.Line2D([0], [0], color="#2563eb", marker="o", linewidth=2.5, label="Success rate (%)"),
        plt.Line2D([0], [0], color="#ef4444", marker="s", linewidth=2.5, label="Avg. steps"),
    ]
    ax1.legend(handles=lines, loc="upper left")

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
