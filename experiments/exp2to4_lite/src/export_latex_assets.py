"""Export latest results figures and summary to docs/paper/figures for LaTeX use."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List


def _latest_result(results_dir: Path, prefix: str) -> Path | None:
    files = sorted(results_dir.glob(f"{prefix}_*.json"))
    return files[-1] if files else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Copy figures and summary to docs/paper/figures")
    ap.add_argument("--results-dir", type=Path, default=Path("experiments/exp2to4_lite/results"))
    ap.add_argument("--figs-dir", type=Path, default=Path("experiments/exp2to4_lite/results/figs"))
    ap.add_argument("--paper-figs", type=Path, default=Path("docs/paper/figures"))
    ap.add_argument("--prefix", type=str, default="exp23_paper")
    args = ap.parse_args()

    args.paper_figs.mkdir(parents=True, exist_ok=True)

    latest = _latest_result(args.results_dir, args.prefix)
    if latest is None:
        raise SystemExit("No results found with given prefix")

    # Copy figures with the same stem
    stem = latest.stem
    fig_names = [
        f"{stem}_psz_curve.png",
        f"{stem}_latency_curve.png",
        f"{stem}_gating_timeseries.png",
    ]
    copied: List[Path] = []
    for fn in fig_names:
        src = args.figs_dir / fn
        if src.exists():
            dst = args.paper_figs / fn
            shutil.copy2(src, dst)
            copied.append(dst)

    # Copy summary CSV if present
    summary = sorted(args.results_dir.glob("exp23_summary_*.csv"))
    if summary:
        dst = args.paper_figs / summary[0].name
        shutil.copy2(summary[0], dst)
        copied.append(dst)

    # Write a TeX snippet listing included figures
    tex = args.paper_figs / "exp23_exports.tex"
    with tex.open("w", encoding="utf-8") as fh:
        fh.write("% Auto-generated figure includes\n")
        for p in copied:
            rel = p.name
            if rel.endswith(".png"):
                fh.write("\\begin{figure}[H]\n\\centering\n")
                fh.write(f"\\includegraphics[width=0.8\\linewidth]{{{rel}}}\n")
                fh.write(f"\\caption{{{rel}}}\n\\end{figure}\n\n")
        if summary:
            fh.write("% Summary CSV copied for reference: "+summary[0].name+"\n")

    print("Exported:")
    for p in copied:
        print(" -", p)
    print("TeX snippet:", tex)


if __name__ == "__main__":
    main()

