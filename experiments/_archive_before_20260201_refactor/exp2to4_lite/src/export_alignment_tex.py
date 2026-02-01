"""Export alignment extras to LaTeX table (booktabs)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _latest(globpat: str) -> Path | None:
    files = sorted(Path().glob(globpat))
    return files[-1] if files else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Export alignment table to LaTeX")
    ap.add_argument("--results-dir", type=Path, default=Path("experiments/exp2to4_lite/results"))
    ap.add_argument("--paper-figs", type=Path, default=Path("docs/paper/figures"))
    args = ap.parse_args()

    args.paper_figs.mkdir(parents=True, exist_ok=True)
    aln = _latest(str(args.results_dir / "exp23_paper_*_alignment.json"))
    if aln is None:
        raise SystemExit("No alignment json found")
    data = json.loads(aln.read_text(encoding="utf-8"))
    stats = data.get("stats", {})
    extras = data.get("extras", {})

    out = args.paper_figs / "exp23_alignment_table.tex"
    with out.open("w", encoding="utf-8") as fh:
        fh.write("% Auto-generated alignment summary\n")
        fh.write("\\begin{tabular}{lrr}\\toprule\n")
        fh.write("Metric & Value & Note \\\\ \\midrule\n")
        fh.write("$s_{support}$ & %.4f & mean \\\ \n" % (stats.get('s_support_mean',0.0)))
        fh.write("$s_{random}$ & %.4f & mean \\\ \n" % (stats.get('s_random_mean',0.0)))
        fh.write("$s_{topk}$ & %.4f & mean \\\ \n" % (extras.get('s_topk_mean',0.0)))
        fh.write("$s_{AG\\text{-}pick}$ & %.4f & mean \\\ \n" % (extras.get('s_ag_pick_mean',0.0)))
        fh.write("$\\Delta s_{support-random}$ & %.4f & sign-test $p=%.2e$ \\\ \n" % (stats.get('delta_sr_mean',0.0), stats.get('p_sign_sr',1.0)))
        ci = extras.get('delta_sr_ci95',[0.0,0.0])
        fh.write("Cohen's $d_{support-random}$ & %.3f & 95\\%% CI [%.4f, %.4f] \\\ \n" % (extras.get('cohen_d_sr',0.0), ci[0], ci[1]))
        fh.write("\\bottomrule\\end{tabular}\n")
    print("Wrote", out)


if __name__ == "__main__":
    main()
