"""Export LaTeX tables (booktabs) from summary CSV and ablation MD."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Dict


def _latest(globpat: str) -> Path | None:
    files = sorted(Path().glob(globpat))
    return files[-1] if files else None


def summary_csv_to_tex(csv_path: Path, out_path: Path) -> None:
    rows: List[Dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    # Base columns from summary CSV (+ derived s_PSZ). Include ZSR (zero-hop success) when present or default to 0.0.
    cols = [
        ("baseline", "Method"),
        ("per_mean", "PER"),
        ("acceptance_rate", "Acc"),
        ("zsr", "ZSR"),
        ("fmr", "FMR"),
        ("latency_p50", "P50 (ms)"),
        ("latency_p95", "P95 (ms)"),
        ("s_psz", "$s_{PSZ}$"),
    ]
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write("% Auto-generated from summary CSV\n\\begin{tabular}{lrrrrrrrr}\n\\toprule\n")
        fh.write(" {} & {} & {} & {} & {} & {} & {} & {} \\\\ \n".format(*[c[1] for c in cols]))
        fh.write("\\midrule\n")
        for r in rows:
            # Compute PSZ shortfall: max(0, 0.95-Acc) + max(0, FMR-0.02) + max(0, (P50-200)/1000)
            try:
                acc = float(r.get("acceptance_rate", 0.0))
                fmr = float(r.get("fmr", 0.0))
                p50 = float(r.get("latency_p50", 0.0))
            except Exception:
                acc, fmr, p50 = 0.0, 1.0, 1e9
            s_psz = max(0.0, 0.95 - acc) + max(0.0, fmr - 0.02) + max(0.0, (p50 - 200.0) / 1000.0)

            vals = [
                str(r["baseline"]),
                f"{float(r['per_mean']):.3f}",
                f"{acc:.3f}",
                f"{float(r.get('zsr', 0.0)):.3f}",
                f"{fmr:.3f}",
                f"{p50:.1f}",
                f"{float(r.get('latency_p95', 0.0)):.1f}",
                f"{s_psz:.3f}",
            ]
            fh.write(" {} \\\\ \n".format(" & ".join(vals)))
        fh.write("\\bottomrule\n\\end{tabular}\n")


def ablation_md_to_tex(md_path: Path, out_path: Path) -> None:
    lines = md_path.read_text(encoding="utf-8").splitlines()
    data_lines = [ln for ln in lines if ln.startswith("|") and not ln.startswith("|---")]
    if not data_lines:
        out_path.write_text("% No data\n", encoding="utf-8")
        return
    headers = [h.strip() for h in data_lines[0].strip("|").split("|")]
    body = [ln for ln in data_lines[1:] if ln.strip()]
    # Column alignment: variant (l) + 5 numeric (r)
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write("% Auto-generated from ablation MD\n\\begin{tabular}{lrrrrr}\n\\toprule\n")
        fh.write(" {} \\\\ \n".format(" & ".join(headers)))
        fh.write("\\midrule\n")
        for ln in body:
            cells = [c.strip() for c in ln.strip("|").split("|")]
            fh.write(" {} \\\ \n".format(" & ".join(cells)))
        fh.write("\\bottomrule\n\\end{tabular}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export LaTeX tables to docs/paper/figures")
    ap.add_argument("--results-dir", type=Path, default=Path("experiments/exp2to4_lite/results"))
    ap.add_argument("--paper-figs", type=Path, default=Path("docs/paper/figures"))
    args = ap.parse_args()

    args.paper_figs.mkdir(parents=True, exist_ok=True)

    summary_csv = _latest(str(args.results_dir / "exp23_summary_*.csv"))
    if summary_csv:
        out = args.paper_figs / "exp23_summary_table.tex"
        summary_csv_to_tex(summary_csv, out)
        print("Wrote", out)

    abl_md = _latest(str(args.results_dir / "exp23_paper_*_ablations.md"))
    if abl_md:
        out = args.paper_figs / "exp23_ablation_table.tex"
        ablation_md_to_tex(abl_md, out)
        print("Wrote", out)


if __name__ == "__main__":
    main()
