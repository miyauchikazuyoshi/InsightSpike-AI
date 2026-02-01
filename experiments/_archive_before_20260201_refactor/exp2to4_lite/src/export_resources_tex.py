"""Convert latest resources Markdown into LaTeX table (booktabs)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List


def _latest(globpat: str) -> Path | None:
    files = sorted(Path().glob(globpat))
    return files[-1] if files else None


def _escape(s: str) -> str:
    return (
        s.replace("\\", "\\textbackslash{}").replace("_", "\\_")
         .replace("%", "\\%").replace("#", "\\#").replace("&", "\\&")
    )


def md_to_tex(md_path: Path, out_path: Path) -> None:
    lines = md_path.read_text(encoding="utf-8").splitlines()
    table_lines = [ln for ln in lines if ln.startswith("|") and not ln.startswith("|---")]
    if not table_lines:
        out_path.write_text("% No resources table found\n", encoding="utf-8")
        return
    headers = [_escape(h.strip()) for h in table_lines[0].strip("|").split("|")]
    rows = [ln for ln in table_lines[1:] if ln.strip()]
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write("% Auto-generated from resources.md\n\\begin{tabular}{ll}\n\\toprule\n")
        fh.write(f"{headers[0]} & {headers[1]} \\\\ \n")
        fh.write("\\midrule\n")
        for ln in rows:
            c = [_escape(x.strip()) for x in ln.strip("|").split("|")]
            fh.write(f"{c[0]} & {c[1]} \\\\ \n")
        fh.write("\\bottomrule\n\\end{tabular}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export resources table to LaTeX")
    ap.add_argument("--results-dir", type=Path, default=Path("experiments/exp2to4_lite/results"))
    ap.add_argument("--paper-figs", type=Path, default=Path("docs/paper/figures"))
    args = ap.parse_args()

    args.paper_figs.mkdir(parents=True, exist_ok=True)
    md = _latest(str(args.results_dir / "exp23_paper_*_resources.md"))
    if md is None:
        raise SystemExit("No resources markdown found")
    out = args.paper_figs / "exp23_resources_table.tex"
    md_to_tex(md, out)
    print("Wrote", out)


if __name__ == "__main__":
    main()
