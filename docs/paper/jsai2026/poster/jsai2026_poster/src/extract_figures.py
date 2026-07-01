#!/usr/bin/env python3
"""
poster.html から各セクションのSVGを個別ファイルに抽出する。
リポジトリでSVG単位のdiff管理を可能にするためのユーティリティ。
"""
import re
from pathlib import Path

HTML_PATH = Path(__file__).parent.parent / "poster.html"
OUT_DIR = Path(__file__).parent.parent / "figures" / "svg"

# SVG ID and description from HTML comments
SVG_DEFINITIONS = [
    ("01_motivation_brain.svg", "SVG: motivation brain graph"),
    ("02_formula_fepmdl.svg", "SVG: 式+対応表 (文字ずれ修正版)"),
    ("03_figure1_dsp_branching.svg", "SVG: Figure 1 (v10)"),
    ("04_agdg_flow.svg", "SVG: AG/DG flow (v6)"),
    ("06_table2_maze.svg", "SVG: 表2"),
    ("07_table3_regularization.svg", "SVG: 表3 (v2)"),
]

SVG_HEADER = '<?xml version="1.0" encoding="UTF-8"?>\n'


def extract_svgs(html_text: str) -> list[tuple[str, str]]:
    """Return list of (comment, svg_text) pairs in order of appearance."""
    pattern = re.compile(
        r'<!--\s*(SVG:[^-]+?)\s*-->\s*(<svg[\s\S]+?</svg>)',
        re.MULTILINE,
    )
    return [(m.group(1).strip(), m.group(2)) for m in pattern.finditer(html_text)]


def main() -> None:
    html_text = HTML_PATH.read_text(encoding="utf-8")
    extracted = extract_svgs(html_text)

    # Build lookup by trimmed comment
    lookup = {comment: svg for comment, svg in extracted}

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for filename, comment in SVG_DEFINITIONS:
        svg = lookup.get(comment)
        if svg is None:
            print(f"  [SKIP] {filename} — comment '{comment}' not found")
            continue
        out_path = OUT_DIR / filename
        out_path.write_text(SVG_HEADER + svg + "\n", encoding="utf-8")
        print(f"  [OK] {filename} ({len(svg)} bytes)")

    print(f"\nExtracted {len([f for f, _ in SVG_DEFINITIONS if _ in lookup])} SVG files to {OUT_DIR}")


if __name__ == "__main__":
    main()
