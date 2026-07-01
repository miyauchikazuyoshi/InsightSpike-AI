#!/usr/bin/env python3
"""
poster.html (base64埋め込み版) を元に、
poster_linked.html (外部ファイル参照版) を生成する。

- インラインSVG → <object data="figures/svg/XX.svg"> で外部参照
- base64 PNG → <img src="figures/figX_XX.png"> で外部参照

リポジトリでのSVG個別編集・差分管理を可能にする。
"""
import base64
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
HTML_IN = ROOT / "poster.html"
HTML_OUT = ROOT / "poster_linked.html"
FIG_DIR = ROOT / "figures"
SVG_DIR = FIG_DIR / "svg"

SVG_MAPPING = [
    ("SVG: motivation brain graph", "svg/01_motivation_brain.svg"),
    ("SVG: 式+対応表 (文字ずれ修正版)", "svg/02_formula_fepmdl.svg"),
    ("SVG: Figure 1 (v10)", "svg/03_figure1_dsp_branching.svg"),
    ("SVG: AG/DG flow (v6)", "svg/04_agdg_flow.svg"),
    ("SVG: 表2", "svg/06_table2_maze.svg"),
    ("SVG: 表3 (v2)", "svg/07_table3_regularization.svg"),
]


def identify_png(b64_head: str) -> str | None:
    """base64 PNG の先頭部分から元画像ファイルを特定する。"""
    for png_name in ("fig3_maze.png", "fig4_bert.png"):
        png_path = FIG_DIR / png_name
        if not png_path.exists():
            continue
        png_b64 = base64.b64encode(png_path.read_bytes()).decode()
        if png_b64[:200] == b64_head[:200]:
            return png_name
    return None


def main() -> None:
    html = HTML_IN.read_text(encoding="utf-8")

    # 1. Replace inline SVGs with <object> tags (comment-based matching)
    for comment, rel_path in SVG_MAPPING:
        # Escape special regex chars in comment
        c_escaped = re.escape(comment)
        pattern = re.compile(
            r'(<!--\s*' + c_escaped + r'\s*-->\s*)<svg[\s\S]+?</svg>',
            re.MULTILINE,
        )
        replacement = r'\1<object type="image/svg+xml" data="figures/' + rel_path + r'" class="linked-svg"></object>'
        html, count = pattern.subn(replacement, html)
        if count == 0:
            print(f"  [WARN] no match for '{comment}'")
        else:
            print(f"  [OK] SVG {rel_path}")

    # 2. Replace base64 PNG data URIs with relative paths
    # Pattern: src="data:image/png;base64,XXXXXX..."
    png_pattern = re.compile(r'src="data:image/png;base64,([A-Za-z0-9+/=]+)"')
    def replace_png(m: re.Match[str]) -> str:
        b64 = m.group(1)
        png_name = identify_png(b64)
        if png_name:
            print(f"  [OK] PNG {png_name}")
            return f'src="figures/{png_name}"'
        print(f"  [WARN] unknown PNG base64 (first 50 chars: {b64[:50]}...)")
        return m.group(0)

    html = png_pattern.sub(replace_png, html)

    # 3. Add CSS for linked SVG to maintain display
    linked_svg_css = """
.linked-svg {
  display: block;
  width: 100%;
  height: auto;
}
"""
    html = html.replace(
        ".figure-wrap svg { display: block; width: 100%; height: auto; }",
        ".figure-wrap svg { display: block; width: 100%; height: auto; }\n"
        + linked_svg_css,
    )

    HTML_OUT.write_text(html, encoding="utf-8")
    print(f"\nWrote {HTML_OUT} ({HTML_OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
