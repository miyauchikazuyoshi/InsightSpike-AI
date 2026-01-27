#!/usr/bin/env python3
"""
Generate Figure 2 (AG/DG control flow) with a tight canvas (no clipping).

Rationale
- The previous exported image was wider than a column and got scaled down a lot,
  and the right-most box was clipped at export time.
- This script regenerates a clean, column-width-ready figure as both PDF and PNG.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


HERE = Path(__file__).resolve().parent


def add_box(
    ax,
    *,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    fc: str = "white",
    ec: str = "#333333",
    lw: float = 1.6,
    fontsize: int = 9,
    text_color: str = "black",
):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=text_color,
    )
    return box


def add_arrow(ax, *, x1: float, y1: float, x2: float, y2: float, color: str = "#333333", lw: float = 1.2):
    arr = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="->",
        mutation_scale=10,
        linewidth=lw,
        color=color,
    )
    ax.add_patch(arr)
    return arr


def main() -> None:
    # Column-width-friendly figure size (inches). 3.3in ~= 84mm.
    fig, ax = plt.subplots(figsize=(3.3, 1.45))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Layout (normalized coords)
    top_y = 0.62
    h = 0.30
    gap = 0.02
    w = (1.0 - 5 * gap) / 4.0
    x0 = gap

    b1 = add_box(ax, x=x0 + 0 * (w + gap), y=top_y, w=w, h=h, text="observe\n(local view)")
    b2 = add_box(ax, x=x0 + 1 * (w + gap), y=top_y, w=w, h=h, text="candidate\n$S_{link}/S_{cand}$")
    b3 = add_box(ax, x=x0 + 2 * (w + gap), y=top_y, w=w, h=h, text="g0 eval\n(AG gate)")
    b4 = add_box(ax, x=x0 + 3 * (w + gap), y=top_y, w=w, h=h, text="gmin eval\n(DG gate)")

    # Arrows between top boxes
    for left, right in [(b1, b2), (b2, b3), (b3, b4)]:
        add_arrow(
            ax,
            x1=left.get_x() + left.get_width(),
            y1=left.get_y() + left.get_height() / 2,
            x2=right.get_x(),
            y2=right.get_y() + right.get_height() / 2,
            color="#333333",
            lw=1.2,
        )

    # Commit box
    blue = "#1f77b4"
    commit_w = 0.40
    commit_h = 0.26
    commit_x = 0.52 - commit_w / 2
    commit_y = 0.18
    commit = add_box(
        ax,
        x=commit_x,
        y=commit_y,
        w=commit_w,
        h=commit_h,
        text="commit edges\n(update graph)",
        fc="#e8f3ff",
        ec=blue,
        lw=1.8,
        fontsize=10,
        text_color=blue,
    )

    # Downward arrows to commit
    for src in [b3, b4]:
        add_arrow(
            ax,
            x1=src.get_x() + src.get_width() / 2,
            y1=src.get_y(),
            x2=commit.get_x() + commit.get_width() / 2,
            y2=commit.get_y() + commit.get_height(),
            color=blue,
            lw=1.4,
        )

    # Legend
    ax.text(0.82, 0.14, "AG: ambiguity\nDG: shortcut", ha="left", va="bottom", fontsize=8, color="#555555")

    out_pdf = HERE / "agdg_flow.pdf"
    out_png = HERE / "agdg_flow.png"
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[ok] wrote: {out_pdf}")
    print(f"[ok] wrote: {out_png}")


if __name__ == "__main__":
    main()

