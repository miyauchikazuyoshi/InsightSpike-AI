#!/usr/bin/env python3
"""
Render maze layout from a run_summary.json that contains a `maze` grid.

Input JSON format (viewer-compatible):
{
  "maze": [[1,0,1,...], ...],
  "size": 25,
  "seed": 33,
  "path": [[x,y], ...]           # optional
}

Outputs a simple black/white grid PDF/PNG with optional start/goal markers and
an optional overlaid path if present.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def render_maze(maze: np.ndarray, path: list[tuple[int,int]] | None = None,
                title: str | None = None, out: Path | None = None) -> None:
    h, w = maze.shape
    # imshow wants origin at top-left for our coordinate style
    fig, ax = plt.subplots(figsize=(min(8, w/5), min(8, h/5)))
    ax.imshow(maze, cmap='binary', interpolation='nearest', origin='upper')
    ax.set_xticks([]); ax.set_yticks([])
    if path:
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(xs, ys, color='#1f77b4', linewidth=1.0, alpha=0.85)
        # Mark endpoints
        ax.scatter([xs[0]], [ys[0]], c='#2ca02c', s=30, label='start')
        ax.scatter([xs[-1]], [ys[-1]], c='#d62728', s=30, label='goal?')
        ax.legend(loc='upper right', framealpha=0.6, fontsize=8)
    if title:
        ax.set_title(title)
    plt.tight_layout()
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out)
        plt.close(fig)
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input', type=Path, help='run_summary.json')
    ap.add_argument('output', type=Path, help='output image (png/pdf)')
    ap.add_argument('--title', type=str, default=None)
    ap.add_argument('--show-path', action='store_true', help='overlay path if present')
    args = ap.parse_args()

    data = json.loads(args.input.read_text(encoding='utf-8'))
    maze = np.array(data.get('maze'))
    if maze is None or maze.size == 0:
        raise SystemExit('No maze grid found in JSON (expected key "maze").')
    size = int(data.get('size') or maze.shape[0])
    seed = data.get('seed')
    title = args.title or f'Maze layout (size={size}, seed={seed})'
    path = data.get('path') if args.show_path else None
    render_maze(maze, path=path, title=title, out=args.output)
    print(f'Saved maze layout to {args.output}')


if __name__ == '__main__':
    main()

