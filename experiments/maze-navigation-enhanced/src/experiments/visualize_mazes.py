#!/usr/bin/env python3
"""Utility to visualize 25x25 maze layouts (large, complex, perfect variants).

Usage examples:
  python visualize_mazes.py --variant large --show
  python visualize_mazes.py --variant complex --save-dir ../../results/maze_layouts
  python visualize_mazes.py --variant both --ascii

Outputs:
  - PNG images (if --save-dir or --show) with wall layout, start/goal markers
  - ASCII representation (if --ascii)
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Import maze generators from test_25x25_maze (safe: main guarded by __name__)
# Local imports (same directory) – no package prefix needed
from maze_layouts import (
        create_large_maze,
        create_complex_maze,
        create_perfect_maze,
        create_ultra_maze,
        LARGE_DEFAULT_START,
        LARGE_DEFAULT_GOAL,
        COMPLEX_DEFAULT_START,
        COMPLEX_DEFAULT_GOAL,
        PERFECT_DEFAULT_START,
        PERFECT_DEFAULT_GOAL,
        ULTRA_DEFAULT_START,
        ULTRA_DEFAULT_GOAL,
)  # type: ignore
    # Backwards compatibility names already imported


def ascii_maze(maze: np.ndarray, start, goal) -> str:
    h, w = maze.shape
    lines = []
    for y in range(h):
        row_chars = []
        for x in range(w):
            if (x, y) == start:
                row_chars.append('S')
            elif (x, y) == goal:
                row_chars.append('G')
            else:
                row_chars.append('#' if maze[y, x] == 1 else '.')
        lines.append(''.join(row_chars))
    return '\n'.join(lines)


def plot_maze(maze: np.ndarray, start, goal, title: str):
    fig, ax = plt.subplots(figsize=(6,6))
    h, w = maze.shape
    for y in range(h):
        for x in range(w):
            if maze[y, x] == 1:
                ax.add_patch(Rectangle((x-0.5, y-0.5), 1, 1, facecolor='black', edgecolor='gray', lw=0.3))
    ax.plot(start[0], start[1], 'go', markersize=8, label='Start')
    ax.plot(goal[0], goal[1], 'ro', markersize=8, label='Goal')
    ax.set_xlim(-0.5, w-0.5)
    ax.set_ylim(-0.5, h-0.5)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.2)
    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize 25x25 maze layouts.')
    parser.add_argument('--variant', choices=['large','complex','perfect','ultra','both','all'], default='all',
                        help='Variant to render: large / complex / perfect / ultra / both (large+complex) / all (all)')
    parser.add_argument('--save-dir', type=str, default=None, help='Directory to save PNG(s).')
    parser.add_argument('--show', action='store_true', help='Display the figure(s).')
    parser.add_argument('--ascii', action='store_true', help='Print ASCII maze(s).')
    args = parser.parse_args()

    variants = []
    if args.variant in ('large','both','all'):
        variants.append(('large', create_large_maze(), LARGE_DEFAULT_START, LARGE_DEFAULT_GOAL))
    if args.variant in ('complex','both','all'):
        variants.append(('complex', create_complex_maze(), COMPLEX_DEFAULT_START, COMPLEX_DEFAULT_GOAL))
    if args.variant in ('perfect','all'):
        variants.append(('perfect', create_perfect_maze(), PERFECT_DEFAULT_START, PERFECT_DEFAULT_GOAL))
    if args.variant in ('ultra','all'):
        variants.append(('ultra', create_ultra_maze(), ULTRA_DEFAULT_START, ULTRA_DEFAULT_GOAL))

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    for name, maze, start, goal in variants:
        if args.ascii:
            print(f"===== {name.upper()} MAZE (ASCII) =====")
            print(ascii_maze(maze, start, goal))
            print()
        fig = plot_maze(maze, start, goal, f"{name.capitalize()} Maze")
        if args.save_dir:
            out_path = os.path.join(args.save_dir, f"{name}_maze.png")
            fig.savefig(out_path, dpi=140, bbox_inches='tight')
            print(f"Saved: {out_path}")
        if not args.show:
            plt.close(fig)
    if args.show:
        plt.show()

if __name__ == '__main__':
    main()
