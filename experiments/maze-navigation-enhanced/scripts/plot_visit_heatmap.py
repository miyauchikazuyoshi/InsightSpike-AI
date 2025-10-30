#!/usr/bin/env python3
"""Plot visit frequency heatmap from multihop record JSON."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def build_heatmap(rows):
    max_row = max(row.get("row", 0) for row in rows)
    max_col = max(row.get("col", 0) for row in rows)
    grid = np.zeros((max_row + 1, max_col + 1), dtype=int)

    for row in rows:
        r = row.get("row")
        c = row.get("col")
        if r is None or c is None:
            continue
        grid[r, c] += 1
    return grid


def plot_heatmap(grid, output_path):
    plt.figure(figsize=(6, 6))
    sns.heatmap(grid, cmap="magma", cbar=True)
    plt.title("Visit frequency heatmap")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot visit heatmap from JSON record")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input {args.input} not found")

    with args.input.open() as f:
        payload = json.load(f)

    rows = payload.get("rows")
    if not rows:
        raise SystemExit("No rows found in JSON input")

    grid = build_heatmap(rows)
    plot_heatmap(grid, args.output)
    print(f"Saved visit heatmap to {args.output}")


if __name__ == "__main__":
    main()
