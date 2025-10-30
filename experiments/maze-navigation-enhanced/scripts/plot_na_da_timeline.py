#!/usr/bin/env python3
"""Plot NA -> multi-hop -> BT timeline from multihop record JSON."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_scores(rows):
    steps = []
    g0 = []
    gmin = []
    bt_flags = []

    hop_keys = [k for k in rows[0].keys() if k.startswith("hop") and k != "hop0_score"]

    for row in rows:
        steps.append(row.get("step"))
        g0.append(row.get("hop0_score"))

        scores = []
        for key in hop_keys:
            val = row.get(key)
            if val is not None:
                scores.append(val)
        if scores:
            gmin.append(min(scores))
        else:
            gmin.append(np.nan)

        # treat dead_end flag as BT indicator if present, fallback to shortcut
        bt = row.get("dead_end") or row.get("shortcut") or 0
        bt_flags.append(bool(bt))

    return np.array(steps), np.array(g0, dtype=float), np.array(gmin, dtype=float), np.array(bt_flags)


def plot_timeline(steps, g0, gmin, bt_flags, theta_na, theta_bt, out_path):
    plt.figure(figsize=(10, 4.5))

    plt.plot(steps, g0, label=r"0-hop $g_0$", color="#1f77b4")
    if not np.all(np.isnan(gmin)):
        plt.plot(steps, gmin, label=r"multi-hop $g_{\min}$", color="#d62728")

    plt.axhline(theta_na, color="#2ca02c", linestyle="--", linewidth=1.2, label=r"$\theta_{NA}")
    plt.axhline(theta_bt, color="#9467bd", linestyle=":", linewidth=1.2, label=r"$\theta_{BT}")

    if bt_flags.any():
        bt_steps = steps[bt_flags]
        bt_values = gmin[bt_flags]
        bt_values = np.where(np.isnan(bt_values), g0[bt_flags], bt_values)
        plt.scatter(bt_steps, bt_values, color="#ff7f0e", marker="o", s=40, label="BT trigger")

    plt.xlabel("Step")
    plt.ylabel("geDIG score")
    plt.title("NA → multi-hop → BT timeline")
    plt.legend(loc="best")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot NA/DA timeline from JSON record")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--theta-na", type=float, default=-5.0e-3)
    parser.add_argument("--theta-bt", type=float, default=-1.2e-2)
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input {args.input} not found")

    with args.input.open() as f:
        payload = json.load(f)

    rows = payload.get("rows")
    if not rows:
        raise SystemExit("No rows found in JSON input")

    steps, g0, gmin, bt_flags = load_scores(rows)
    plot_timeline(steps, g0, gmin, bt_flags, args.theta_na, args.theta_bt, args.output)
    print(f"Saved timeline plot to {args.output}")


if __name__ == "__main__":
    main()
