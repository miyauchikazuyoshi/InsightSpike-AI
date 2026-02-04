#!/usr/bin/env python3
"""
連続チェックポイント間のΔ（変化量）を分析

仮説: 学習が進むにつれてΔが単調減少する（収束）
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def analyze_deltas(results_file: str = "results/training_dynamics.json"):
    """連続チェックポイント間のΔを計算"""

    with open(results_file) as f:
        data = json.load(f)

    results = data["results"]
    checkpoints = sorted([int(k) for k in results.keys()])

    print("=" * 60)
    print("連続チェックポイント間のΔ分析")
    print("=" * 60)

    # 連続するチェックポイント間のΔを計算
    deltas = []

    for i in range(1, len(checkpoints)):
        prev_step = checkpoints[i - 1]
        curr_step = checkpoints[i]

        prev = results[str(prev_step)]
        curr = results[str(curr_step)]

        if prev is None or curr is None:
            continue

        delta = {
            "from": prev_step,
            "to": curr_step,
            "delta_F": curr["F_mean"] - prev["F_mean"],
            "delta_EPC": curr["EPC_mean"] - prev["EPC_mean"],
            "delta_H": curr["H_mean"] - prev["H_mean"],
            # 変化の絶対量
            "abs_delta_F": abs(curr["F_mean"] - prev["F_mean"]),
            "abs_delta_EPC": abs(curr["EPC_mean"] - prev["EPC_mean"]),
            "abs_delta_H": abs(curr["H_mean"] - prev["H_mean"]),
        }
        deltas.append(delta)

        print(f"\n[{prev_step} → {curr_step}]")
        print(f"  ΔF   = {delta['delta_F']:+.4f} (|Δ| = {delta['abs_delta_F']:.4f})")
        print(f"  ΔEPC = {delta['delta_EPC']:+.4f} (|Δ| = {delta['abs_delta_EPC']:.4f})")
        print(f"  ΔH   = {delta['delta_H']:+.4f} (|Δ| = {delta['abs_delta_H']:.4f})")

    # 可視化
    plot_deltas(deltas, Path("results"))

    return deltas


def plot_deltas(deltas, output_path):
    """Δの推移を可視化"""

    transitions = [f"{d['from']}→{d['to']}" for d in deltas]
    x = range(len(transitions))

    abs_delta_F = [d["abs_delta_F"] for d in deltas]
    abs_delta_EPC = [d["abs_delta_EPC"] for d in deltas]
    abs_delta_H = [d["abs_delta_H"] for d in deltas]

    delta_F = [d["delta_F"] for d in deltas]
    delta_EPC = [d["delta_EPC"] for d in deltas]
    delta_H = [d["delta_H"] for d in deltas]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 絶対変化量 |Δ|
    ax1 = axes[0, 0]
    ax1.bar(x, abs_delta_F, color='blue', alpha=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(transitions, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel("|ΔF|", fontsize=12)
    ax1.set_title("Absolute Change in F (|ΔF|)", fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')

    # 2. EPC変化量
    ax2 = axes[0, 1]
    ax2.bar(x, abs_delta_EPC, color='purple', alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(transitions, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel("|ΔEPC|", fontsize=12)
    ax2.set_title("Absolute Change in EPC (|ΔEPC|)", fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. H変化量
    ax3 = axes[1, 0]
    ax3.bar(x, abs_delta_H, color='green', alpha=0.7)
    ax3.set_xticks(x)
    ax3.set_xticklabels(transitions, rotation=45, ha='right', fontsize=9)
    ax3.set_ylabel("|ΔH|", fontsize=12)
    ax3.set_title("Absolute Change in H (|ΔH|)", fontsize=14)
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. 全成分の推移（符号付き）
    ax4 = axes[1, 1]
    width = 0.25
    x_arr = np.array(x)
    ax4.bar(x_arr - width, delta_F, width, label='ΔF', color='blue', alpha=0.7)
    ax4.bar(x_arr, delta_EPC, width, label='ΔEPC', color='purple', alpha=0.7)
    ax4.bar(x_arr + width, delta_H, width, label='ΔH', color='green', alpha=0.7)
    ax4.set_xticks(x)
    ax4.set_xticklabels(transitions, rotation=45, ha='right', fontsize=9)
    ax4.set_ylabel("Δ (signed)", fontsize=12)
    ax4.set_title("Signed Changes (ΔF, ΔEPC, ΔH)", fontsize=14)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    plt.suptitle("Changes Between Consecutive Checkpoints", fontsize=16)
    plt.tight_layout()

    fig_path = output_path / "delta_analysis.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved figure to {fig_path}")
    plt.close()

    # 収束確認用：|Δ|の推移を折れ線で
    fig2, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x, abs_delta_F, 'o-', label='|ΔF|', color='blue', linewidth=2, markersize=8)
    ax.plot(x, abs_delta_EPC, 's-', label='|ΔEPC|', color='purple', linewidth=2, markersize=8)
    ax.plot(x, abs_delta_H, '^-', label='|ΔH|', color='green', linewidth=2, markersize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(transitions, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel("Absolute Change |Δ|", fontsize=14)
    ax.set_xlabel("Training Transition", fontsize=14)
    ax.set_title("Convergence Check: Are Changes Decreasing?", fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    fig2_path = output_path / "convergence_check.png"
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    print(f"Saved convergence figure to {fig2_path}")
    plt.close()


if __name__ == "__main__":
    analyze_deltas()
