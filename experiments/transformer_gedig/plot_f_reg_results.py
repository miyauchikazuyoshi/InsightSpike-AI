#!/usr/bin/env python3
"""
Plot F-Regularization experiment results.

Creates:
1. Alpha vs Accuracy plot (with error bars)
2. Alpha vs Final F plot
3. Accuracy vs F scatter plot (correlation analysis)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import numpy as np


def load_results(results_dir: Path) -> List[Dict[str, Any]]:
    """Load all results from the experiment directory."""
    all_results_path = results_dir / "all_results.json"
    if all_results_path.exists():
        return json.loads(all_results_path.read_text())

    # Try to collect from individual run directories
    results = []
    for alpha_dir in sorted(results_dir.glob("alpha_*")):
        result_file = alpha_dir / "result.json"
        if result_file.exists():
            results.append(json.loads(result_file.read_text()))
    return results


def plot_alpha_vs_accuracy(results: List[Dict], output_path: Path):
    """Plot alpha vs accuracy with error bars."""
    import pandas as pd

    df = pd.DataFrame(results)
    grouped = df.groupby("alpha").agg({
        "final_accuracy": ["mean", "std", "count"]
    }).reset_index()
    grouped.columns = ["alpha", "acc_mean", "acc_std", "count"]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Use log scale for alpha (skip 0)
    alphas = grouped["alpha"].values
    acc_means = grouped["acc_mean"].values
    acc_stds = grouped["acc_std"].values

    # Plot with error bars
    ax.errorbar(
        range(len(alphas)),
        acc_means,
        yerr=acc_stds,
        marker='o',
        markersize=10,
        linewidth=2,
        capsize=5,
        capthick=2,
    )

    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f"{a}" for a in alphas])
    ax.set_xlabel("Alpha (F-regularization weight)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("F-Regularization: Alpha vs Accuracy", fontsize=14)
    ax.grid(True, alpha=0.3)

    # Highlight best
    best_idx = np.argmax(acc_means)
    ax.scatter([best_idx], [acc_means[best_idx]], color='red', s=200, zorder=5, marker='*')
    ax.annotate(
        f"Best: {acc_means[best_idx]:.4f}",
        (best_idx, acc_means[best_idx]),
        textcoords="offset points",
        xytext=(10, 10),
        fontsize=10,
    )

    # Baseline annotation
    if alphas[0] == 0:
        ax.axhline(y=acc_means[0], color='gray', linestyle='--', alpha=0.7, label='Baseline (α=0)')
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_alpha_vs_f(results: List[Dict], output_path: Path):
    """Plot alpha vs final F value."""
    import pandas as pd

    # Extract F values from final_gedig
    data = []
    for r in results:
        if r.get("final_gedig"):
            data.append({
                "alpha": r["alpha"],
                "f_mean": r["final_gedig"].get("f_mean"),
            })

    if not data:
        print("No F metrics found, skipping F plot")
        return

    df = pd.DataFrame(data)
    grouped = df.groupby("alpha").agg({
        "f_mean": ["mean", "std"]
    }).reset_index()
    grouped.columns = ["alpha", "f_mean", "f_std"]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(
        range(len(grouped)),
        grouped["f_mean"].values,
        yerr=grouped["f_std"].values,
        marker='s',
        markersize=10,
        linewidth=2,
        capsize=5,
        capthick=2,
        color='orange',
    )

    ax.set_xticks(range(len(grouped)))
    ax.set_xticklabels([f"{a}" for a in grouped["alpha"].values])
    ax.set_xlabel("Alpha (F-regularization weight)", fontsize=12)
    ax.set_ylabel("Final F (geDIG gauge)", fontsize=12)
    ax.set_title("F-Regularization: Alpha vs Final F Value", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_accuracy_vs_f_scatter(results: List[Dict], output_path: Path):
    """Scatter plot of accuracy vs F with correlation."""
    data = []
    for r in results:
        if r.get("final_gedig") and r.get("final_accuracy"):
            data.append({
                "alpha": r["alpha"],
                "accuracy": r["final_accuracy"],
                "f_mean": r["final_gedig"].get("f_mean"),
            })

    if len(data) < 3:
        print("Not enough data points for scatter plot")
        return

    accuracies = [d["accuracy"] for d in data]
    f_means = [d["f_mean"] for d in data]
    alphas = [d["alpha"] for d in data]

    # Compute correlation
    correlation = np.corrcoef(accuracies, f_means)[0, 1]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Color by alpha
    scatter = ax.scatter(
        f_means,
        accuracies,
        c=[np.log10(a + 1e-4) for a in alphas],
        cmap='viridis',
        s=100,
        alpha=0.7,
    )

    # Trend line
    z = np.polyfit(f_means, accuracies, 1)
    p = np.poly1d(z)
    f_range = np.linspace(min(f_means), max(f_means), 100)
    ax.plot(f_range, p(f_range), "r--", alpha=0.5, label=f"Trend (r={correlation:.3f})")

    ax.set_xlabel("Final F (geDIG gauge)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"Accuracy vs F Correlation (r={correlation:.3f})", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label("log10(alpha)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def create_summary_table(results: List[Dict], output_path: Path):
    """Create a markdown summary table."""
    import pandas as pd

    df = pd.DataFrame(results)
    grouped = df.groupby("alpha").agg({
        "final_accuracy": ["mean", "std"],
    }).reset_index()
    grouped.columns = ["alpha", "acc_mean", "acc_std"]

    # Add F metrics if available
    f_data = {}
    for r in results:
        alpha = r["alpha"]
        if r.get("final_gedig"):
            if alpha not in f_data:
                f_data[alpha] = []
            f_data[alpha].append(r["final_gedig"].get("f_mean", 0))

    grouped["f_mean"] = grouped["alpha"].apply(
        lambda a: np.mean(f_data.get(a, [0])) if a in f_data else None
    )

    # Generate markdown
    lines = [
        "# F-Regularization Experiment Results\n",
        "| Alpha | Accuracy (mean ± std) | Final F |",
        "|-------|----------------------|---------|",
    ]

    best_acc = grouped["acc_mean"].max()
    for _, row in grouped.iterrows():
        acc_str = f"{row['acc_mean']:.4f} ± {row['acc_std']:.4f}"
        f_str = f"{row['f_mean']:.4f}" if row['f_mean'] else "N/A"
        marker = " **" if row["acc_mean"] == best_acc else ""
        lines.append(f"| {row['alpha']}{marker} | {acc_str}{marker} | {f_str} |")

    # Add conclusions
    baseline = grouped[grouped["alpha"] == 0]
    best = grouped.loc[grouped["acc_mean"].idxmax()]

    lines.extend([
        "",
        "## Key Findings",
        "",
    ])

    if not baseline.empty:
        improvement = best["acc_mean"] - baseline["acc_mean"].values[0]
        lines.append(f"- **Baseline (α=0)**: {baseline['acc_mean'].values[0]:.4f}")
        lines.append(f"- **Best (α={best['alpha']})**: {best['acc_mean']:.4f}")
        lines.append(f"- **Improvement**: {improvement:+.4f} ({improvement/baseline['acc_mean'].values[0]*100:+.2f}%)")

        if improvement > 0:
            lines.append("")
            lines.append("**Conclusion**: F-regularization improves performance, supporting the causal hypothesis.")
        else:
            lines.append("")
            lines.append("**Conclusion**: F-regularization did not improve performance in this setting.")

    output_path.write_text("\n".join(lines))
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot F-regularization results")
    parser.add_argument("--results-dir", type=Path, default=Path("results/transformer_gedig/f_reg"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    results_dir = args.results_dir
    output_dir = args.output_dir or results_dir

    results = load_results(results_dir)
    if not results:
        print(f"No results found in {results_dir}")
        return

    print(f"Loaded {len(results)} results")

    # Create plots
    plot_alpha_vs_accuracy(results, output_dir / "fig_alpha_vs_accuracy.png")
    plot_alpha_vs_f(results, output_dir / "fig_alpha_vs_f.png")
    plot_accuracy_vs_f_scatter(results, output_dir / "fig_accuracy_vs_f_scatter.png")
    create_summary_table(results, output_dir / "RESULTS.md")

    print("\nAll plots generated!")


if __name__ == "__main__":
    main()
