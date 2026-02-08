"""
Visualize F-trajectory results.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_results(model_name: str, results_dir: str = "results") -> dict:
    """Load results for a model."""
    safe_name = model_name.replace("/", "_")
    path = Path(results_dir) / f"trajectory_{safe_name}.json"
    with open(path) as f:
        return json.load(f)


def plot_f_trajectory(results_dir: str = "results", output_path: str = None):
    """Plot F trajectory for all models."""
    models = ["bert-base-uncased", "gpt2"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for model_idx, model_name in enumerate(models):
        try:
            data = load_results(model_name, results_dir)
        except FileNotFoundError:
            continue

        samples = data["samples"]

        # Plot F trajectory
        ax = axes[0, model_idx]
        for sample in samples:
            f_traj = sample["trajectory"]["F"]
            layers = list(range(len(f_traj)))
            label = sample["text"][:20] + "..."
            ax.plot(layers, f_traj, marker="o", markersize=3, alpha=0.7, label=label)

        ax.axhline(y=0, color="black", linestyle="--", alpha=0.3)
        ax.set_xlabel("Layer Transition")
        ax.set_ylabel("F value")
        ax.set_title(f"{model_name}: F trajectory")
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

        # Plot H and SP
        ax2 = axes[1, model_idx]
        for sample in samples:
            h_traj = sample["trajectory"]["H"]
            sp_traj = sample["trajectory"]["SP"]
            layers_h = list(range(len(h_traj)))
            layers_sp = list(range(len(sp_traj)))

        # Use mean across samples
        mean_h = np.mean([s["trajectory"]["H"] for s in samples], axis=0)
        mean_sp = np.mean([s["trajectory"]["SP"] for s in samples], axis=0)

        ax2_twin = ax2.twinx()
        line1, = ax2.plot(range(len(mean_h)), mean_h, "b-o", markersize=4, label="H (entropy)")
        line2, = ax2_twin.plot(range(len(mean_sp)), mean_sp, "r-s", markersize=4, label="SP")

        ax2.set_xlabel("Layer")
        ax2.set_ylabel("Entropy (H)", color="blue")
        ax2_twin.set_ylabel("Shortcut Purity (SP)", color="red")
        ax2.set_title(f"{model_name}: H and SP evolution")
        ax2.tick_params(axis="y", labelcolor="blue")
        ax2_twin.tick_params(axis="y", labelcolor="red")
        ax2.grid(True, alpha=0.3)

        # Combined legend
        lines = [line1, line2]
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc="upper right")

    plt.tight_layout()

    if output_path is None:
        output_path = Path(results_dir) / "trajectory_visualization.png"

    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()


def plot_layer_comparison(results_dir: str = "results", output_path: str = None):
    """Plot layer-wise comparison between models."""
    models = ["bert-base-uncased", "gpt2"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    colors = {"bert-base-uncased": "blue", "gpt2": "orange"}

    for model_name in models:
        try:
            data = load_results(model_name, results_dir)
        except FileNotFoundError:
            continue

        samples = data["samples"]

        # Stack all F trajectories
        all_f = np.array([s["trajectory"]["F"] for s in samples])
        all_epc = np.array([s["trajectory"]["EPC"] for s in samples])
        all_dh = np.array([s["trajectory"]["delta_H"] for s in samples])

        n_layers = all_f.shape[1]
        layers = np.arange(n_layers)

        color = colors[model_name]

        # F boxplot-like: mean and std
        mean_f = all_f.mean(axis=0)
        std_f = all_f.std(axis=0)
        axes[0].fill_between(layers, mean_f - std_f, mean_f + std_f, alpha=0.2, color=color)
        axes[0].plot(layers, mean_f, "-o", color=color, label=model_name, markersize=4)

        # EPC
        mean_epc = all_epc.mean(axis=0)
        std_epc = all_epc.std(axis=0)
        axes[1].fill_between(layers, mean_epc - std_epc, mean_epc + std_epc, alpha=0.2, color=color)
        axes[1].plot(layers, mean_epc, "-o", color=color, label=model_name, markersize=4)

        # delta_H
        mean_dh = all_dh.mean(axis=0)
        std_dh = all_dh.std(axis=0)
        axes[2].fill_between(layers, mean_dh - std_dh, mean_dh + std_dh, alpha=0.2, color=color)
        axes[2].plot(layers, mean_dh, "-o", color=color, label=model_name, markersize=4)

    axes[0].axhline(y=0, color="black", linestyle="--", alpha=0.3)
    axes[0].set_xlabel("Layer Transition")
    axes[0].set_ylabel("F value")
    axes[0].set_title("F trajectory comparison")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Layer Transition")
    axes[1].set_ylabel("EPC")
    axes[1].set_title("EPC (structure change cost)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].axhline(y=0, color="black", linestyle="--", alpha=0.3)
    axes[2].set_xlabel("Layer Transition")
    axes[2].set_ylabel("ΔH")
    axes[2].set_title("Entropy change (ΔH)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path is None:
        output_path = Path(results_dir) / "layer_comparison.png"

    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    plot_f_trajectory()
    plot_layer_comparison()
    print("\nVisualization complete!")
