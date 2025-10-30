#!/usr/bin/env python3
"""
Generic Visualization Utilities for InsightSpike-AI
==================================================

Reusable visualization components extracted from experiments directory
for consistent chart generation across the system.

This module provides standardized visualization functions for:
- Performance metrics dashboards
- Comparison charts
- Progress tracking
- Insight detection visualization
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class InsightSpikeVisualizer:
    """Centralized visualization utilities for InsightSpike-AI"""

    def __init__(self, style: str = "default", figsize: Tuple[int, int] = (12, 8)):
        self.style = style
        self.figsize = figsize
        self.setup_style()

    def setup_style(self) -> None:
        """Setup consistent visualization style"""
        plt.style.use(self.style)
        plt.rcParams["figure.figsize"] = self.figsize
        plt.rcParams["font.size"] = 10
        plt.rcParams["font.family"] = "Arial"

        # Setup seaborn for better defaults
        sns.set_style("whitegrid")
        sns.set_palette("husl")

    def create_performance_dashboard(
        self,
        metrics: Dict[str, float],
        title: str = "Performance Dashboard",
        save_path: Optional[str] = None,
    ) -> str:
        """Create a comprehensive performance dashboard"""

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight="bold")

        # 1. Success Rate
        ax1 = axes[0, 0]
        success_rate = metrics.get("success_rate", 0)
        self._create_gauge_chart(ax1, success_rate, "Success Rate", "%")

        # 2. Processing Time
        ax2 = axes[0, 1]
        processing_time = metrics.get("processing_time", 0)
        ax2.bar(["Processing Time"], [processing_time], color="blue", alpha=0.7)
        ax2.set_title("Processing Time", fontweight="bold")
        ax2.set_ylabel("Time (seconds)")

        # 3. Accuracy
        ax3 = axes[0, 2]
        accuracy = metrics.get("accuracy", 0)
        self._create_gauge_chart(ax3, accuracy, "Accuracy", "%")

        # 4. Memory Usage
        ax4 = axes[1, 0]
        memory_usage = metrics.get("memory_usage", 0)
        ax4.bar(["Memory Usage"], [memory_usage], color="orange", alpha=0.7)
        ax4.set_title("Memory Usage", fontweight="bold")
        ax4.set_ylabel("MB")

        # 5. Insight Detection
        ax5 = axes[1, 1]
        insights = metrics.get("insight_detection_count", 0)
        ax5.bar(["Insights"], [insights], color="green", alpha=0.7)
        ax5.set_title("Insights Detected", fontweight="bold")
        ax5.set_ylabel("Count")

        # 6. Efficiency Score
        ax6 = axes[1, 2]
        efficiency = metrics.get("efficiency_score", 0)
        self._create_gauge_chart(ax6, efficiency, "Efficiency Score", "")

        candidate_selection = metrics.get("candidate_selection")
        if isinstance(candidate_selection, dict) and candidate_selection:
            summary_pairs = [
                ("k★", candidate_selection.get("k_star")),
                ("θcand", candidate_selection.get("theta_cand")),
                ("θlink", candidate_selection.get("theta_link")),
                ("cap", candidate_selection.get("k_cap")),
                ("top_m", candidate_selection.get("top_m")),
                ("log k★", candidate_selection.get("log_k_star")),
            ]
            rendered = []
            for label, value in summary_pairs:
                if value is None:
                    continue
                try:
                    rendered.append(f"{label}: {float(value):.3f}")
                except (TypeError, ValueError):
                    rendered.append(f"{label}: {value}")
            fig.text(
                0.5,
                0.02,
                "Candidate Selection → " + ", ".join(rendered),
                ha="center",
                va="bottom",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", edgecolor="gray"),
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            return save_path
        else:
            plt.show()
            return "displayed"

    def create_comparison_chart(
        self,
        data1: List[float],
        data2: List[float],
        labels: List[str],
        title: str = "Performance Comparison",
        label1: str = "InsightSpike",
        label2: str = "Baseline",
        save_path: Optional[str] = None,
    ) -> str:
        """Create side-by-side comparison chart"""

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=self.figsize)

        bars1 = ax.bar(
            x - width / 2, data1, width, label=label1, color="#4ECDC4", alpha=0.8
        )
        bars2 = ax.bar(
            x + width / 2, data2, width, label=label2, color="#FF6B6B", alpha=0.8
        )

        ax.set_title(title, fontweight="bold", fontsize=14)
        ax.set_xlabel("Test Cases")
        ax.set_ylabel("Performance Score")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + height * 0.01,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            return save_path
        else:
            plt.show()
            return "displayed"

    def create_progress_visualization(
        self,
        episodes: List[int],
        rewards: List[float],
        success_indicators: List[bool],
        title: str = "Learning Progress",
        save_path: Optional[str] = None,
    ) -> str:
        """Create learning progress visualization"""

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Reward progression
        ax1.plot(episodes, rewards, color="blue", linewidth=2, alpha=0.7)
        ax1.fill_between(episodes, rewards, alpha=0.3, color="blue")
        ax1.set_title(f"{title} - Reward Progression", fontweight="bold")
        ax1.set_ylabel("Cumulative Reward")
        ax1.grid(True, alpha=0.3)

        # Success rate (moving average)
        window_size = min(10, len(success_indicators) // 4)
        if window_size > 0:
            success_ma = self._moving_average(
                [int(s) for s in success_indicators], window_size
            )
            episodes_ma = episodes[window_size - 1 :]

            ax2.plot(
                episodes_ma,
                success_ma,
                color="green",
                linewidth=2,
                label="Success Rate (MA)",
            )
            ax2.fill_between(episodes_ma, success_ma, alpha=0.3, color="green")

        ax2.set_title("Success Rate Over Time", fontweight="bold")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Success Rate")
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            return save_path
        else:
            plt.show()
            return "displayed"

    def create_insight_spike_visualization(
        self,
        episodes: List[int],
        dged_values: List[float],
        dig_values: List[float],
        spike_episodes: List[int],
        title: str = "Insight Spike Detection",
        save_path: Optional[str] = None,
    ) -> str:
        """Create insight spike detection visualization"""

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        # ΔGED values
        ax1.plot(episodes, dged_values, color="red", linewidth=1, alpha=0.7)
        ax1.axhline(
            y=0.5, color="red", linestyle="--", alpha=0.5, label="ΔGED Threshold"
        )
        ax1.set_title("Δ Global Exploration Difficulty (ΔGED)", fontweight="bold")
        ax1.set_ylabel("ΔGED Value")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # ΔIG values
        ax2.plot(episodes, dig_values, color="blue", linewidth=1, alpha=0.7)
        ax2.axhline(
            y=1.5, color="blue", linestyle="--", alpha=0.5, label="ΔIG Threshold"
        )
        ax2.set_title("Δ Information Gain (ΔIG)", fontweight="bold")
        ax2.set_ylabel("ΔIG Value")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Insight spikes
        ax3.scatter(
            spike_episodes,
            [1] * len(spike_episodes),
            color="gold",
            s=100,
            alpha=0.8,
            label="Insight Spikes",
        )
        ax3.set_title("Detected Insight Moments", fontweight="bold")
        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Insight Detected")
        ax3.set_ylim(-0.5, 1.5)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            return save_path
        else:
            plt.show()
            return "displayed"

    def create_algorithm_comparison(
        self,
        algorithm_results: Dict[str, Dict[str, float]],
        title: str = "Algorithm Performance Comparison",
        save_path: Optional[str] = None,
    ) -> str:
        """Create comprehensive algorithm comparison visualization"""

        algorithms = list(algorithm_results.keys())
        metrics = ["success_rate", "processing_time", "accuracy", "insight_count"]

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight="bold")

        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]

        for i, metric in enumerate(metrics):
            ax = axes[i // 2, i % 2]

            values = [algorithm_results[algo].get(metric, 0) for algo in algorithms]
            bars = ax.bar(
                algorithms, values, color=colors[: len(algorithms)], alpha=0.8
            )

            ax.set_title(metric.replace("_", " ").title(), fontweight="bold")
            ax.set_ylabel("Value")

            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + height * 0.01,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

            # Highlight best performer
            best_idx = (
                np.argmax(values) if metric != "processing_time" else np.argmin(values)
            )
            bars[best_idx].set_edgecolor("gold")
            bars[best_idx].set_linewidth(3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            return save_path
        else:
            plt.show()
            return "displayed"

    def create_radar_chart(
        self,
        categories: List[str],
        values: List[float],
        title: str = "Performance Profile",
        save_path: Optional[str] = None,
    ) -> str:
        """Create radar/spider chart for performance profile"""

        # Number of variables
        N = len(categories)

        # Compute angle for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle

        # Close the plot
        values += values[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))

        # Plot
        ax.plot(angles, values, "o-", linewidth=2, label="Performance", color="#45B7D1")
        ax.fill(angles, values, alpha=0.25, color="#45B7D1")

        # Add category labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)

        # Set y-axis limits
        ax.set_ylim(0, 1)

        # Add title
        ax.set_title(title, fontweight="bold", fontsize=14, pad=20)

        # Add grid
        ax.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            return save_path
        else:
            plt.show()
            return "displayed"

    def _create_gauge_chart(self, ax, value: float, title: str, unit: str) -> None:
        """Create a gauge-style chart for a single metric"""
        # Normalize value to 0-1 if it's a percentage
        if unit == "%":
            display_value = value * 100 if value <= 1 else value
            normalized_value = value if value <= 1 else value / 100
        else:
            display_value = value
            normalized_value = min(value, 1.0)  # Cap at 1 for gauge

        # Create gauge
        theta = np.linspace(0, np.pi, 100)
        radius = 1

        # Background arc
        ax.plot(
            radius * np.cos(theta),
            radius * np.sin(theta),
            color="lightgray",
            linewidth=8,
            alpha=0.3,
        )

        # Value arc
        value_theta = theta[: int(len(theta) * normalized_value)]
        color = (
            "green"
            if normalized_value > 0.7
            else "orange"
            if normalized_value > 0.4
            else "red"
        )
        ax.plot(
            radius * np.cos(value_theta),
            radius * np.sin(value_theta),
            color=color,
            linewidth=8,
        )

        # Center text
        ax.text(
            0,
            -0.3,
            f"{display_value:.1f}{unit}",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )

        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.6, 1.2)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontweight="bold", pad=10)

    def _moving_average(self, data: List[float], window_size: int) -> List[float]:
        """Calculate moving average"""
        if window_size <= 0 or window_size > len(data):
            return data

        moving_avg = []
        for i in range(window_size - 1, len(data)):
            window = data[i - window_size + 1 : i + 1]
            moving_avg.append(sum(window) / window_size)

        return moving_avg

    def save_visualization_config(self, filepath: str) -> None:
        """Save current visualization configuration"""
        config = {
            "style": self.style,
            "figsize": self.figsize,
            "font_size": plt.rcParams["font.size"],
            "font_family": plt.rcParams["font.family"],
        }

        with open(filepath, "w") as f:
            json.dump(config, f, indent=2)

    def load_visualization_config(self, filepath: str) -> None:
        """Load visualization configuration"""
        with open(filepath, "r") as f:
            config = json.load(f)

        self.style = config.get("style", "default")
        self.figsize = tuple(config.get("figsize", [12, 8]))
        plt.rcParams["font.size"] = config.get("font_size", 10)
        plt.rcParams["font.family"] = config.get("font_family", "Arial")

        self.setup_style()


# Convenience functions for quick visualizations


def quick_performance_chart(
    metrics: Dict[str, float],
    title: str = "Performance",
    save_path: Optional[str] = None,
) -> str:
    """Quickly create a performance chart"""
    visualizer = InsightSpikeVisualizer()
    return visualizer.create_performance_dashboard(metrics, title, save_path)


def quick_comparison(
    data1: List[float],
    data2: List[float],
    labels: List[str],
    title: str = "Comparison",
    save_path: Optional[str] = None,
) -> str:
    """Quickly create a comparison chart"""
    visualizer = InsightSpikeVisualizer()
    return visualizer.create_comparison_chart(
        data1, data2, labels, title, save_path=save_path
    )


def quick_progress_chart(
    episodes: List[int], rewards: List[float], save_path: Optional[str] = None
) -> str:
    """Quickly create a progress chart"""
    visualizer = InsightSpikeVisualizer()
    success_indicators = [r > 0 for r in rewards]  # Simple success indicator
    return visualizer.create_progress_visualization(
        episodes, rewards, success_indicators, save_path=save_path
    )
