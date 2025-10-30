"""
Figure A template: BT aggregation ablation (min / soft-min / sum).
Replace dummy data with real metrics (e.g., BT precision/recall, exploration length).
Outputs: docs/paper/figures/fig_bt_aggregation_template.pdf
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def main():
    labels = ["min", "soft-min", "sum"]
    # Dummy example: higher is better for precision, lower is better for length
    precision = np.array([0.82, 0.79, 0.75])
    length = np.array([1.00, 1.08, 1.15])  # normalized exploration length

    x = np.arange(len(labels))
    w = 0.35
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax2 = ax1.twinx()

    b1 = ax1.bar(x - w/2, precision, width=w, label="BT Precision")
    b2 = ax2.bar(x + w/2, length, width=w, color="#C44E52", alpha=0.8, label="Exploration Length")

    ax1.set_ylim(0, 1.0)
    ax2.set_ylim(0.8, 1.3)
    ax1.set_ylabel("Precision")
    ax2.set_ylabel("Length (normalized)")
    ax1.set_xticks(x, labels)
    ax1.set_title("BT Aggregation Ablation (template)")

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    out = Path("docs/paper/figures/fig_bt_aggregation_template.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    print("Saved:", out)

if __name__ == "__main__":
    main()

