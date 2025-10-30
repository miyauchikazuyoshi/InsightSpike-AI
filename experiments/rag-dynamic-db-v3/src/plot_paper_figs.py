"""
Plot figures for the paper (RAG):
 - fig6_rag_performance.pdf: PER (mean±95% CI) and acceptance rate bars
 - fig7_psz.pdf: Acceptance vs PER scatter with PSZ guideline box (conceptual)

Inputs (existing results in this repo):
 - results/rag_prompt_impact/prompt_impact_*.json (list of per-query dicts)
 - results/decision_analysis/summary_*.json (acceptance summary dict)

Outputs:
 - docs/paper/figures/fig6_rag_performance.pdf
 - docs/paper/figures/fig7_psz.pdf
"""
from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
FIG_DIR = Path(__file__).resolve().parents[4] / "docs" / "paper" / "figures"


def _latest(path_glob: str) -> str | None:
    files = sorted(glob.glob(path_glob))
    return files[-1] if files else None


def load_prompt_impact() -> List[dict]:
    p = _latest(str(RES / "rag_prompt_impact" / "prompt_impact_results_*.json"))
    if not p:
        return []
    with open(p, "r") as f:
        return json.load(f)


def compute_per_stats(impacts: List[dict]) -> Tuple[float, float, int]:
    """Compute PER mean and 95% CI using 1-hop as baseline.

    Returns: (mean, ci_95, n)
    """
    pers = []
    for r in impacts:
        base = r.get("prompt_1hop_length") or r.get("prompt_2hop_length")
        enhanced = r.get("prompt_enhanced_length")
        if isinstance(base, (int, float)) and isinstance(enhanced, (int, float)) and base > 0:
            pers.append(enhanced / base)
    if not pers:
        return (float("nan"), float("nan"), 0)
    arr = np.array(pers, dtype=float)
    mean = float(arr.mean())
    # Normal approx 95% CI
    ci = 1.96 * float(arr.std(ddof=1)) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
    return mean, ci, len(arr)


def load_acceptance_rate() -> Tuple[float, int]:
    p = _latest(str(RES / "decision_analysis" / "summary_*.json"))
    if not p:
        return float("nan"), 0
    with open(p, "r") as f:
        d = json.load(f)
    rate = float(d.get("acceptance_rate", 0.0)) / 100.0 if d.get("acceptance_rate", 1) > 1 else float(d.get("acceptance_rate", 0.0))
    n = int(d.get("total_queries", 0))
    return rate, n


def plot_rag_performance(per_mean: float, per_ci: float, per_n: int, acc_rate: float, acc_n: int):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))

    labels = ["PER", "Acceptance"]
    means = [per_mean, acc_rate]
    cis = [per_ci, 0]  # Acceptance: no CI from summary → omit

    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=cis, capsize=6, color=["#4C72B0", "#55A868"], alpha=0.9)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, max(1.0, max(means) * 1.2))
    ax.set_ylabel("Score (ratio)")
    ax.set_title("RAG Performance: PER and Acceptance")

    ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.text(0.02, 0.96, f"PER n={per_n}", transform=ax.transAxes, va="top", fontsize=9)
    ax.text(0.02, 0.90, f"Acceptance n={acc_n}", transform=ax.transAxes, va="top", fontsize=9)

    out = FIG_DIR / "fig6_rag_performance.pdf"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_psz(per_mean: float, acc_rate: float):
    """Conceptual PSZ plot using available dimensions (Acceptance vs PER).
    Shades the region with Acceptance≥0.95; annotates PER.
    """
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 4))

    # Shade acceptance ≥ 0.95 as a horizontal band (PSZ acceptance criterion)
    ax.axhspan(0.95, 1.0, color="#C6E6C3", alpha=0.5, label="Acceptance ≥ 95%")

    ax.scatter([per_mean], [acc_rate], c="#C44E52", s=80, label="Current config")
    ax.set_xlim(0.6, 2.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("PER (Enhanced / Base)")
    ax.set_ylabel("Acceptance rate")
    ax.set_title("PSZ guideline (Acceptance vs PER)")
    ax.legend(loc="lower right")

    out = FIG_DIR / "fig7_psz.pdf"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def main():
    impacts = load_prompt_impact()
    per_mean, per_ci, per_n = compute_per_stats(impacts)
    acc_rate, acc_n = load_acceptance_rate()

    out1 = plot_rag_performance(per_mean, per_ci, per_n, acc_rate, acc_n)
    out2 = plot_psz(per_mean, acc_rate)
    print(f"Saved: {out1}")
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()

