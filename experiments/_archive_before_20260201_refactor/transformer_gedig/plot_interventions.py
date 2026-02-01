#!/usr/bin/env python3
"""
Plot intervention summary: ΔF vs accuracy / confidence correlation.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    summary_path = Path("results/transformer_gedig/intervene_summary.json")
    if not summary_path.exists():
        raise SystemExit(f"missing {summary_path}")
    data = json.loads(summary_path.read_text())

    labels = [
        "baseline",
        "sparsify_top16_last",
        "noise_scale0.05_last",
        "sparsify_top4_all",
        "noise_scale0.5_all",
    ]
    accs = {
        "baseline": data.get("sst2_accuracy_baseline"),
        "sparsify_top16_last": data.get("sst2_accuracy_sparsify_top16_last"),
        "noise_scale0.05_last": data.get("sst2_accuracy_noise_scale0.05_last"),
        "sparsify_top4_all": data.get("sst2_accuracy_sparsify_top4_all"),
        "noise_scale0.5_all": data.get("sst2_accuracy_noise_scale0.5_all"),
    }
    corr = {
        "baseline": data.get("sst2_conf_vs_F_corr_baseline"),
        "sparsify_top16_last": data.get("sst2_conf_vs_F_corr_sparsify_top16_last"),
        "noise_scale0.05_last": data.get("sst2_conf_vs_F_corr_noise_scale0.05_last"),
        "sparsify_top4_all": data.get("sst2_conf_vs_F_corr_sparsify_top4_all"),
        "noise_scale0.5_all": data.get("sst2_conf_vs_F_corr_noise_scale0.5_all"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    acc_vals = [accs.get(l) if accs.get(l) is not None else 0.0 for l in labels]
    corr_vals = [corr.get(l) if corr.get(l) is not None else 0.0 for l in labels]

    axes[0].bar(labels, acc_vals)
    axes[0].set_ylabel("SST2 accuracy")
    axes[0].set_ylim(0, 1)

    axes[1].bar(labels, corr_vals, color="C1")
    axes[1].set_ylabel("conf_vs_F_corr")
    axes[1].set_ylim(-1, 1)

    fig.suptitle("Intervention summary (acc & conf-F corr)")
    fig.tight_layout()
    out = Path("docs/paper/figures/fig_intervention_summary.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"[plot] wrote {out}")


if __name__ == "__main__":
    main()
