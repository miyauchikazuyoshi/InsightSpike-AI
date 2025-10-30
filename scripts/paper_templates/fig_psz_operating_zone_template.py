"""
Figure C template: PSZ operating zone (Acceptance vs FMR with threshold lines).
Replace dummy points with real (per-config) values. Outputs to docs/paper/figures.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Dummy scatter of (FMR, Acceptance)
    fmr = np.array([0.01, 0.015, 0.03, 0.04, 0.018])
    acc = np.array([0.98, 0.965, 0.93, 0.90, 0.955])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(fmr, acc, c='#4C72B0', s=70)

    # PSZ box: Acceptance ≥ 0.95, FMR ≤ 0.02
    ax.axhline(0.95, color='green', ls='--', alpha=0.7)
    ax.axvline(0.02, color='red', ls='--', alpha=0.7)
    ax.fill_betweenx([0.95, 1.0], 0.0, 0.02, color='#C6E6C3', alpha=0.4)

    ax.set_xlim(0.0, 0.06)
    ax.set_ylim(0.85, 1.0)
    ax.set_xlabel('FMR (False Merge Rate)')
    ax.set_ylabel('Acceptance')
    ax.set_title('PSZ Operating Zone (template)')

    out = Path('docs/paper/figures/fig_psz_operating_zone_template.pdf')
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    print('Saved:', out)

if __name__ == '__main__':
    main()

