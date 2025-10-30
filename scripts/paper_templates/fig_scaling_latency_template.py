"""
Figure B template: Scaling latency percentiles (N vs latency P50/P95/P99).
Replace dummy data with real measurements. Outputs PDF to docs/paper/figures.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def main():
    N = np.array([1e3, 5e3, 1e4, 5e4, 1e5])
    p50 = np.array([80, 95, 110, 180, 260])
    p95 = np.array([120, 150, 190, 280, 380])
    p99 = np.array([150, 190, 240, 340, 460])

    fig, ax = plt.subplots(figsize=(6.2, 4))
    ax.plot(N, p50, marker='o', label='P50')
    ax.plot(N, p95, marker='s', label='P95')
    ax.plot(N, p99, marker='^', label='P99')
    ax.set_xscale('log')
    ax.set_xlabel('Knowledge size N (log)')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Scaling of Latency (template)')
    ax.grid(True, which='both', ls='--', alpha=0.3)
    ax.legend()

    out = Path('docs/paper/figures/fig_scaling_latency_template.pdf')
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    print('Saved:', out)

if __name__ == '__main__':
    main()

