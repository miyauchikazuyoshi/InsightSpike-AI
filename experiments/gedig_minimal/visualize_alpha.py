"""Visualize alpha sweep results (success rate, mean steps, frontier hist / zero-streak).

Usage:
  python visualize_alpha.py --dirs results_r2_a0_15 results_r2_a0_25 ... --out-dir figures_alpha
"""
from __future__ import annotations
import argparse, json, pathlib
import matplotlib.pyplot as plt
from typing import List, Dict, Any

def load_dir(d: pathlib.Path) -> Dict[str, Any]:
    data = json.loads((d/"summary"/"by_policy.json").read_text())
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dirs', nargs='+', required=True, help='alpha result dirs')
    ap.add_argument('--out-dir', default='figures_alpha')
    args = ap.parse_args()
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for d in args.dirs:
        p = pathlib.Path(d)
        alpha_str = d.split('a')[-1].replace('_', '.')
        try:
            alpha = float(alpha_str)
        except ValueError:
            # fallback extract after last 'a'
            parts = d.split('_a')
            alpha = float(parts[-1].replace('_', '.'))
        jp = load_dir(p)
        if 'gedig_r2' not in jp: # skip incomplete
            continue
        r2 = jp['gedig_r2']
        r1 = jp['gedig_simple']
        rows.append({
            'alpha': alpha,
            'r2_success': r2['success_rate'],
            'r2_steps': r2['mean_steps_success'],
            'r1_success': r1['success_rate'],
            'r1_steps': r1['mean_steps_success'],
        })

    rows.sort(key=lambda x: x['alpha'])

    # Plot success rate
    plt.figure(figsize=(5,3))
    plt.plot([r['alpha'] for r in rows], [r['r2_success'] for r in rows], marker='o', label='gedig_r2')
    plt.hlines(rows[0]['r1_success'], rows[0]['alpha'], rows[-1]['alpha'], colors='gray', linestyles='dashed', label='gedig_simple baseline')
    plt.xlabel('alpha')
    plt.ylabel('success_rate')
    plt.title('Success Rate vs alpha')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out/'success_rate_vs_alpha.png')

    # Plot mean steps (success)
    plt.figure(figsize=(5,3))
    plt.plot([r['alpha'] for r in rows], [r['r2_steps'] for r in rows], marker='o', label='gedig_r2 steps')
    plt.hlines(rows[0]['r1_steps'], rows[0]['alpha'], rows[-1]['alpha'], colors='gray', linestyles='dashed', label='gedig_simple steps')
    plt.xlabel('alpha')
    plt.ylabel('mean_steps_success')
    plt.title('Mean Steps (success) vs alpha')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out/'mean_steps_vs_alpha.png')

    print('Saved figures to', out)

if __name__ == '__main__':
    main()
