"""Visualize frontier distribution (n1/n2 hist) & zero-streak metrics across alpha dirs.

Usage:
  python visualize_frontier.py --dirs results_r2_a0_15 ... --out-dir figures_frontier

It scans raw episode JSON (needs meta with n1_hist, n2_hist, streak metrics).
"""
from __future__ import annotations
import argparse, json, pathlib
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import numpy as np

META_FIELDS = ["n1_hist", "n2_hist", "n2_longest_zero_streak", "n2_mean_zero_streak", "n2_zero_fraction"]


def iter_episode_files(result_dir: pathlib.Path):
    raw = result_dir / 'raw'
    if not raw.exists():
        return
    for size_dir in raw.iterdir():
        if not size_dir.is_dir():
            continue
        for policy_dir in size_dir.iterdir():
            if not policy_dir.is_dir():
                continue
            for f in policy_dir.glob('*.json'):
                yield f


def load_meta(f: pathlib.Path):
    try:
        data = json.loads(f.read_text())
        return data.get('policy'), data.get('meta', {})
    except Exception:
        return None, {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dirs', nargs='+', required=True)
    ap.add_argument('--out-dir', default='figures_frontier')
    ap.add_argument('--policy', default='gedig_r2')
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
            alpha = float(alpha_str.replace('_',''))
        n1_acc = np.zeros(5, dtype=float)
        n2_acc = np.zeros(9, dtype=float)
        longest_list = []
        mean_list = []
        frac_list = []
        count = 0
        for f in iter_episode_files(p):
            policy, meta = load_meta(f)
            if policy != args.policy:
                continue
            if 'n1_hist' not in meta:
                continue
            n1_acc += np.array(meta['n1_hist'])
            n2_acc += np.array(meta['n2_hist'])
            longest_list.append(meta['n2_longest_zero_streak'])
            mean_list.append(meta['n2_mean_zero_streak'])
            frac_list.append(meta['n2_zero_fraction'])
            count += 1
        if count == 0:
            continue
        n1_norm = n1_acc / n1_acc.sum()
        n2_norm = n2_acc / n2_acc.sum()
        rows.append({
            'alpha': alpha,
            'n1': n1_norm.tolist(),
            'n2': n2_norm.tolist(),
            'longest_mean': float(np.mean(longest_list)),
            'zero_frac_mean': float(np.mean(frac_list)),
            'zero_frac_std': float(np.std(frac_list)),
            'episodes': count
        })

    rows.sort(key=lambda x: x['alpha'])

    # Plot n1 distribution heatmap-like
    plt.figure(figsize=(6,3))
    data_n1 = np.array([r['n1'] for r in rows])  # shape (A,5)
    plt.imshow(data_n1, aspect='auto', cmap='Blues')
    plt.yticks(range(len(rows)), [r['alpha'] for r in rows])
    plt.xticks(range(5), [f'n1={i}' for i in range(5)])
    plt.colorbar(label='frequency')
    plt.title(f'n1 distribution vs alpha ({args.policy})')
    plt.tight_layout()
    plt.savefig(out/'n1_distribution.png')

    # Plot n2 distribution
    plt.figure(figsize=(8,3))
    data_n2 = np.array([r['n2'] for r in rows])
    plt.imshow(data_n2, aspect='auto', cmap='Purples')
    plt.yticks(range(len(rows)), [r['alpha'] for r in rows])
    plt.xticks(range(9), [f'n2={i}' for i in range(9)])
    plt.colorbar(label='frequency')
    plt.title(f'n2 distribution vs alpha ({args.policy})')
    plt.tight_layout()
    plt.savefig(out/'n2_distribution.png')

    # Zero fraction curve
    plt.figure(figsize=(5,3))
    plt.errorbar([r['alpha'] for r in rows], [r['zero_frac_mean'] for r in rows], yerr=[r['zero_frac_std'] for r in rows], marker='o')
    plt.xlabel('alpha')
    plt.ylabel('n2 zero fraction (mean ± sd)')
    plt.title('n2 zero fraction vs alpha')
    plt.tight_layout()
    plt.savefig(out/'n2_zero_fraction.png')

    # Longest zero streak mean
    plt.figure(figsize=(5,3))
    plt.plot([r['alpha'] for r in rows], [r['longest_mean'] for r in rows], marker='o')
    plt.xlabel('alpha')
    plt.ylabel('mean longest zero streak')
    plt.title('Longest n2 zero streak (mean) vs alpha')
    plt.tight_layout()
    plt.savefig(out/'n2_longest_zero_streak.png')

    # Save summary JSON
    (out/'frontier_summary.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    print('Saved frontier figures to', out)

if __name__ == '__main__':
    main()
