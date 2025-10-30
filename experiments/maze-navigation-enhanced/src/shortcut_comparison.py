"""Compare shortcut_candidate events vs others on path_shortening_mean and multihop_raw gradient.

Outputs:
  - Summary stats (mean/median) for path_shortening_mean per group
  - Distribution of first-hop to last-hop raw delta per group
  - Saves matplotlib plots (hist + box) under ./logs/analysis (created if absent)
  - Emits a Markdown summary to stdout (copy-pastable)
"""
from __future__ import annotations
import os, sys, re
from typing import Dict, List, Tuple
import numpy as np
import math

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    plt = None

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'navigation'))
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
for p in [PARENT_DIR, BASE_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from complex_maze_branch_probe_v2 import build_maze  # type: ignore
from navigation.maze_navigator import MazeNavigator  # type: ignore

def _extract_sorted(seq: Dict) -> List[float]:
    parsed = []
    for k,v in seq.items():
        if isinstance(k,int):
            idx = k
        else:
            m = re.search(r'(\d+)', str(k))
            idx = int(m.group(1)) if m else 10_000 + len(parsed)
        parsed.append((idx, float(v)))
    parsed.sort(key=lambda t: t[0])
    return [x for _,x in parsed]

def main():
    maze, start, goal = build_maze()
    nav = MazeNavigator(
        maze=maze,
        start_pos=start,
        goal_pos=goal,
        wiring_strategy='simple',
        use_escalation=True,
        escalation_threshold='dynamic',
        dynamic_escalation=True,
        dynamic_offset=0.06,
        gedig_threshold=0.25,
        backtrack_threshold=-0.05,
    )
    nav.run(max_steps=3500)

    shortcut_rows = []
    other_rows = []

    for r in nav.gedig_structural:
        psm = r.get('path_shortening_mean')
        raw = r.get('multihop_raw')
        if raw:
            vec = _extract_sorted(raw)
            grad = vec[0]-vec[-1] if len(vec) > 1 else 0.0
        else:
            grad = math.nan
        row = {
            'step': r['step'],
            'psm': psm if psm is not None else math.nan,
            'raw_grad': grad,
            'len_raw': (len(raw) if raw else 0),
            'shortcut': bool(r.get('shortcut'))
        }
        if row['shortcut']:
            shortcut_rows.append(row)
        else:
            other_rows.append(row)

    def stats(rows: List[Dict]) -> Dict[str,float]:
        arr_psm = np.array([x['psm'] for x in rows if not math.isnan(x['psm'])])
        arr_grad = np.array([x['raw_grad'] for x in rows if not math.isnan(x['raw_grad'])])
        return {
            'count': len(rows),
            'psm_mean': float(arr_psm.mean()) if arr_psm.size else math.nan,
            'psm_median': float(np.median(arr_psm)) if arr_psm.size else math.nan,
            'grad_mean': float(arr_grad.mean()) if arr_grad.size else math.nan,
            'grad_median': float(np.median(arr_grad)) if arr_grad.size else math.nan,
        }

    s_stats = stats(shortcut_rows)
    o_stats = stats(other_rows)

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'logs','analysis'))
    os.makedirs(out_dir, exist_ok=True)

    if plt:
        # Histogram path_shortening_mean
        plt.figure(figsize=(6,4))
        plt.hist([x['psm'] for x in shortcut_rows if not math.isnan(x['psm'])], bins=30, alpha=0.6, label='shortcut')
        plt.hist([x['psm'] for x in other_rows if not math.isnan(x['psm'])], bins=30, alpha=0.6, label='other')
        plt.legend(); plt.title('path_shortening_mean distribution'); plt.xlabel('psm'); plt.ylabel('count')
        plt.tight_layout(); plt.savefig(os.path.join(out_dir,'psm_hist.png'))

        # Boxplot raw grad
        plt.figure(figsize=(4,4))
        data = [
            [x['raw_grad'] for x in shortcut_rows if not math.isnan(x['raw_grad'])],
            [x['raw_grad'] for x in other_rows if not math.isnan(x['raw_grad'])]
        ]
        plt.boxplot(data, labels=['shortcut','other'])
        plt.title('multihop_raw gradient (hop0-hopN)')
        plt.tight_layout(); plt.savefig(os.path.join(out_dir,'raw_grad_box.png'))

    # Markdown summary
    md = []
    md.append('# Shortcut vs Other Comparison')
    md.append('')
    md.append('## Counts')
    md.append(f"- shortcut rows: {s_stats['count']}")
    md.append(f"- other rows: {o_stats['count']}")
    md.append('')
    md.append('## path_shortening_mean')
    md.append('| group | mean | median |')
    md.append('|-------|------|--------|')
    md.append(f"| shortcut | {s_stats['psm_mean']:.4f} | {s_stats['psm_median']:.4f} |")
    md.append(f"| other | {o_stats['psm_mean']:.4f} | {o_stats['psm_median']:.4f} |")
    md.append('')
    md.append('## multihop_raw gradient (hop0 - hopN)')
    md.append('| group | mean | median |')
    md.append('|-------|------|--------|')
    md.append(f"| shortcut | {s_stats['grad_mean']:.4f} | {s_stats['grad_median']:.4f} |")
    md.append(f"| other | {o_stats['grad_mean']:.4f} | {o_stats['grad_median']:.4f} |")
    md.append('')
    md.append('Images saved under logs/analysis: psm_hist.png, raw_grad_box.png')

    print('\n'.join(md))

if __name__ == '__main__':
    main()
