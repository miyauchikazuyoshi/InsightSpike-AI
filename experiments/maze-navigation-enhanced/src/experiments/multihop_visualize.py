#!/usr/bin/env python3
"""Visualize maze path + compare hop0(global) vs hop1 geDIG scores.

Produces:
  - CSV: step,row,col,hop0_score,hop1_score,escalated,dead_end,shortcut
  - PNG: line plot (hop0 vs hop1) + difference
  - PNG: maze with traversal path colored by hop0 score (optional simple colormap)

Example:
  python multihop_visualize.py --variant ultra50 --seed 123 --max_steps 300 \
      --outdir experiments/maze-navigation-enhanced/results/multihop_vis
"""
from __future__ import annotations
import os, sys, argparse, csv, json
import numpy as np
import matplotlib.pyplot as plt

# Reuse existing helpers
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
from baseline_vs_simple_plot import run_simple  # type: ignore


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def build_table(structural, path):
    rows = []
    # path[i] corresponds roughly to structural[i]['step'] (step_count at record)
    pos_map = {}
    for idx, pos in enumerate(path):
        pos_map[idx] = pos
    for rec in structural:
        step = rec.get('step')
        hop0 = rec.get('value')
        mh = rec.get('multihop') or {}
        hop1 = mh.get(1)
        multihop_missing = rec.get('multihop_missing') or []
        raw_core = rec.get('raw_core') or {}
        raw_struct_impr = raw_core.get('structural_improvement') if isinstance(raw_core, dict) else None
        raw_ig = raw_core.get('ig_value') if isinstance(raw_core, dict) else None
        raw_gedig = raw_core.get('gedig_value') if isinstance(raw_core, dict) else None
        pos = pos_map.get(step, (-1,-1))
        rows.append({
            'step': step,
            'row': pos[0],
            'col': pos[1],
            'hop0_score': hop0,
            'hop1_score': hop1,
            'escalated': int(bool(rec.get('escalated'))),
            'dead_end': int(bool(rec.get('dead_end'))),
            'shortcut': int(bool(rec.get('shortcut'))),
            'hop1_approx': int(bool(rec.get('hop1_approx_fallback'))),
            'gedig_mode': rec.get('gedig_mode'),
            'multihop_missing': ';'.join(str(x) for x in multihop_missing) if multihop_missing else '',
            'raw_struct_impr': raw_struct_impr,
            'raw_ig': raw_ig,
            'raw_gedig': raw_gedig
        })
    return rows


def plot_scores(rows, out_png):
    steps = [r['step'] for r in rows]
    hop0 = [r['hop0_score'] for r in rows]
    hop1 = [r['hop1_score'] if r['hop1_score'] is not None else np.nan for r in rows]
    diff = [h1 - h0 if (h1 is not None and h1==h1) else np.nan for h0,h1 in zip(hop0, hop1)]
    fig, ax = plt.subplots(2,1, figsize=(10,6), sharex=True)
    ax[0].plot(steps, hop0, label='hop0 (base)', linewidth=1.2)
    ax[0].plot(steps, hop1, label='hop1', linewidth=1.0, alpha=0.8)
    ax[0].set_ylabel('Score'); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(steps, diff, label='hop1-hop0', color='purple', linewidth=1.0)
    ax[1].axhline(0, color='#888', linestyle='--', linewidth=0.8)
    ax[1].set_xlabel('Step'); ax[1].set_ylabel('Delta'); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_maze(maze, path, rows, out_png):
    # Normalize hop0 for color
    hop0_scores = [r['hop0_score'] for r in rows if r['hop0_score'] is not None]
    if not hop0_scores:
        hop0_min, hop0_max = 0.0, 1.0
    else:
        hop0_min, hop0_max = min(hop0_scores), max(hop0_scores)
        if hop0_max - hop0_min < 1e-9:
            hop0_max = hop0_min + 1e-6
    norm = lambda v: (v - hop0_min)/(hop0_max - hop0_min)
    score_by_step = {r['step']: r['hop0_score'] for r in rows}
    # Build color path segments
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(maze, cmap='binary')
    for i in range(1, len(path)):
        r0,c0 = path[i-1]; r1,c1 = path[i]
        s = score_by_step.get(i, None)
        if s is None:
            color = (0.3,0.3,0.3,0.4)
        else:
            t = norm(s)
            color = plt.cm.inferno(t)
        ax.plot([c0,c1],[r0,r1], color=color, linewidth=1.2)
    ax.set_title('Maze path colored by hop0 score')
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', default='ultra50')
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--max_steps', type=int, default=400)
    ap.add_argument('--gedig_threshold', type=float, default=0.3)
    ap.add_argument('--backtrack_threshold', type=float, default=-0.2)
    ap.add_argument('--temperature', type=float, default=0.1)
    ap.add_argument('--escalation_threshold', type=float, default=1.0, help='High threshold to force escalation (base_score < threshold)')
    ap.add_argument('--force_multihop', action='store_true')
    ap.add_argument('--max_hops', type=int, default=2)
    ap.add_argument('--gedig_mode', choices=['legacy','core_raw'], default='legacy')
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()

    ensure_dir(args.outdir)

    rec = run_simple(args.variant, args.seed, args.max_steps, args.temperature,
                     args.gedig_threshold, args.backtrack_threshold,
                     include_trace=True, include_structural=True,
                     use_escalation=True, escalation_threshold=args.escalation_threshold,
                     force_multihop=args.force_multihop, max_hops=args.max_hops,
                     gedig_mode=args.gedig_mode)

    maze_meta = {
        'variant': args.variant,
        'seed': args.seed,
        'max_steps': args.max_steps,
        'goal_reached': rec.get('goal_reached'),
        'path_length': rec.get('path_length'),
        'unique_positions': rec.get('unique_positions')
    }

    structural = rec.get('gedig_structural', [])
    path = rec.get('path', [])
    # We don't have maze array in this context; rely on baseline_vs_simple_plot internal build? Not exported.
    # Instead embed only path-based visualization without walls if maze not available.
    maze = None
    # Attempt to reconstruct maze size from path extents
    if path:
        max_r = max(p[0] for p in path)+2
        max_c = max(p[1] for p in path)+2
        maze = np.ones((max_r, max_c))  # placeholder walls
    rows = build_table(structural, path)

    # Write CSV
    csv_path = os.path.join(args.outdir, f"multihop_table_{args.variant}_seed{args.seed}.csv")
    with open(csv_path,'w',newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'step','row','col','hop0_score','hop1_score','escalated','dead_end','shortcut',
            'hop1_approx','gedig_mode','multihop_missing','raw_struct_impr','raw_ig','raw_gedig'
        ])
        w.writeheader(); w.writerows(rows)

    # Plots
    plot_scores(rows, os.path.join(args.outdir, f"scores_{args.variant}_seed{args.seed}.png"))
    if maze is not None:
        plot_maze(maze, path, rows, os.path.join(args.outdir, f"maze_{args.variant}_seed{args.seed}.png"))
    # Save metadata / full record subset
    with open(os.path.join(args.outdir, f"record_subset_{args.variant}_seed{args.seed}.json"),'w') as f:
        json.dump({**maze_meta, 'rows': rows[:200]}, f, indent=2)
    print(f"[multihop_visualize] Wrote CSV & plots to {args.outdir}")

if __name__ == '__main__':
    main()
