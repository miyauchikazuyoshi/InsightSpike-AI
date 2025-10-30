#!/usr/bin/env python3
"""Quick visualization of a single simple run: maze layout, path, geDIG timeline, and event markers.
Usage:
  python -m experiments.visualize_run --variant ultra50 --seed 123 --steps 800 --ann-upgrade-threshold 200 --flush --show
Outputs:
  PNG files in results/visual/ (maze_path.png, gedig_timeline.png, combined.png)
"""
from __future__ import annotations
import argparse, os, sys, json
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiments.baseline_vs_simple_plot import run_simple  # type: ignore
from experiments.maze_layouts import (
    create_complex_maze, create_ultra_maze, create_large_maze,
    create_ultra_maze_50 as _create_ultra_maze_50,
    create_ultra_maze_50_dense_deadends as _create_ultra_maze_50_dense_deadends,
    create_ultra_maze_50_moderate_deadends as _create_ultra_maze_50_moderate_deadends
)

VARIANTS = {
    'complex': create_complex_maze,
    'ultra': create_ultra_maze,
    'large': create_large_maze,
    'ultra50': _create_ultra_maze_50,
    'ultra50hd': _create_ultra_maze_50_dense_deadends,
    'ultra50md': _create_ultra_maze_50_moderate_deadends,
}

FOCUS_EVENT_TYPES = [
    'ann_upgrade','flush_eviction','rehydration','goal','backtrack_path_plan'
]


def build_maze(variant: str):
    if variant not in VARIANTS:
        raise SystemExit(f"Unsupported variant {variant}")
    return VARIANTS[variant]()


def plot_maze_and_path(ax, maze: np.ndarray, path):
    ax.imshow(maze, cmap='gray_r', interpolation='none')
    if path:
        ys = [p[1] for p in path]
        xs = [p[0] for p in path]
        ax.plot(xs, ys, color='lime', linewidth=1.2, alpha=0.85)
        ax.scatter([xs[0]],[ys[0]], c='blue', s=25, label='start')
        ax.scatter([xs[-1]],[ys[-1]], c='red', s=25, label='end')
    ax.set_title('Maze & Path')
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc='upper right', fontsize=8)


def plot_gedig(ax, gedig_history, events_full):
    if gedig_history:
        ax.plot(range(len(gedig_history)), gedig_history, color='purple', linewidth=0.8)
    ax.set_xlabel('Step')
    ax.set_ylabel('geDIG')
    ax.set_title('geDIG Timeline')
    # Overlay event markers
    for ev in events_full:
        t = ev.get('type')
        if t in FOCUS_EVENT_TYPES:
            step = ev.get('step') or ev.get('s') or None
            if step is None:
                continue
            color = {
                'ann_upgrade':'orange',
                'flush_eviction':'crimson',
                'rehydration':'dodgerblue',
                'goal':'green',
                'backtrack_path_plan':'gold'
            }.get(t,'black')
            ax.axvline(step, color=color, alpha=0.35, linewidth=0.8)
    # Legend proxies
    from matplotlib.lines import Line2D
    proxies = [
        Line2D([0],[0], color='orange', lw=2, label='ann_upgrade'),
        Line2D([0],[0], color='crimson', lw=2, label='flush_eviction'),
        Line2D([0],[0], color='dodgerblue', lw=2, label='rehydration'),
        Line2D([0],[0], color='green', lw=2, label='goal'),
        Line2D([0],[0], color='gold', lw=2, label='backtrack_plan')
    ]
    ax.legend(handles=proxies, fontsize=7, loc='upper right')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', default='ultra50')
    ap.add_argument('--seed', type=int, default=101)
    ap.add_argument('--steps', type=int, default=800)
    ap.add_argument('--ann-upgrade-threshold', type=int, default=200)
    ap.add_argument('--flush', action='store_true')
    ap.add_argument('--flush-interval', type=int, default=80)
    ap.add_argument('--max-in-memory', type=int, default=8000)
    ap.add_argument('--max-in-memory-positions', type=int, default=2500)
    ap.add_argument('--out-dir', default='experiments/maze-navigation-enhanced/results/visual')
    ap.add_argument('--show', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    # baseline_vs_simple_plot expects these globals set; emulate CLI usage
    import experiments.baseline_vs_simple_plot as bmod  # type: ignore
    bmod.GLOBAL_VERBOSITY = 0
    bmod.GLOBAL_PROGRESS_INTERVAL = 100

    record = run_simple(
        args.variant, args.seed, args.steps, 0.1, 0.3, -0.2,
        wiring_strategy_override='query', use_vector_index=True,
        enable_flush=args.flush, flush_interval=args.flush_interval,
        max_in_memory=args.max_in_memory, max_in_memory_positions=args.max_in_memory_positions,
        ann_upgrade_threshold=args.ann_upgrade_threshold,
        ann_backend='hnsw', include_events=True, include_trace=True
    )

    maze = build_maze(args.variant)

    fig, axes = plt.subplots(1,2, figsize=(12,6))
    plot_maze_and_path(axes[0], maze, record.get('path'))
    plot_gedig(axes[1], record.get('gedig_history'), record.get('events_full', []))
    fig.suptitle(f"Variant={args.variant} seed={args.seed} steps={record.get('steps')} goal={record.get('goal_reached')}")
    combined_path = os.path.join(args.out_dir, f"viz_{args.variant}_seed{args.seed}.png")
    fig.tight_layout(rect=[0,0,1,0.95])
    fig.savefig(combined_path, dpi=140)
    print('Saved', combined_path)

    # Save raw record JSON for inspection
    with open(os.path.join(args.out_dir, f"record_{args.variant}_seed{args.seed}.json"),'w') as f:
        json.dump(record, f, indent=2)
    if args.show:
        plt.show()

if __name__ == '__main__':
    main()
