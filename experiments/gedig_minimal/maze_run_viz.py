"""Visualize a single episode (maze + agent path animation).

Generates PNG frames and optionally an animated GIF (if imageio available).

Usage:
  python maze_run_viz.py --policy gedig_r2 --maze-size 8x8 --seed 0 --frontier-weight 0.2 --out-dir viz_run
"""
from __future__ import annotations
import argparse, pathlib, json
import matplotlib.pyplot as plt
from simple_navigator import run_episode, generate_maze

def draw_maze(ax, maze):
    ax.imshow(maze, cmap='binary')
    ax.set_xticks([]); ax.set_yticks([])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', default='gedig_simple')
    ap.add_argument('--maze-size', default='8x8')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--step-limit', type=int, default=300)
    ap.add_argument('--frontier-weight', type=float, default=0.2)
    ap.add_argument('--out-dir', default='viz_run')
    ap.add_argument('--gif', action='store_true')
    args = ap.parse_args()

    w,h = map(int, args.maze_size.lower().split('x'))
    ep, steps = run_episode(args.policy, w, h, args.seed, args.step_limit, save_steps=True, frontier_weight=args.frontier_weight)
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    maze = generate_maze(w,h,args.seed)

    path = []
    for s in steps:
        path.append(tuple(s.agent_pos))
    # ensure start
    if not path or path[0] != (0,0):
        path.insert(0,(0,0))

    frames = []
    for i, pos in enumerate(path):
        fig, ax = plt.subplots(figsize=(4,4))
        draw_maze(ax, maze)
        xs = [p[0] for p in path[:i+1]]
        ys = [p[1] for p in path[:i+1]]
        ax.plot(xs, ys, color='orange', linewidth=2)
        ax.scatter([0],[0], c='green', marker='s', s=100, label='start')
        ax.scatter([w-1],[h-1], c='red', marker='*', s=140, label='goal')
        ax.scatter([pos[0]],[pos[1]], c='blue', s=60)
        ax.set_title(f"{args.policy} step {i} success={ep.success}")
        fig.tight_layout()
        frame_path = out / f"frame_{i:04d}.png"
        fig.savefig(frame_path)
        plt.close(fig)
        frames.append(frame_path)
    # Optional GIF
    if args.gif:
        try:
            import imageio
            images = [imageio.v2.imread(f) for f in frames]
            imageio.mimsave(out / 'run.gif', images, duration=0.08)
        except Exception as e:
            print('GIF generation failed (install imageio):', e)
    # Save episode JSON (metadata only)
    (out/'episode_meta.json').write_text(json.dumps({k:v for k,v in ep.__dict__.items() if k!='gedig_series'}, ensure_ascii=False, indent=2))
    print('Saved frames to', out)

if __name__ == '__main__':
    main()
