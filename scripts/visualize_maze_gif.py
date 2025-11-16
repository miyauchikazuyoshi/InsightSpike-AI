#!/usr/bin/env python3
"""
Generate an animated GIF of a maze run from a step log.

Inputs
- steps: JSON list where each element is a step record containing at least
  { "position": [row, col], ... }
- layout (optional): JSON with key "layout" holding a 2D int grid (0=open, 1=wall, 2=start, 3=goal)
- If layout is omitted, grid size is inferred from the first step using ring_Rr/ring_Rc or from the
  maximum position seen.

Example
  python scripts/visualize_maze_gif.py \
    --steps experiments/maze-query-hub-prototype/results/batch_25x25/paper25_25x25_s500_seed0_steps.json \
    --layout docs/paper/data/maze_25x25.json \
    --out docs/images/maze_demo.gif --cell 16 --fps 10

Notes
- Requires Pillow (PIL). This script is for documentation/social assets and is not used in CI.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw

# Ensure we can import project modules when called from repo root
try:
    import sys
    _PROJ_ROOT = Path(__file__).resolve().parents[1]
    _SRC = _PROJ_ROOT / "src"
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))
except Exception:
    pass


Color = Tuple[int, int, int]


def load_steps(path: Path) -> List[dict]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError("steps JSON must be a list of step records")
    return data


def load_layout(path: Optional[Path]) -> Optional[List[List[int]]]:
    if not path:
        return None
    obj = json.loads(path.read_text())
    if isinstance(obj, dict) and "layout" in obj:
        return obj["layout"]
    if isinstance(obj, list):
        return obj
    raise ValueError("layout JSON must be a list[list[int]] or { 'layout': [...] }")


def infer_grid_size(steps: List[dict], layout: Optional[List[List[int]]]) -> Tuple[int, int]:
    if layout is not None:
        rows = len(layout)
        cols = len(layout[0]) if rows else 0
        return rows, cols
    # Try ring_Rr/Rc in first step
    if steps:
        r = steps[0].get("ring_Rr")
        c = steps[0].get("ring_Rc")
        if isinstance(r, int) and isinstance(c, int) and r > 0 and c > 0:
            return r, c
    # Fallback to max seen position
    max_r = 0
    max_c = 0
    for rec in steps:
        pos = rec.get("position")
        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
            max_r = max(max_r, int(pos[0]))
            max_c = max(max_c, int(pos[1]))
    # Positions are 1-based in many logs; add +2 margin to be safe
    return max(max_r + 2, 2), max(max_c + 2, 2)


def render_background(
    rows: int,
    cols: int,
    cell: int,
    margin: int,
    layout: Optional[List[List[int]]],
    colors: dict,
) -> Image.Image:
    width = cols * cell + 2 * margin
    height = rows * cell + 2 * margin
    img = Image.new("RGB", (width, height), color=colors["bg"])
    draw = ImageDraw.Draw(img)
    wall_color: Color = colors["wall"]
    open_color: Color = colors["open"]
    start_color: Color = colors["start"]
    goal_color: Color = colors["goal"]

    for r in range(rows):
        for c in range(cols):
            x0 = margin + c * cell
            y0 = margin + r * cell
            x1 = x0 + cell
            y1 = y0 + cell
            val = 0
            if layout and r < len(layout) and c < len(layout[r]):
                val = int(layout[r][c])
            if val == 1:
                draw.rectangle([x0, y0, x1, y1], fill=wall_color)
            elif val == 2:
                draw.rectangle([x0, y0, x1, y1], fill=start_color)
            elif val == 3:
                draw.rectangle([x0, y0, x1, y1], fill=goal_color)
            else:
                draw.rectangle([x0, y0, x1, y1], fill=open_color)
    return img


def draw_agent(draw: ImageDraw.ImageDraw, pos: Tuple[int, int], cell: int, margin: int, color: Color):
    r, c = int(pos[0]), int(pos[1])
    x0 = margin + c * cell
    y0 = margin + r * cell
    pad = max(2, cell // 6)
    draw.ellipse([x0 + pad, y0 + pad, x0 + cell - pad, y0 + cell - pad], fill=color)


def draw_trail(draw: ImageDraw.ImageDraw, trail: Iterable[Tuple[int, int]], cell: int, margin: int, color: Color):
    pad = max(3, cell // 5)
    for r, c in trail:
        x0 = margin + c * cell
        y0 = margin + r * cell
        draw.rectangle([x0 + pad, y0 + pad, x0 + cell - pad, y0 + cell - pad], fill=color)


def build_frames(
    steps: List[dict],
    bg: Image.Image,
    cell: int,
    margin: int,
    colors: dict,
    stride: int,
    max_frames: Optional[int] = None,
) -> List[Image.Image]:
    frames: List[Image.Image] = []
    trail: List[Tuple[int, int]] = []
    for idx, rec in enumerate(steps):
        if stride > 1 and (idx % stride != 0):
            continue
        pos = rec.get("position")
        if not (isinstance(pos, (list, tuple)) and len(pos) >= 2):
            continue
        r, c = int(pos[0]), int(pos[1])
        trail.append((r, c))
        frame = bg.copy()
        draw = ImageDraw.Draw(frame)
        draw_trail(draw, trail[:-1], cell, margin, colors["trail"])  # trail before current
        draw_agent(draw, (r, c), cell, margin, colors["agent"])      # current position
        frames.append(frame)
        if max_frames is not None and len(frames) >= max_frames:
            break
    return frames


def save_gif(frames: List[Image.Image], out_path: Path, fps: int):
    if not frames:
        raise ValueError("no frames to save")
    duration = int(1000 / max(1, fps))  # ms per frame
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        str(out_path),
        save_all=True,
        append_images=frames[1:],
        optimize=False,
        duration=duration,
        loop=0,
        disposal=2,
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Visualize maze step log as animated GIF")
    ap.add_argument("--steps", type=Path, required=True, help="Path to *_steps.json")
    ap.add_argument("--layout", type=Path, default=None, help="Path to layout JSON (with 'layout')")
    ap.add_argument("--summary", type=Path, default=None, help="Path to *_summary.json (to reconstruct layout if needed)")
    ap.add_argument("--out", type=Path, default=Path("docs/images/maze_demo.gif"))
    ap.add_argument("--cell", type=int, default=16, help="Cell size in pixels")
    ap.add_argument("--margin", type=int, default=8, help="Margin in pixels around the grid")
    ap.add_argument("--fps", type=int, default=10, help="Frames per second for the GIF")
    ap.add_argument("--stride", type=int, default=1, help="Use every Nth step to reduce frames")
    ap.add_argument("--max-frames", type=int, default=None, help="Limit number of frames (optional)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    steps = load_steps(args.steps)
    layout = load_layout(args.layout)
    rows, cols = infer_grid_size(steps, layout)

    # If layout missing, try to reconstruct from summary (DFS/Kruskal/Prim) using seed
    if layout is None and args.summary is not None and args.summary.exists():
        try:
            import json as _json
            from insightspike.environments.proper_maze_generator import ProperMazeGenerator as _PM
            cfg = _json.loads(args.summary.read_text()).get("config", {})
            size = int(cfg.get("maze_size", rows))
            mtype = str(cfg.get("maze_type", "dfs")).lower()
            seed = int(cfg.get("seed_start", 0))
            if mtype in ("dfs", "kruskal", "prim"):
                if mtype == "dfs":
                    grid = _PM.generate_dfs_maze((size, size), seed=seed)
                elif mtype == "kruskal":
                    grid = _PM.generate_kruskal_maze((size, size), seed=seed)
                else:
                    grid = _PM.generate_prim_maze((size, size), seed=seed)
                # Mark start/goal as in SimpleMaze
                grid = grid.copy()
                grid[1,1] = 2
                grid[size-2, size-2] = 3
                layout = grid.tolist()
                rows, cols = size, size
                print(f"[gif] reconstructed layout via summary (type={mtype}, seed={seed})")
        except Exception as e:
            print(f"[gif] layout reconstruction skipped: {e}")

    colors = {
        "bg": (18, 18, 18),
        "open": (245, 245, 245),
        "wall": (60, 60, 60),
        "start": (210, 235, 255),
        "goal": (230, 255, 210),
        "trail": (120, 170, 255),
        "agent": (255, 80, 80),
    }

    bg = render_background(rows, cols, args.cell, args.margin, layout, colors)
    frames = build_frames(steps, bg, args.cell, args.margin, colors, stride=max(1, args.stride), max_frames=args.max_frames)
    save_gif(frames, args.out, fps=max(1, args.fps))
    print(f"[gif] wrote {args.out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
