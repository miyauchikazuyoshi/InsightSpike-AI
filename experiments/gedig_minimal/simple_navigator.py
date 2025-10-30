"""Minimal Layer3 (geDIG local gain) navigator implementation.

R1 scope: no norm search (Layer1), no frontier approx (Layer2), no adaptive (Layer4).

Provides:
- generate_maze(width, height, seed)
- run_episode(policy_name, width, height, seed, step_limit=500, save_steps=True)
- Policy implementations: random, gedig_simple, threshold_{1..3}
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional, Set
import random
import json
import math
import numpy as np

Direction = Tuple[int, int]
DIRECTIONS: List[Direction] = [(0,-1),(0,1),(-1,0),(1,0)]  # U,D,L,R

@dataclass
class CandidateFeature:
    pos: Tuple[int,int]
    passable: bool
    visited: bool
    vec: np.ndarray  # shape (8,)
    neighbor_unvisited: int          # 1-step 未訪問数
    neighbor_unvisited_2: int        # 2-step まで (1-step を除くユニーク) 未訪問数

@dataclass
class StepRecord:
    t: int
    agent_pos: Tuple[int,int]
    candidate_vecs: List[List[float]]
    gedig_values: List[float]
    chosen_index: int
    chosen_pos: Tuple[int,int]
    r2_contrib_1: List[int]
    r2_contrib_2: List[int]

@dataclass
class EpisodeResult:
    seed: int
    maze_size: Tuple[int,int]
    policy: str
    success: bool
    steps: int
    unique_cells: int
    path_efficiency: float
    exploration_ratio: float
    gedig_series: List[float]
    gedig_auc: float
    step_limit: int
    # Optional: meta extension placeholder
    meta: Optional[Dict[str, Any]] = None

# ---------- Maze Generation (recursive backtracker) ---------- #

def generate_maze(width: int, height: int, seed: int) -> np.ndarray:
    random.seed(seed)
    np.random.seed(seed)
    # Initialize all walls
    maze = np.ones((height, width), dtype=np.int8)

    def in_bounds(x: int, y: int) -> bool:
        return 0 <= x < width and 0 <= y < height

    # Start from (0,0)
    stack: List[Tuple[int,int]] = [(0,0)]
    maze[0,0] = 0

    while stack:
        x,y = stack[-1]
        # 2-cell jumps
        candidates = []
        for dx,dy in DIRECTIONS:
            nx, ny = x + 2*dx, y + 2*dy
            if in_bounds(nx, ny) and maze[ny, nx] == 1:
                candidates.append((nx, ny, dx, dy))
        if candidates:
            nx, ny, dx, dy = random.choice(candidates)
            maze[y + dy, x + dx] = 0  # carve intermediate
            maze[ny, nx] = 0
            stack.append((nx, ny))
        else:
            stack.pop()

    # Ensure start and goal are open
    maze[0,0] = 0
    maze[height-1, width-1] = 0

    # If even dimensions, open last row/col minimal path to guarantee reachability
    if width % 2 == 0:
        maze[:, width-1] = 0
    if height % 2 == 0:
        maze[height-1, :] = 0

    return maze

# ---------- Feature & geDIG computation ---------- #

def _build_candidate(agent: Tuple[int,int], c: Tuple[int,int], maze: np.ndarray, visited: Set[Tuple[int,int]], goal: Tuple[int,int]) -> CandidateFeature:
    h, w = maze.shape
    x, y = c
    passable = False
    if 0 <= x < w and 0 <= y < h and maze[y, x] == 0:
        passable = True
    visited_flag = (c in visited)

    # neighbor unvisited count (only if passable)
    neighbor_unvisited = 0
    first_ring: List[Tuple[int,int]] = []
    if passable:
        for dx,dy in DIRECTIONS:
            nx, ny = x+dx, y+dy
            if 0 <= nx < w and 0 <= ny < h and maze[ny, nx] == 0:
                first_ring.append((nx,ny))
                if (nx,ny) not in visited:
                    neighbor_unvisited += 1

    # second ring unique cells excluding first ring & candidate
    neighbor_unvisited_2 = 0
    if passable:
        seen2: Set[Tuple[int,int]] = set(first_ring)
        for (nx,ny) in first_ring:
            for dx2,dy2 in DIRECTIONS:
                sx, sy = nx+dx2, ny+dy2
                if (sx,sy) == c or (sx,sy) in seen2:
                    continue
                if 0 <= sx < w and 0 <= sy < h and maze[sy, sx] == 0:
                    seen2.add((sx,sy))
        # count unvisited nodes at distance 2 (exclude distance1 already counted)
        for cell in seen2:
            if cell == c:
                continue
            if cell not in visited and cell not in first_ring:
                neighbor_unvisited_2 += 1

    # Normalized coords
    ax, ay = agent
    gx, gy = goal
    denom_w = max(1, w-1)
    denom_h = max(1, h-1)
    vec = np.array([
        ax/denom_w,
        ay/denom_h,
        gx/denom_w,
        gy/denom_h,
        x/denom_w,
        y/denom_h,
        1.0 if passable else 0.0,
        1.0 if visited_flag else 0.0
    ], dtype=np.float32)

    return CandidateFeature(pos=c, passable=passable, visited=visited_flag, vec=vec, neighbor_unvisited=neighbor_unvisited, neighbor_unvisited_2=neighbor_unvisited_2)

def compute_gedig(cf: CandidateFeature, mode: str = 'R1', frontier_weight: float = 0.5) -> float:
    if not cf.passable:
        return -1.0
    base = 1 - int(cf.visited)
    if mode == 'R1':
        return float(base + cf.neighbor_unvisited)
    # R2: add weighted 2-step gain (excluding first ring) -> scale control via frontier_weight
    return float(base + cf.neighbor_unvisited + frontier_weight * cf.neighbor_unvisited_2)

# ---------- Policy selection ---------- #

def select_action(policy: str, candidates: List[CandidateFeature], rng: random.Random, frontier_weight: float) -> Tuple[int,List[float]]:
    mode = 'R2' if ('_r2' in policy) else 'R1'
    gedigs = [compute_gedig(c, mode=mode, frontier_weight=frontier_weight) for c in candidates]
    # adaptive policy name pattern: gedig_r2_adaptive_<fwmin>_<fwmax>
    if policy.startswith('gedig_r2_adaptive'):
        # treat same as r2 for selection (dynamic weight handled outside per step)
        pass
    if policy == 'random':
        viable = [i for i,c in enumerate(candidates) if c.passable]
        if not viable:
            return rng.randrange(len(candidates)), gedigs
        return rng.choice(viable), gedigs
    elif policy in ('gedig_simple','gedig_r2'):
        max_val = max(gedigs)
        if max_val <= -1:
            # fallback random among passable
            viable = [i for i,c in enumerate(candidates) if c.passable]
            if not viable:
                return rng.randrange(len(candidates)), gedigs
            return rng.choice(viable), gedigs
        best = [i for i,v in enumerate(gedigs) if v == max_val]
        return rng.choice(best), gedigs
    elif policy.startswith('threshold_'):
        try:
            t = int(policy.split('_',1)[1])
        except ValueError:
            t = 1
        eligible = [i for i,v in enumerate(gedigs) if v >= t and candidates[i].passable]
        if not eligible:
            viable = [i for i,c in enumerate(candidates) if c.passable]
            if not viable:
                return rng.randrange(len(candidates)), gedigs
            return rng.choice(viable), gedigs
        # choose max among eligible
        max_val = max(gedigs[i] for i in eligible)
        pool = [i for i in eligible if gedigs[i] == max_val]
        return rng.choice(pool), gedigs
    else:
        raise ValueError(f"Unknown policy: {policy}")

# ---------- Episode run ---------- #

def run_episode(policy: str, width: int, height: int, seed: int, step_limit: int = 500, save_steps: bool = True, frontier_weight: float = 0.5) -> Tuple[EpisodeResult, List[StepRecord]]:
    rng = random.Random(seed)
    maze = generate_maze(width, height, seed)
    start = (0,0)
    goal = (width-1, height-1)
    agent = start
    visited = {agent}
    step_records: List[StepRecord] = []
    gedig_series: List[int] = []

    manhattan_goal = (goal[0]-start[0]) + (goal[1]-start[1])

    # Series for frontier analysis
    chosen_n1: List[int] = []
    chosen_n2: List[int] = []

    # Adaptive frontier weight state (R3)
    adaptive = policy.startswith('gedig_r2_adaptive')
    if adaptive:
        # parse min/max from policy e.g. gedig_r2_adaptive_0.1_0.25
        try:
            parts = policy.split('_')
            fw_min = float(parts[-2])
            fw_max = float(parts[-1])
        except Exception:
            fw_min, fw_max = 0.1, frontier_weight
        # Start at fw_max (aggressive) then reduce on stagnation
        current_fw = fw_max
        cooldown = 0  # steps until raising allowed again
        zero_streak = 0
        # tuned thresholds (R3 final): faster reaction
        drop_threshold = 3      # consecutive n2=0 to trigger drop
        recover_threshold = 8   # consecutive n2>0 accumulation to raise
        raise_accum = 0
        drop_events = 0
        raise_events = 0
        fw_sum = 0.0

    for t in range(step_limit):
        if agent == goal:
            break
        candidates: List[CandidateFeature] = []
        r2_mode = ('_r2' in policy)
        if adaptive:
            r2_mode = True  # force R2 mode
        for dx,dy in DIRECTIONS:
            c = (agent[0]+dx, agent[1]+dy)
            candidates.append(_build_candidate(agent, c, maze, visited, goal))
        effective_fw = current_fw if adaptive else frontier_weight
        chosen_index, gedigs = select_action('gedig_r2' if r2_mode else policy, candidates, rng, effective_fw)
        chosen = candidates[chosen_index]
        gedig_series.append(float(gedigs[chosen_index]))
        # record frontier stats per chosen
        chosen_n1.append(chosen.neighbor_unvisited)
        chosen_n2.append(chosen.neighbor_unvisited_2)
        # adaptive update
        if adaptive:
            fw_sum += current_fw
            if chosen.neighbor_unvisited_2 == 0:
                zero_streak += 1
                raise_accum = 0
            else:
                raise_accum += 1
                zero_streak = 0
            # drop condition
            if zero_streak >= drop_threshold and current_fw > fw_min:
                current_fw = max(fw_min, current_fw * 0.5)
                cooldown = 6
                zero_streak = 0
                drop_events += 1
            # recovery: linear gentle increase
            if cooldown > 0:
                cooldown -= 1
            else:
                if raise_accum >= recover_threshold and current_fw < fw_max:
                    current_fw = min(fw_max, current_fw + 0.02)
                    raise_accum = 0
                    raise_events += 1
        # Move if passable
        if chosen.passable:
            agent = chosen.pos
            if agent not in visited:
                visited.add(agent)
        if save_steps:
            step_records.append(StepRecord(
                t=t,
                agent_pos=agent,
                candidate_vecs=[c.vec.tolist() for c in candidates],
                gedig_values=gedigs,
                chosen_index=chosen_index,
                chosen_pos=agent,
                r2_contrib_1=[c.neighbor_unvisited for c in candidates],
                r2_contrib_2=[c.neighbor_unvisited_2 for c in candidates],
            ))
        if agent == goal:
            break

    success = agent == goal
    steps_taken = len(gedig_series)
    if success and steps_taken > 0:
        path_eff = manhattan_goal / steps_taken if steps_taken > 0 else 0.0
    else:
        path_eff = 0.0
    unique_cells = len(visited)
    exploration_ratio = unique_cells / steps_taken if steps_taken > 0 else 0.0
    # AUC 正規化: R1 最大 ~5, R2 最大近似: base(1)+n1(4)+frontier_weight*(~8) ≈ 1+4+8*fw
    is_r2 = ('_r2' in policy)
    if is_r2:
        # adaptive の場合は実際の最大候補 (fw_max) を用いて正規化上限を近似
        effective_fw_for_max = fw_max if adaptive else frontier_weight
        assumed_max = 1 + 4 + effective_fw_for_max * 8
    else:
        assumed_max = 5
    if steps_taken > 0 and assumed_max > 0:
        gedig_auc = sum(gedig_series) / (steps_taken * assumed_max)
    else:
        gedig_auc = 0.0

    # Frontier hist & stagnation proxy metrics
    def _hist(xs: List[int], bins: int) -> List[int]:
        h = [0]*bins
        for v in xs:
            if v < 0:
                continue
            idx = v if v < bins else bins-1
            h[idx] += 1
        return h
    n1_hist = _hist(chosen_n1, 5)  # 0..4
    n2_hist = _hist(chosen_n2, 9)  # 0..8 (>=8 まとめ)

    # zero streak metrics for n2
    longest_zero = 0
    current = 0
    zero_segments: List[int] = []
    for v in chosen_n2:
        if v == 0:
            current += 1
        else:
            if current>0:
                zero_segments.append(current)
            longest_zero = max(longest_zero, current)
            current = 0
    if current>0:
        zero_segments.append(current)
        longest_zero = max(longest_zero, current)
    mean_zero_streak = (sum(zero_segments)/len(zero_segments)) if zero_segments else 0.0
    zero_fraction = (sum(1 for v in chosen_n2 if v==0)/len(chosen_n2)) if chosen_n2 else 0.0

    ep = EpisodeResult(
        seed=seed,
        maze_size=(width, height),
        policy=policy,
        success=success,
        steps=steps_taken,
        unique_cells=unique_cells,
        path_efficiency=path_eff,
        exploration_ratio=exploration_ratio,
        gedig_series=gedig_series,
        gedig_auc=gedig_auc,
        step_limit=step_limit,
        meta={
            "gedig_version": "R2_frontier" if is_r2 else "R1_local",
            "frontier_weight": frontier_weight,
            "gedig_assumed_max": assumed_max,
            "n1_hist": n1_hist,
            "n2_hist": n2_hist,
            "n2_longest_zero_streak": longest_zero,
            "n2_mean_zero_streak": mean_zero_streak,
            "n2_zero_fraction": zero_fraction,
            **({
                "adaptive": True,
                "adaptive_fw_min": fw_min,
                "adaptive_fw_max": fw_max,
                "adaptive_fw_final": current_fw,
                "adaptive_fw_avg": (fw_sum/steps_taken) if steps_taken>0 else current_fw,
                "adaptive_drop_events": drop_events,
                "adaptive_raise_events": raise_events,
                "adaptive_params": {
                    "drop_threshold": drop_threshold,
                    "recover_threshold": recover_threshold,
                    "linear_raise_step": 0.02
                }
            } if adaptive else {})
        }
    )
    return ep, step_records

# Convenience: simple CLI episode test
if __name__ == "__main__":
    import argparse, pathlib
    p = argparse.ArgumentParser()
    p.add_argument('--policy', default='gedig_simple')
    p.add_argument('--maze-size', default='8x8')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--step-limit', type=int, default=500)
    p.add_argument('--no-save-steps', action='store_true')
    p.add_argument('--out', default='episode.json')
    p.add_argument('--frontier-weight', type=float, default=0.5)
    args = p.parse_args()
    w,h = map(int, args.maze_size.lower().split('x'))
    ep, steps = run_episode(args.policy, w, h, args.seed, args.step_limit, save_steps=not args.no_save_steps, frontier_weight=args.frontier_weight)
    out = {
        **ep.__dict__,
        **({"steps_detail":[s.__dict__ for s in steps]} if not args.no_save_steps else {})
    }
    pathlib.Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"Saved episode to {args.out}")
