#!/usr/bin/env python3
"""Baseline (oracle-free) explorers: Random Walk and DFS, for comparison.

Outputs metrics similar to Simple Mode report without using MazeNavigator:
  - steps, unique_positions, loop_redundancy, loop_erased_length, success

CLI examples:
  python baseline_explorers.py --variant ultra --algo random --seeds 11 22 --max_steps 600
  python baseline_explorers.py --variant complex --algo dfs --seeds 101 202 303 --max_steps 1500

Writes results to results/maze_report/baseline_<variant>_<timestamp>/
  - results.json (per-seed rows)
  - summary.md  (human-readable)
"""
from __future__ import annotations
import os, sys, argparse, json, datetime, random
import numpy as np
from collections import deque

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from maze_layouts import (
    create_complex_maze, create_ultra_maze, create_large_maze,
    COMPLEX_DEFAULT_START, COMPLEX_DEFAULT_GOAL,
    ULTRA_DEFAULT_START, ULTRA_DEFAULT_GOAL,
    LARGE_DEFAULT_START, LARGE_DEFAULT_GOAL
)
from results_paths import RESULTS_BASE
from generate_complex_maze_report import loop_erased_length

Coord = tuple[int,int]

# --- Helper utilities ---

def neighbors(maze: np.ndarray, p: Coord):
    x,y = p
    for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
        nx, ny = x+dx, y+dy
        if 0 <= nx < maze.shape[1] and 0 <= ny < maze.shape[0] and maze[ny,nx]==0:
            yield (nx,ny)

# --- Random Walk ---

def run_random(maze: np.ndarray, start: Coord, goal: Coord, max_steps: int, rng: random.Random):
    path=[start]
    current=start
    visited=set([start])
    for step in range(1, max_steps+1):
        nbrs=list(neighbors(maze,current))
        if not nbrs:
            break  # stuck (shouldn't happen often)
        # prefer unvisited with some probability
        unvisited=[n for n in nbrs if n not in visited]
        if unvisited and rng.random()<0.8:
            nxt=rng.choice(unvisited)
        else:
            nxt=rng.choice(nbrs)
        path.append(nxt)
        visited.add(nxt)
        current=nxt
        if current==goal:
            break
    return path, current==goal

# --- DFS (iterative) ---

def run_dfs(maze: np.ndarray, start: Coord, goal: Coord, max_steps: int, rng: random.Random):
    stack=[start]
    parent={start: None}
    expanded_order=[]
    steps=0
    while stack and steps < max_steps:
        node=stack.pop()
        expanded_order.append(node)
        steps+=1
        if node==goal:
            break
        nbrs=list(neighbors(maze,node))
        rng.shuffle(nbrs)
        for n in nbrs:
            if n not in parent:
                parent[n]=node
                stack.append(n)
    # reconstruct path (if goal reached), else use expanded order as trace
    if expanded_order and expanded_order[-1]==goal:
        path=[]
        cur=goal
        while cur is not None:
            path.append(cur)
            cur=parent[cur]
        path.reverse()
        success=True
    else:
        path=expanded_order
        success=(expanded_order and expanded_order[-1]==goal)
    return path, success

# --- Metric packaging ---

def summarize(path: list[Coord], success: bool):
    le_len=loop_erased_length(path)
    redundancy=(len(path)/le_len) if le_len else None
    unique=len(set(path))
    return {
        'steps': len(path)-1,
        'path_length': len(path),
        'loop_erased_length': le_len,
        'loop_redundancy': redundancy,
        'unique_positions': unique,
        'success': success,
    }

# --- Main batch runner ---

def run_batch(variant: str, algo: str, seeds, max_steps: int):
    results=[]
    for s in seeds:
        rng=random.Random(s)
        if variant=='complex':
            maze=create_complex_maze(); start, goal = COMPLEX_DEFAULT_START, COMPLEX_DEFAULT_GOAL
        elif variant=='ultra':
            maze=create_ultra_maze(); start, goal = ULTRA_DEFAULT_START, ULTRA_DEFAULT_GOAL
        else:
            maze=create_large_maze(); start, goal = LARGE_DEFAULT_START, LARGE_DEFAULT_GOAL
        if algo=='random':
            path, success=run_random(maze,start,goal,max_steps,rng)
        else:
            path, success=run_dfs(maze,start,goal,max_steps,rng)
        metrics=summarize(path, success)
        row={'seed': s, 'algo': algo, **metrics}
        results.append(row)
        print(f"{algo} seed {s} steps={metrics['steps']} unique={metrics['unique_positions']} loop_red={metrics['loop_redundancy']:.2f} success={metrics['success']}")
    return results


def aggregate(rows):
    import statistics
    agg={}
    if not rows: return agg
    for k in ['steps','loop_redundancy','unique_positions']:
        vals=[r[k] for r in rows if r.get(k) is not None]
        if not vals: continue
        agg[k]= {
            'mean': float(statistics.fmean(vals)),
            'stdev': float(statistics.pstdev(vals)) if len(vals)>1 else 0.0
        }
    agg['success_rate']= sum(1 for r in rows if r['success'])/len(rows)
    return agg


def write_summary_md(path: str, variant: str, algo: str, params: dict, rows, agg):
    with open(path,'w') as f:
        f.write(f"# Baseline {algo} ({variant}) Summary\n\n")
        f.write('## Params\n')
        for k,v in params.items(): f.write(f'- **{k}**: {v}\n')
        f.write('\n## Per-seed\n')
        f.write('| seed | steps | unique | loop_red | success |\n')
        f.write('|------|-------|--------|----------|---------|\n')
        for r in rows:
            f.write(f"| {r['seed']} | {r['steps']} | {r['unique_positions']} | {r['loop_redundancy']:.2f} | {int(r['success'])} |\n")
        f.write('\n## Aggregate\n')
        f.write(f"Success rate: {agg.get('success_rate'):.2f}\n")
        for k in ['steps','loop_redundancy','unique_positions']:
            if k in agg:
                f.write(f"- {k}: {agg[k]['mean']:.3f} ± {agg[k]['stdev']:.3f}\n")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--variant', choices=['complex','ultra','large'], default='complex')
    ap.add_argument('--algo', choices=['random','dfs'], required=True)
    ap.add_argument('--seeds', type=int, nargs='+', default=[101,202,303])
    ap.add_argument('--max_steps', type=int, default=1500)
    args=ap.parse_args()

    rows=run_batch(args.variant, args.algo, args.seeds, args.max_steps)
    agg=aggregate(rows)
    ts=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir=os.path.join(RESULTS_BASE, f'baseline_{args.algo}_{args.variant}_{ts}')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir,'results.json'),'w') as f: json.dump({'params': vars(args), 'rows': rows, 'aggregate': agg}, f, indent=2)
    write_summary_md(os.path.join(out_dir,'summary.md'), args.variant, args.algo, vars(args), rows, agg)
    print(f'Baseline results written to {out_dir}')

if __name__=='__main__':
    main()
