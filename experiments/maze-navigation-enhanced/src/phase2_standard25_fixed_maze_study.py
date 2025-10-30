"""Fixed 25x25 standard-like maze geDIG / branch / loop exploratory study.

This approximates the provided "standard 25x25" maze: start (1,1), goal (23,23).
We procedurally generate a maze with controlled loop injection to resemble
branch richness + moderate dead-end count, then run both normal and loop_test
wiring strategies to capture:
  - geDIG distributions (exploration / completion / shortcut)
  - branch completion margins (episode distance margin, unvisited ratio)
  - multi-hop geDIG decay comparison

If you want to exactly replicate an external layout, replace FIXED_MAZE with
an ASCII import (implement TODO section) keeping coordinates stable.
"""
from navigation.maze_navigator import MazeNavigator
from core.gedig_evaluator import GeDIGEvaluator
import numpy as np
import random
from statistics import mean
from typing import List, Tuple

SIZE = 25
START = (1,1)
GOAL = (23,23)
RUNS = 4
MAX_STEPS = 5000

def generate_base(seed: int) -> np.ndarray:
    random.seed(seed)
    np.random.seed(seed)
    maze = np.ones((SIZE, SIZE), dtype=int)
    def carve(x,y):
        maze[y,x] = 0
    carve(START[0], START[1])
    # Depth-first carve on odd coordinates
    def neighbors(cx, cy):
        for dx,dy in [(2,0),(-2,0),(0,2),(0,-2)]:
            nx, ny = cx+dx, cy+dy
            if 1 <= nx < SIZE-1 and 1 <= ny < SIZE-1:
                yield nx, ny, dx, dy
    stack=[(START[0]|1, START[1]|1)]
    visited={stack[0]}
    maze[stack[0][1], stack[0][0]] = 0
    while stack:
        x,y = stack[-1]
        nbs=[(nx,ny,dx,dy) for nx,ny,dx,dy in neighbors(x,y) if (nx,ny) not in visited]
        if not nbs:
            stack.pop(); continue
        nx,ny,dx,dy = random.choice(nbs)
        maze[y+dy//2, x+dx//2] = 0
        maze[ny, nx] = 0
        visited.add((nx,ny))
        stack.append((nx,ny))
    # Add controlled loops by removing random interior walls
    loops_target = 45
    attempts=0; loops=0
    while loops < loops_target and attempts < 800:
        attempts += 1
        x = random.randint(2,SIZE-3)
        y = random.randint(2,SIZE-3)
        if maze[y,x] == 1:
            # ensure at least two open neighbors to become loop
            open_cnt=0
            for dx,dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                nx,ny = x+dx,y+dy
                if maze[ny,nx]==0:
                    open_cnt+=1
            if open_cnt>=2:
                maze[y,x]=0
                loops+=1
    maze[GOAL[1], GOAL[0]] = 0
    return maze

def count_dead_ends(maze: np.ndarray) -> List[Tuple[int,int]]:
    ends=[]
    h,w = maze.shape
    for y in range(h):
        for x in range(w):
            if maze[y,x]==0:
                open_n=0
                for dx,dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                    nx,ny = x+dx,y+dy
                    if 0<=nx<w and 0<=ny<h and maze[ny,nx]==0:
                        open_n+=1
                if open_n==1:
                    ends.append((x,y))
    return ends

def pct(values, p):
    if not values: return None
    arr = sorted(values);
    k = int((p/100)*(len(arr)-1))
    return arr[k]

def run():
    branch_completion_gedigs=[]
    branch_explore_gedigs=[]
    shortcut_gedigs=[]
    completion_margins=[]
    completion_unvisited_ratio=[]
    multi_hop_shortcut=[]
    multi_hop_positive=[]
    branch_steps=[]
    evaluator = GeDIGEvaluator()

    for seed in range(RUNS):
        maze = generate_base(seed)
        dead_ends = count_dead_ends(maze)
        strategy = 'loop_test' if seed % 2 == 1 else 'simple'
        nav = MazeNavigator(maze, START, GOAL, temperature=0.1, wiring_strategy=strategy, gedig_threshold=0.3)
        nav.run(MAX_STEPS)
        gedig_map = {i+1:v for i,v in enumerate(nav.gedig_history)}
        # events
        for ev in nav.event_log:
            if ev['type']=='branch_completion':
                step=ev['step']; g = gedig_map.get(step,0.0)
                branch_completion_gedigs.append(g)
                # decision margin
                dh = [d for d in nav.decision_history if d['step']<=step]
                if dh:
                    dec=dh[-1]
                    opts = dec['options']['options'] if 'options' in dec['options'] else dec['options']
                    distances=[info.get('distance') for info in opts.values() if 'distance' in info]
                    if len(distances)>=2:
                        ds=sorted(distances); completion_margins.append(ds[1]-ds[0])
                    unv=[1 for info in opts.values() if info.get('visit_count')==0 and not info.get('is_wall')]
                    total=sum(1 for info in opts.values() if not info.get('is_wall'))
                    if total>0:
                        completion_unvisited_ratio.append(len(unv)/total)
                if nav.branch_detector.completed_branches:
                    branch_steps.append(nav.branch_detector.completed_branches[-1].get('steps_in_branch'))
            elif ev['type']=='branch_analysis':
                analysis=ev['message']; g=analysis.get('gedig_1hop',0.0)
                mh = evaluator.calculate_multihop(nav.graph_manager.graph_history[-2], nav.graph_manager.graph_history[-1])
                if analysis.get('is_shortcut'):
                    shortcut_gedigs.append(g)
                    multi_hop_shortcut.append(list(mh.values())[:3])
                else:
                    branch_explore_gedigs.append(g)
                    multi_hop_positive.append(list(mh.values())[:3])

    # summaries
    from math import fabs
    summary={
        'runs': RUNS,
        'branch_completion': len(branch_completion_gedigs),
        'branch_explore': len(branch_explore_gedigs),
        'shortcut_events': len(shortcut_gedigs),
        'mean_completion_gedig': round(mean(branch_completion_gedigs),4) if branch_completion_gedigs else None,
        'mean_explore_gedig': round(mean(branch_explore_gedigs),4) if branch_explore_gedigs else None,
        'mean_shortcut_gedig': round(mean(shortcut_gedigs),4) if shortcut_gedigs else None,
        'p10_completion': pct(branch_completion_gedigs,10),
        'p90_completion': pct(branch_completion_gedigs,90),
        'p10_explore': pct(branch_explore_gedigs,10),
        'p90_explore': pct(branch_explore_gedigs,90),
        'shortcut_band': (min(shortcut_gedigs), max(shortcut_gedigs)) if shortcut_gedigs else None,
        'mean_completion_margin': round(mean(completion_margins),4) if completion_margins else None,
        'mean_unvisited_ratio_completion': round(mean(completion_unvisited_ratio),4) if completion_unvisited_ratio else None,
        'mean_branch_steps': round(mean(branch_steps),2) if branch_steps else None,
    }

    # thresholds tentative
    wiring_threshold = pct(branch_explore_gedigs,60) if branch_explore_gedigs else 0.3
    backtrack_threshold = (pct(branch_completion_gedigs,10) or 0) - 0.01 if branch_completion_gedigs else -0.2
    if shortcut_gedigs:
        shortcut_trigger = max(shortcut_gedigs)
    else:
        shortcut_trigger = -0.05

    def avg_decay(samples: List[List[float]]):
        if not samples: return None
        a1=mean(s[0] for s in samples if len(s)>=1)
        a2=mean(s[1] for s in samples if len(s)>=2)
        a3=mean(s[2] for s in samples if len(s)>=3)
        return [round(a1,4), round(a2,4), round(a3,4)]

    decay_pos = avg_decay(multi_hop_positive)
    decay_short = avg_decay(multi_hop_shortcut)

    print("=== Standard25 Fixed Maze Study ===")
    for k,v in summary.items():
        print(f"{k}: {v}")
    print("\nTentative thresholds:")
    print(f" wiring_threshold ≈ {round(wiring_threshold,3)}")
    print(f" backtrack_threshold ≈ {round(backtrack_threshold,3)}")
    print(f" shortcut_trigger ≈ {round(shortcut_trigger,3)}")
    print(f" shortcut_band ≈ {summary['shortcut_band']}")
    if decay_pos:
        print(f" multi-hop positive avg (1,2,3): {decay_pos}")
    if decay_short:
        print(f" multi-hop shortcut avg (1,2,3): {decay_short}")
        if decay_short and decay_pos and abs(decay_short[0]) < abs(decay_pos[0])*0.6:
            print(" multi-hop smoothing: consider averaging 1-2 hop for stability.")

if __name__ == "__main__":
    run()
