"""25x25 Maze branch completion / loop (shortcut) geDIG exploratory study.

Focus:
 1. 分岐入口に入って誤った枝を全探索 → 分岐完了時の geDIG 分布とエピソード類似度マージン。
 2. 人工ループ(loop_test wiring) により密エッジ追加 → geDIG 負スパイク(短絡) 分布観測。
 3. multi-hop geDIG 減衰が負スパイク判定の安定化に有用かの初期検証。

Outputs: summary + tentative thresholds.
"""
from navigation.maze_navigator import MazeNavigator
from navigation.branch_detector import BranchDetector
from core.gedig_evaluator import GeDIGEvaluator
import numpy as np
import networkx as nx
from statistics import mean
from typing import List, Tuple, Dict
import random

SIZE = 25
START = (0,0)
GOAL = (24,24)
RUNS = 3  # exploratory
MAX_STEPS = 4000

def generate_perfect_maze(width: int, height: int, seed: int) -> np.ndarray:
    random.seed(seed)
    np.random.seed(seed)
    # grid of walls
    maze = np.ones((height, width), dtype=int)
    # carve passages using recursive backtracker on odd coordinates
    def neighbors(cx, cy):
        for dx, dy in [(2,0),(-2,0),(0,2),(0,-2)]:
            nx_, ny_ = cx+dx, cy+dy
            if 0 < nx_ < width and 0 < ny_ < height:
                yield nx_, ny_, dx, dy
    # start at (1,1)
    stack = [(1,1)]
    maze[1,1] = 0
    visited = {(1,1)}
    while stack:
        x,y = stack[-1]
        nbrs = [(nx_,ny_,dx,dy) for nx_,ny_,dx,dy in neighbors(x,y) if (nx_,ny_) not in visited]
        if not nbrs:
            stack.pop()
            continue
        nx_,ny_,dx,dy = random.choice(nbrs)
        maze[y+dy//2, x+dx//2] = 0
        maze[ny_, nx_] = 0
        visited.add((nx_,ny_))
        stack.append((nx_,ny_))
    # ensure start/goal open
    maze[START[1], START[0]] = 0
    maze[GOAL[1], GOAL[0]] = 0
    # introduce a few random loops (remove walls) to create branch/shortcut opportunities
    loop_pairs = 0
    attempts = 0
    while loop_pairs < 40 and attempts < 400:
        attempts += 1
        x = random.randrange(1, width-1)
        y = random.randrange(1, height-1)
        if maze[y, x] == 1:
            # if removing does not disconnect start-goal path viability drastically, carve
            maze[y, x] = 0
            loop_pairs += 1
    return maze

def pct(values: List[float], p: float):
    if not values: return None
    arr = sorted(values)
    k = int((p/100) * (len(arr)-1))
    return arr[k]

def run_study():
    branch_completion_gedigs = []
    branch_explore_gedigs = []
    shortcut_gedigs = []
    completion_margins = []
    completion_unvisited_ratio = []
    branch_steps = []
    multi_hop_shortcut = []
    multi_hop_positive = []

    evaluator = GeDIGEvaluator()

    for seed in range(RUNS):
        maze = generate_perfect_maze(SIZE, SIZE, seed)
        # alternate wiring strategies: even seed simple, odd seed loop_test
        strategy = 'loop_test' if seed % 2 == 1 else 'simple'
        nav = MazeNavigator(maze, START, GOAL, temperature=0.1, wiring_strategy=strategy, gedig_threshold=0.3)
        nav.run(MAX_STEPS)

        # Map step->gedig value
        gedig_map = {i+1: v for i, v in enumerate(nav.gedig_history)}

        # Event iteration
        for ev in nav.event_log:
            if ev['type'] == 'branch_completion':
                step = ev['step']
                gval = gedig_map.get(step, 0.0)
                branch_completion_gedigs.append(gval)
                # distance margin / unvisited ratio from decision history at or before step
                dh = [d for d in nav.decision_history if d['step'] <= step]
                if dh:
                    dec = dh[-1]
                    opts = dec['options']['options'] if 'options' in dec['options'] else dec['options']
                    distances = [info.get('distance') for info in opts.values() if 'distance' in info]
                    if len(distances) >= 2:
                        ds = sorted(distances)
                        completion_margins.append(ds[1]-ds[0])
                    unv = [1 for info in opts.values() if info.get('visit_count') == 0 and not info.get('is_wall')]
                    total = sum(1 for info in opts.values() if not info.get('is_wall'))
                    if total>0:
                        completion_unvisited_ratio.append(len(unv)/total)
                # steps in branch (from detector)
                if nav.branch_detector.completed_branches:
                    br = nav.branch_detector.completed_branches[-1]
                    branch_steps.append(br.get('steps_in_branch'))
            elif ev['type'] == 'branch_analysis':
                analysis = ev['message']
                gval = analysis.get('gedig_1hop', 0.0)
                if analysis.get('is_shortcut'):
                    shortcut_gedigs.append(gval)
                    # multi-hop at shortcut
                    mh = evaluator.calculate_multihop(nav.graph_manager.graph_history[-2], nav.graph_manager.graph_history[-1])
                    multi_hop_shortcut.append(list(mh.values())[:3])
                else:
                    branch_explore_gedigs.append(gval)
                    mh = evaluator.calculate_multihop(nav.graph_manager.graph_history[-2], nav.graph_manager.graph_history[-1])
                    multi_hop_positive.append(list(mh.values())[:3])

    # Summaries
    summary = {
        'runs': RUNS,
        'branch_completion_count': len(branch_completion_gedigs),
        'branch_explore_samples': len(branch_explore_gedigs),
        'shortcut_samples': len(shortcut_gedigs),
        'mean_completion_gedig': round(mean(branch_completion_gedigs),4) if branch_completion_gedigs else None,
        'mean_explore_gedig': round(mean(branch_explore_gedigs),4) if branch_explore_gedigs else None,
        'mean_shortcut_gedig': round(mean(shortcut_gedigs),4) if shortcut_gedigs else None,
        'p10_completion_gedig': pct(branch_completion_gedigs,10),
        'p90_completion_gedig': pct(branch_completion_gedigs,90),
        'p10_explore_gedig': pct(branch_explore_gedigs,10),
        'p90_explore_gedig': pct(branch_explore_gedigs,90),
        'shortcut_min': min(shortcut_gedigs) if shortcut_gedigs else None,
        'shortcut_max': max(shortcut_gedigs) if shortcut_gedigs else None,
        'mean_completion_margin': round(mean(completion_margins),4) if completion_margins else None,
        'mean_unvisited_ratio_at_completion': round(mean(completion_unvisited_ratio),4) if completion_unvisited_ratio else None,
        'mean_branch_steps': round(mean(branch_steps),2) if branch_steps else None,
    }

    # Tentative thresholds
    if branch_explore_gedigs:
        wiring_threshold = pct(branch_explore_gedigs, 60)
    else:
        wiring_threshold = 0.3
    if branch_completion_gedigs:
        backtrack_threshold = (pct(branch_completion_gedigs, 10) or 0) - 0.01
    else:
        backtrack_threshold = -0.2
    if shortcut_gedigs:
        negative_band = (min(shortcut_gedigs), max(shortcut_gedigs))
        shortcut_trigger = max(shortcut_gedigs)  # boundary near 0
    else:
        negative_band = None
        shortcut_trigger = -0.05

    # Multi-hop decay comparison
    def avg_decay(samples: List[List[float]]):
        if not samples: return None
        a1 = mean(s[0] for s in samples if len(s)>=1)
        a2 = mean(s[1] for s in samples if len(s)>=2)
        a3 = mean(s[2] for s in samples if len(s)>=3)
        return [round(a1,4), round(a2,4), round(a3,4)]

    decay_shortcut = avg_decay(multi_hop_shortcut)
    decay_positive = avg_decay(multi_hop_positive)

    print("=== 25x25 Branch / Loop geDIG Study ===")
    for k,v in summary.items():
        print(f"{k}: {v}")
    print("\nTentative Thresholds:")
    print(f" wiring_threshold ≈ {round(wiring_threshold,3)}")
    print(f" backtrack_threshold ≈ {round(backtrack_threshold,3)}")
    print(f" shortcut_negative_band ≈ {negative_band}")
    print(f" shortcut_trigger ≈ {round(shortcut_trigger,3)}")
    if decay_positive:
        print(f" multi-hop positive avg (1,2,3hop): {decay_positive}")
    if decay_shortcut:
        print(f" multi-hop shortcut avg (1,2,3hop): {decay_shortcut}")
        if decay_shortcut and decay_positive:
            # Heuristic: if |shortcut 1hop| / |positive 1hop| < 0.5 and further decays flatten → consider hop averaging
            if abs(decay_shortcut[0]) < abs(decay_positive[0])*0.6:
                print(" multi-hop: shortcut spikes comparatively damped; averaging 1-2 hop may stabilize noise.")

if __name__ == "__main__":
    run_study()
