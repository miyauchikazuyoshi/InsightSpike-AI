"""Backtracking performance / event characterization on 25x25 maze.

Uses threshold estimates derived from phase2_standard25_fixed_maze_study:
 - gedig (wiring) threshold ≈ 0.30 (fallback if not recomputed)
 - backtrack threshold: slightly below lower decile of completion geDIG (fallback -0.20)

We re-run several seeds and collect:
  * backtrack_trigger count
  * length distribution of backtrack_path_plan
  * time-to-first-backtrack
  * goal success rate & steps
  * reverse_trace events correlation with shortcut_candidate

Outputs concise summary + optional raw event sample.
"""
from navigation.maze_navigator import MazeNavigator
from core.gedig_evaluator import GeDIGEvaluator
import numpy as np, random
from statistics import mean
from typing import Dict, Any, List, Tuple

SIZE = 25
START = (1,1)
GOAL = (23,23)
RUNS = 5
MAX_STEPS = 5000
# Fallback thresholds (can be overridden by CLI args later)
GEDIG_THRESHOLD = 0.30
BACKTRACK_THRESHOLD = -0.22  # slightly below -0.2 to encourage triggers in sparse growth

# Reuse maze generation from study (could import but keep self-contained to avoid circular import risk)
def generate_maze(seed: int) -> np.ndarray:
    random.seed(seed); np.random.seed(seed)
    maze = np.ones((SIZE,SIZE), dtype=int)
    def carve(x,y): maze[y,x] = 0
    carve(START[0], START[1])
    def neighbors(cx,cy):
        for dx,dy in [(2,0),(-2,0),(0,2),(0,-2)]:
            nx,ny = cx+dx, cy+dy
            if 1 <= nx < SIZE-1 and 1 <= ny < SIZE-1:
                yield nx,ny,dx,dy
    stack=[(START[0]|1, START[1]|1)]; visited={stack[0]}
    maze[stack[0][1], stack[0][0]] = 0
    while stack:
        x,y = stack[-1]
        nbs=[(nx,ny,dx,dy) for nx,ny,dx,dy in neighbors(x,y) if (nx,ny) not in visited]
        if not nbs: stack.pop(); continue
        nx,ny,dx,dy = random.choice(nbs)
        maze[y+dy//2, x+dx//2] = 0; maze[ny,nx] = 0
        visited.add((nx,ny)); stack.append((nx,ny))
    # loops
    loops_target=40; attempts=0; loops=0
    while loops < loops_target and attempts < 800:
        attempts += 1
        x = random.randint(2,SIZE-3); y = random.randint(2,SIZE-3)
        if maze[y,x]==1:
            open_cnt=sum(1 for dx,dy in [(0,1),(0,-1),(1,0),(-1,0)] if maze[y+dy,x+dx]==0)
            if open_cnt>=2:
                maze[y,x]=0; loops+=1
    maze[GOAL[1], GOAL[0]] = 0
    return maze

def summarize_backtracks(events: List[Dict[str,Any]]) -> Dict[str,Any]:
    triggers=[e for e in events if e['type']=='backtrack_trigger']
    plans=[e for e in events if e['type']=='backtrack_path_plan']
    steps=[e for e in events if e['type']=='backtrack_step']
    reverse=[e for e in events if e['type']=='reverse_trace']
    shortcuts=[e for e in events if e['type']=='shortcut_candidate']
    lens=[p['message']['length'] for p in plans if isinstance(p.get('message'), dict)]
    first_bt = triggers[0]['step'] if triggers else None
    return {
        'backtrack_triggers': len(triggers),
        'backtrack_plans': len(plans),
        'backtrack_steps': len(steps),
        'mean_plan_len': round(mean(lens),2) if lens else None,
        'max_plan_len': max(lens) if lens else None,
        'first_backtrack_step': first_bt,
        'reverse_traces': len(reverse),
        'shortcuts': len(shortcuts),
        'reverse_per_shortcut': (len(reverse)/len(shortcuts)) if shortcuts else None
    }

def run():
    run_stats=[]; goal_success=0; steps_to_goal=[]
    for seed in range(RUNS):
        maze = generate_maze(seed)
        nav = MazeNavigator(maze, START, GOAL, gedig_threshold=GEDIG_THRESHOLD, backtrack_threshold=BACKTRACK_THRESHOLD, wiring_strategy='simple')
        goal = nav.run(MAX_STEPS)
        if goal: goal_success+=1; steps_to_goal.append(nav.step_count)
        summary = summarize_backtracks(nav.event_log)
        summary.update({'seed': seed, 'goal': goal, 'steps': nav.step_count})
        run_stats.append(summary)
    # aggregate
    agg: Dict[str,Any] = {}
    def collect(key):
        vals=[r[key] for r in run_stats if r.get(key) is not None]
        return (round(mean(vals),2) if vals else None, max(vals) if vals else None)
    agg['goal_success_rate'] = goal_success / RUNS
    agg['mean_steps_to_goal'] = round(mean(steps_to_goal),2) if steps_to_goal else None
    for k in ['backtrack_triggers','backtrack_plans','backtrack_steps','mean_plan_len','max_plan_len','first_backtrack_step','reverse_traces','shortcuts','reverse_per_shortcut']:
        agg[k] = collect(k)[0]
    print("=== Backtrack 25x25 Experiment ===")
    for r in run_stats:
        print(f"seed={r['seed']} goal={r['goal']} steps={r['steps']} triggers={r['backtrack_triggers']} plans={r['backtrack_plans']} avg_plan={r['mean_plan_len']}")
    print("-- Aggregate --")
    for k,v in agg.items():
        print(f"{k}: {v}")
    # sample events from last run
    last = run_stats[-1]
    print("-- Last run sample events (types) --")
    print([e['type'] for e in nav.event_log[:30]])

if __name__ == '__main__':
    run()
