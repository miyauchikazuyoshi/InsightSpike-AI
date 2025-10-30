"""Backtracking sensitivity vs maze complexity on 25x25.

We sweep several complexity parameters (loop density + extra openings) and relaxed
backtrack thresholds (>0) so that zero / low geDIG plateaus can trigger backtracking.

Metrics collected per (complexity, threshold, seed):
  - goal, steps
  - backtrack_triggers / plans / steps
  - first_backtrack_step
  - mean plan length
  - dead_end_count, open_ratio, loop_openings (approx)

Complexity axes:
  loops_target: controls number of interior wall removals forming loops
  extra_openings: random single-wall removals (less constrained) to raise branching

Run summary aggregates across seeds, printing a compact table.
"""
from navigation.maze_navigator import MazeNavigator
import numpy as np, random
from statistics import mean
from typing import List, Dict, Any, Tuple

SIZE=25
START=(1,1)
GOAL=(23,23)
SEEDS=5
MAX_STEPS=4000
LOOPS_LEVELS=[20,40,70]
EXTRA_OPENINGS=[0,25,60]
BACKTRACK_THRESHOLDS=[0.05,0.02]

# -------------------------------------------------------
# Maze utilities
# -------------------------------------------------------

def generate_maze(seed:int, loops_target:int, extra_open:int)->np.ndarray:
    random.seed(seed); np.random.seed(seed)
    maze = np.ones((SIZE,SIZE),dtype=int)
    def carve(x,y): maze[y,x]=0
    carve(START[0], START[1])
    # DFS carve on odd grid
    def neighbors(cx,cy):
        for dx,dy in [(2,0),(-2,0),(0,2),(0,-2)]:
            nx,ny = cx+dx, cy+dy
            if 1 <= nx < SIZE-1 and 1 <= ny < SIZE-1:
                yield nx,ny,dx,dy
    stack=[(START[0]|1, START[1]|1)]; visited={stack[0]}
    maze[stack[0][1], stack[0][0]]=0
    while stack:
        x,y = stack[-1]
        nbs=[(nx,ny,dx,dy) for nx,ny,dx,dy in neighbors(x,y) if (nx,ny) not in visited]
        if not nbs: stack.pop(); continue
        nx,ny,dx,dy = random.choice(nbs)
        maze[y+dy//2, x+dx//2]=0; maze[ny,nx]=0
        visited.add((nx,ny)); stack.append((nx,ny))
    # Structured loops (need >=2 open neighbors)
    attempts=0; loops=0
    while loops < loops_target and attempts < loops_target*40:
        attempts+=1
        x=random.randint(2,SIZE-3); y=random.randint(2,SIZE-3)
        if maze[y,x]==1:
            open_cnt=sum(1 for dx,dy in [(0,1),(0,-1),(1,0),(-1,0)] if maze[y+dy,x+dx]==0)
            if open_cnt>=2:
                maze[y,x]=0; loops+=1
    # Extra unconstrained openings (branch inflation)
    eo=0; attempts=0
    while eo < extra_open and attempts < extra_open*50:
        attempts+=1
        x=random.randint(1,SIZE-2); y=random.randint(1,SIZE-2)
        if maze[y,x]==1:
            maze[y,x]=0; eo+=1
    maze[GOAL[1], GOAL[0]]=0
    return maze

def dead_end_count(maze:np.ndarray)->int:
    h,w=maze.shape; c=0
    for y in range(h):
        for x in range(w):
            if maze[y,x]==0:
                open_n=sum(1 for dx,dy in [(0,1),(0,-1),(1,0),(-1,0)] if 0<=x+dx<w and 0<=y+dy<h and maze[y+dy,x+dx]==0)
                if open_n==1: c+=1
    return c

def open_ratio(maze:np.ndarray)->float:
    return float(np.count_nonzero(maze==0))/maze.size

# -------------------------------------------------------
# Backtrack summarization
# -------------------------------------------------------

def summarize_events(events:List[Dict[str,Any]])->Dict[str,Any]:
    t=[e for e in events if e['type']=='backtrack_trigger']
    p=[e for e in events if e['type']=='backtrack_path_plan']
    s=[e for e in events if e['type']=='backtrack_step']
    lens=[e['message']['length'] for e in p if isinstance(e.get('message'),dict)]
    return {
        'triggers':len(t),
        'plans':len(p),
        'steps':len(s),
        'mean_plan_len': round(mean(lens),2) if lens else None,
        'first_trigger_step': t[0]['step'] if t else None
    }

# -------------------------------------------------------
# Experiment
# -------------------------------------------------------

def run():
    rows=[]
    for loops in LOOPS_LEVELS:
        for extra in EXTRA_OPENINGS:
            for bt_th in BACKTRACK_THRESHOLDS:
                trig_total=0
                for seed in range(SEEDS):
                    maze=generate_maze(seed, loops, extra)
                    metrics={'seed':seed,'loops':loops,'extra':extra}
                    metrics['dead_ends']=dead_end_count(maze)
                    metrics['open_ratio']=round(open_ratio(maze),3)
                    nav=MazeNavigator(maze, START, GOAL, gedig_threshold=0.30, backtrack_threshold=bt_th, wiring_strategy='simple')
                    goal=nav.run(MAX_STEPS)
                    evs=summarize_events(nav.event_log)
                    metrics.update(evs)
                    metrics['goal']=goal; metrics['steps']=nav.step_count
                    metrics['bt_threshold']=bt_th
                    rows.append(metrics)
                    trig_total+=evs['triggers']
                # quick aggregate line per combination
                subset=[r for r in rows if r['loops']==loops and r['extra']==extra and r['bt_threshold']==bt_th]
                if subset:
                    avg_tr=mean(r['triggers'] for r in subset)
                    avg_first=mean([r['first_trigger_step'] for r in subset if r['first_trigger_step'] is not None]) if any(r['first_trigger_step'] for r in subset) else None
                    print(f"[combo] loops={loops:2d} extra={extra:2d} bt_th={bt_th:0.3f} avg_triggers={avg_tr:.2f} avg_first={avg_first} avg_dead_ends={mean(r['dead_ends'] for r in subset):.1f} open_ratio={mean(r['open_ratio'] for r in subset):.3f}")
    # Final compact table (first 50 rows)
    print("\nSample detailed rows (first 40):")
    for r in rows[:40]:
        print(r)

if __name__=="__main__":
    run()
