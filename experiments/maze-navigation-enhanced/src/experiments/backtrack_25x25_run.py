"""25x25 maze backtracking experiment using thresholds derived from standard study.

Uses:
  wiring_strategy='simple'
  gedig_threshold = 0.0 (from p60 explore ~0)
  backtrack_threshold = -0.01 (p10 completion - 0.01 fallback)
Runs multiple seeds; reports backtrack event counts and basic stats.
"""
from navigation.maze_navigator import MazeNavigator
import numpy as np, random

SIZE=25
START=(1,1)
GOAL=(23,23)
RUNS=3
MAX_STEPS=3000
GEDIG_THRESHOLD=0.0
BACKTRACK_THRESHOLD=-0.01

# Simple DFS style generator (same as in phase2_standard25_fixed_maze_study but lighter)

def generate(seed:int):
    random.seed(seed); np.random.seed(seed)
    maze = np.ones((SIZE,SIZE),dtype=int)
    def neighbors(cx,cy):
        for dx,dy in [(2,0),(-2,0),(0,2),(0,-2)]:
            nx,ny=cx+dx,cy+dy
            if 1<=nx<SIZE-1 and 1<=ny<SIZE-1:
                yield nx,ny,dx,dy
    stack=[(START[0]|1, START[1]|1)]
    maze[stack[0][1], stack[0][0]] = 0
    visited={stack[0]}
    while stack:
        x,y=stack[-1]
        nbs=[n for n in neighbors(x,y) if (n[0],n[1]) not in visited]
        if not nbs:
            stack.pop(); continue
        nx,ny,dx,dy=random.choice(nbs)
        maze[y+dy//2, x+dx//2]=0; maze[ny, nx]=0
        visited.add((nx,ny)); stack.append((nx,ny))
    maze[GOAL[1],GOAL[0]]=0
    # add some loops
    loops=0; attempts=0
    while loops<35 and attempts<600:
        attempts+=1
        x=random.randint(2,SIZE-3); y=random.randint(2,SIZE-3)
        if maze[y,x]==1:
            open_cnt=sum(1 for dx,dy in [(0,1),(0,-1),(1,0),(-1,0)] if maze[y+dy,x+dx]==0)
            if open_cnt>=2:
                maze[y,x]=0; loops+=1
    return maze


def run():
    summaries=[]
    for seed in range(RUNS):
        maze=generate(seed)
        nav=MazeNavigator(maze, START, GOAL, wiring_strategy='simple', gedig_threshold=GEDIG_THRESHOLD, backtrack_threshold=BACKTRACK_THRESHOLD, use_escalation=True, escalation_threshold='dynamic')
        nav.run(MAX_STEPS)
        backtrack_triggers=sum(1 for e in nav.event_log if e['type']=='backtrack_trigger')
        backtrack_steps=sum(1 for e in nav.event_log if e['type']=='backtrack_step')
        summaries.append({
            'seed':seed,
            'steps':nav.step_count,
            'goal':nav.is_goal_reached,
            'unique':len(set(nav.path)),
            'backtrack_triggers':backtrack_triggers,
            'backtrack_steps':backtrack_steps,
            'gedig_mean':float(np.mean(nav.gedig_history)) if nav.gedig_history else None,
        })
    print('=== Backtrack 25x25 Experiment ===')
    for s in summaries:
        print(s)
    if summaries:
        bt_total=sum(s['backtrack_triggers'] for s in summaries)
        print(f'Total backtrack triggers: {bt_total}')

if __name__=='__main__':
    run()
