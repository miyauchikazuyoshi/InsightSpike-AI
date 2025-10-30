"""Complex (larger) maze probe v2.

Goal: Provide richer branching (multiple 3+ degree junctions, deep dead-ends) so that
natural stochastic action selection will likely traverse and complete side branches
before reaching goal, enabling collection of branch_completion windows and geDIG
threshold calibration.

Strategy:
  - 25x17 maze handcrafted with a winding main spine from NW to SE.
  - Many side pockets (dead-ends) hanging off the spine.
  - At least 5 junctions (>=3 open neighbors) early, mid, late.
  - Goal placed far bottom-right; requires passing earlier junction then backtracking around a choke.

Outputs:
  - Summary counts and first few completion windows
  - Escalation rate (moderate threshold)
"""
from __future__ import annotations
import os, sys
from collections import defaultdict
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'navigation'))
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
for p in [PARENT_DIR, BASE_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from navigation.maze_navigator import MazeNavigator  # type: ignore

# Legend lines: 0=free 1=wall
MAZE_ASCII = [
    "0000010000000000000000000",
    "0111010111110111111101110",
    "0000010100010100000101000",
    "0101110101010101110101010",
    "0100000101010001010100010",
    "0101111101011101010111010",
    "0101000001010001010000010",
    "0101010111010101011111010",
    "0101010000010101000000010",
    "0101011111010101011111010",
    "0100010000010101000001010",
    "0111010111010101111101010",
    "0001010100010100000101010",
    "0101010101110111010101010",
    "0100010101000001010100010",
    "0111010101011101010111010",
    "0000000101000001000000010",
]

# Validate shape
WIDTH = len(MAZE_ASCII[0])
HEIGHT = len(MAZE_ASCII)
assert all(len(r)==WIDTH for r in MAZE_ASCII), "Inconsistent row width"


def build_maze():
    maze = np.array([[1 if c=='1' else 0 for c in row] for row in MAZE_ASCII], dtype=int)
    start = (0,0)
    goal = (WIDTH-2, HEIGHT-2)  # near bottom-right but not corner (leave last column partly walls)
    return maze, start, goal


def summarize(nav: MazeNavigator):
    entries = [e for e in nav.event_log if e['type']=='branch_entry']
    completions = [e for e in nav.event_log if e['type']=='branch_completion']
    rec_by_step = {r['step']: r for r in nav.gedig_structural}
    print('\n=== Complex Maze V2 Summary ===')
    print(f"Grid {WIDTH}x{HEIGHT} Steps={nav.step_count} Goal={nav.is_goal_reached} geDIG_evals={len(nav.gedig_structural)}")
    print(f"Branch entries={len(entries)} completions={len(completions)}")
    if entries:
        print(' First entries:', entries[:5])
    if completions:
        print(' First completions:', completions[:5])
    # Show up to first 5 completion windows
    for ev in completions[:5]:
        s = ev['step']
        print(f"\n-- Completion window step {s} --")
        for t in range(max(0,s-5), s+3):
            r = rec_by_step.get(t)
            if not r: continue
            print({k:r.get(k) for k in ['step','value','nodes_added','edges_added','escalated','shortcut','dead_end']})
    esc = [r for r in nav.gedig_structural if r.get('escalated')]
    if nav.gedig_structural:
        print(f"Escalation rate: {len(esc)}/{len(nav.gedig_structural)} = {len(esc)/len(nav.gedig_structural):.2%}")


def main():
    maze,start,goal = build_maze()
    nav = MazeNavigator(
        maze=maze,
        start_pos=start,
        goal_pos=goal,
        wiring_strategy='simple',
        use_escalation=True,
        escalation_threshold=0.18,  # slightly higher to get some escalations in bigger maze
        gedig_threshold=0.25,
        backtrack_threshold=-0.05,
    )
    nav.run(max_steps=4000)
    summarize(nav)

if __name__=='__main__':
    main()
