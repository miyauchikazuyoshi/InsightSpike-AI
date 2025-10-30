"""Complex maze natural exploration probe.
Generates a maze with multiple side branches (some dead-ends) to observe:
  - branch_entry / branch_completion natural timing
  - geDIG hop0 trajectory around each completion
  - escalation frequency without forcing (moderate threshold)
Outputs summarized statistics.
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


def build_complex_maze():
    # 15x11 maze (w=15,h=11) with corridors and pockets.
    # 0 free, 1 wall. Crafted to create several 3+ open neighbor junctions.
    grid = [
        "000001000000000",
        "011101011111010",
        "000001010001010",
        "010111010101010",
        "010000010101000",
        "010111110101110",
        "010100000101000",
        "010101011101010",
        "010101000001010",
        "010101111101010",
        "000100000001000",
    ]
    maze = np.array([[1 if c=='1' else 0 for c in row] for row in grid], dtype=int)
    start = (0,0)
    goal = (14,10)
    return maze, start, goal


def summarize(nav: MazeNavigator):
    # Collect branch completion steps
    completions = [e for e in nav.event_log if e['type']=='branch_completion']
    entries = [e for e in nav.event_log if e['type']=='branch_entry']
    rec_by_step = {r['step']: r for r in nav.gedig_structural}
    print('\n=== Complex Maze Probe Summary ===')
    print(f"Steps={nav.step_count} Goal={nav.is_goal_reached} geDIG_evals={len(nav.gedig_structural)}")
    print(f"Branch entries={len(entries)} completions={len(completions)}")
    if entries:
        print(' First 5 entries:', entries[:5])
    if completions:
        print(' First 5 completions:', completions[:5])
    # For each completion show window
    for ev in completions[:3]:
        s = ev['step']
        print(f"\n-- Completion window around step {s} --")
        for t in range(max(0,s-4), s+2):
            r = rec_by_step.get(t)
            if not r: continue
            print({k:r.get(k) for k in ['step','value','nodes_added','edges_added','escalated','shortcut','dead_end']})
    # Escalation rate
    esc = [r for r in nav.gedig_structural if r.get('escalated')]
    if nav.gedig_structural:
        print(f"Escalation rate: {len(esc)}/{len(nav.gedig_structural)} = {len(esc)/len(nav.gedig_structural):.2%}")


def main():
    maze,start,goal = build_complex_maze()
    nav = MazeNavigator(
        maze=maze,
        start_pos=start,
        goal_pos=goal,
        wiring_strategy='simple',  # keep structure growth natural
        use_escalation=True,
        escalation_threshold=0.15,  # moderate threshold (tuned below midpoint≈0.31)
        gedig_threshold=0.25,
        backtrack_threshold=-0.05,
    )
    nav.run(max_steps=1200)
    summarize(nav)

if __name__=='__main__':
    main()
