"""Escalation smoke test for MazeNavigator multi-hop geDIG.
Creates a small maze with a branch and a dead-end to trigger escalation events.
"""
from __future__ import annotations

import os, sys
from collections import Counter

import numpy as np

# Adjust path to allow relative imports inside the enhanced package structure
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'navigation'))
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from navigation.maze_navigator import MazeNavigator  # type: ignore


def build_test_maze():
    # 0 = free, 1 = wall
    # Layout (7x7): S start, G goal, D dead-end, B branch point
    # Row representation (y):
    # S . . 1 . . .
    # 1 1 . 1 . 1 .
    # . . . . . 1 .
    # . 1 1 1 . 1 .
    # . . B . . . .
    # . 1 1 1 1 1 .
    # . . . . D . G
    maze = np.array([
        [0,0,0,1,0,0,0],
        [1,1,0,1,0,1,0],
        [0,0,0,0,0,1,0],
        [0,1,1,1,0,1,0],
        [0,0,0,0,0,0,0],
        [0,1,1,1,1,1,0],
        [0,0,0,0,0,0,0],
    ], dtype=int)
    start = (0,0)
    goal = (6,6)
    return maze, start, goal


def main():
    maze, start, goal = build_test_maze()
    nav = MazeNavigator(
        maze=maze,
        start_pos=start,
        goal_pos=goal,
        wiring_strategy='simple',  # changed to simpler wiring to reduce structural gains
        use_escalation=True,
    escalation_threshold=2.5,  # force escalation every step (score max is 2.0)
        gedig_threshold=0.25,
        backtrack_threshold=-0.05,  # less negative to classify low values sooner
    )

    nav.run(max_steps=250)

    # Debug: show first 8 structural records
    print('\nFirst structural records (debug):')
    for rec in nav.gedig_structural[:8]:
        print({k: rec.get(k) for k in ['step','value','escalated','shortcut','dead_end']})

    escalated_records = [r for r in nav.gedig_structural if r.get('escalated')]
    shortcut = [r for r in escalated_records if r.get('shortcut')]
    dead_end = [r for r in escalated_records if r.get('dead_end')]
    branch_rem = [r for r in escalated_records if r.get('branch_reminder')]

    events_by_type = Counter(e['type'] for e in nav.event_log)

    print('\n=== Escalation Smoke Test Summary ===')
    print(f"Total steps: {nav.step_count}")
    print(f"Goal reached: {nav.is_goal_reached}")
    print(f"Total geDIG evaluations: {len(nav.gedig_structural)}")
    print(f"Escalated count: {len(escalated_records)}")
    print(f"  Shortcut candidates: {len(shortcut)}  Dead-ends: {len(dead_end)}  Branch reminders: {len(branch_rem)}")
    print('Event counts:', dict(events_by_type))

    print('\nLast escalated entries (up to 5):')
    for rec in escalated_records[-5:]:
        # 'branch_reminder' is not part of structural records; access defensively
        print({k: rec.get(k) for k in ['step','value','escalated','shortcut','dead_end','branch_reminder']})

    print('\nEvent log tail (last 15):')
    for ev in nav.event_log[-15:]:
        print(ev)


if __name__ == '__main__':
    main()
