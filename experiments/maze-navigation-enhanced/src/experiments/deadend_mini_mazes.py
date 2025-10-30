#!/usr/bin/env python3
"""Hand-crafted small mazes for dead-end / branch phase probing.

Maze encoding: 0 = open, 1 = wall. Coordinate: (x,y) with maze[y,x].
Each function returns (maze, start, goal, name, description)
"""
from __future__ import annotations
import numpy as np

# Utility to build empty filled with walls then carve

def empty(w,h):
    m = np.ones((h,w), dtype=int)
    return m

def carve_line(m, pts):
    for x,y in pts:
        m[y,x]=0

# 1. Straight shallow dead-end: Start at left, short corridor, dead-end goal

def maze_shallow_deadend():
    m=empty(11,7)
    line=[(i,3) for i in range(1,9)]  # corridor
    carve_line(m,line)
    start=(1,3); goal=(8,3)
    name='shallow_deadend'
    desc='Single short corridor ending in dead-end.'
    return m,start,goal,name,desc

# 2. Deep cul-de-sac: longer corridor with two slight bends

def maze_deep_deadend():
    m=empty(15,11)
    path=[(1,5),(2,5),(3,5),(4,5),(5,5),(6,5),(7,5),(8,5),(9,5),(10,5),(10,4),(10,3),(11,3),(12,3)]
    carve_line(m,path)
    start=(1,5); goal=(12,3)
    name='deep_deadend'
    desc='Longer corridor with a bend leading to deep dead-end.'
    return m,start,goal,name,desc

# 3. Parallel branches: junction with left dead-end and right path to goal

def maze_parallel_branch():
    m=empty(15,9)
    spine=[(i,4) for i in range(1,7)]
    left_branch=[(3,4),(3,3),(3,2)]  # dead-end up
    right_branch=[(6,4),(7,4),(8,4),(9,4),(10,4),(11,4)]
    goal_path=[(11,4),(11,5),(11,6)]
    carve_line(m, spine+left_branch+right_branch+goal_path)
    start=(1,4); goal=(11,6)
    name='parallel_branch'
    desc='Junction: short dead-end upward vs longer path to goal to the right.'
    return m,start,goal,name,desc

# 4. Cascading dead-ends (stair): sequence of side pockets causing repeated dead-end hits

def maze_cascade_deadends():
    m=empty(17,11)
    main=[(i,5) for i in range(1,14)]
    pockets=[(3,4),(3,3),(6,6),(6,7),(9,4),(9,3),(12,6),(12,7)]
    carve_line(m, main+pockets+[(14,5)])
    start=(1,5); goal=(14,5)
    name='cascade_deadends'
    desc='Linear spine with alternating up/down short pockets (dead-ends).'
    return m,start,goal,name,desc

MAZE_BUILDERS=[maze_shallow_deadend, maze_deep_deadend, maze_parallel_branch, maze_cascade_deadends]
