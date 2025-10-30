"""Maze layout generation utilities for 25x25 benchmarks.

Separated out from test/benchmark scripts so they can be reused
by visualization tools and future experiments without creating
import cycles or relying on package-relative imports.
"""
from __future__ import annotations

import numpy as np

# Default start/goal pairs used in existing benchmarks
LARGE_DEFAULT_START = (12, 23)
LARGE_DEFAULT_GOAL = (20, 1)
COMPLEX_DEFAULT_START = (3, 21)
COMPLEX_DEFAULT_GOAL = (21, 3)


def create_large_maze() -> np.ndarray:
    """Structured 25x25 maze with layered horizontal and vertical corridors.

    Mirrors the layout previously embedded in test_25x25_maze.py.
    """
    maze = np.ones((25, 25), dtype=int)

    # Main vertical corridor
    for y in range(1, 24):
        maze[y, 12] = 0

    # Horizontal corridors (multi-layer)
    for x in range(1, 24):
        maze[4, x] = 0   # upper
        maze[8, x] = 0   # upper-mid
        maze[12, x] = 0  # middle
        maze[16, x] = 0  # lower-mid
        maze[20, x] = 0  # lower

    # Vertical connectors
    for y in range(1, 24):
        maze[y, 4] = 0   # left side
        maze[y, 8] = 0   # left-mid
        maze[y, 16] = 0  # right-mid
        maze[y, 20] = 0  # right side

    # Dead-end clusters
    for x in range(1, 4):
        maze[6, x] = 0
    for x in range(21, 24):
        maze[18, x] = 0

    return maze


def create_complex_maze() -> np.ndarray:
    """Higher branching & dead-end density 25x25 maze variant."""
    maze = np.ones((25, 25), dtype=int)

    # Mesh corridors
    for y in range(1, 24, 2):
        maze[y, 1:24] = 0
    for x in range(1, 24, 3):
        maze[1:24, x] = 0

    # Dead-end clusters
    for base in [(5, 5), (15, 5), (5, 15), (15, 15)]:
        bx, by = base
        for dx, dy in [(1, 0), (2, 0), (0, 1), (0, 2)]:
            if 1 <= bx + dx < 24 and 1 <= by + dy < 24:
                maze[by + dy, bx + dx] = 0

    # Spiral-like corridors (central area)
    for i in range(3, 20):
        maze[3, i] = 0
        maze[21, i] = 0
    for j in range(3, 22):
        maze[j, 3] = 0
        maze[j, 21] = 0
    for i in range(5, 20):
        maze[5, i] = 0
        maze[19, i] = 0
    for j in range(5, 20):
        maze[j, 5] = 0
        maze[j, 19] = 0

    # Open central core
    maze[11:14, 11:14] = 0

    return maze


__all__ = [
    'create_large_maze',
    'create_complex_maze',
    'LARGE_DEFAULT_START',
    'LARGE_DEFAULT_GOAL',
    'COMPLEX_DEFAULT_START',
    'COMPLEX_DEFAULT_GOAL',
]

# Additional advanced / alternative layouts
PERFECT_DEFAULT_START = (1, 1)
PERFECT_DEFAULT_GOAL = (23, 23)

def create_perfect_maze(seed: int = 42) -> np.ndarray:
    """Generate a near-perfect maze (DFS backtracker + a few extra openings).

    Perfect = single unique path between any two cells; we optionally
    add a handful of extra connections to avoid excessive linearity.
    """
    rng = np.random.default_rng(seed)
    w = h = 25
    maze = np.ones((h, w), dtype=int)

    # Cells at odd coordinates
    visited = set()
    stack = [(1, 1)]
    visited.add((1, 1))
    maze[1, 1] = 0
    directions = [(-2,0),(2,0),(0,-2),(0,2)]

    while stack:
        cx, cy = stack[-1]
        rng.shuffle(directions)
        extended = False
        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            if 1 <= nx < w-1 and 1 <= ny < h-1 and (nx, ny) not in visited:
                maze[cy + dy//2, cx + dx//2] = 0  # carve wall
                maze[ny, nx] = 0
                visited.add((nx, ny))
                stack.append((nx, ny))
                extended = True
                break
        if not extended:
            stack.pop()

    # Add a few random extra openings to create limited alternative routes
    for _ in range(18):
        x = rng.integers(2, w-2)
        y = rng.integers(2, h-2)
        if maze[y, x] == 1:
            open_neighbors = sum(maze[y+dy, x+dx] == 0 for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)])
            if open_neighbors >= 1:
                maze[y, x] = 0

    # Ensure start/goal open
    maze[PERFECT_DEFAULT_START[1], PERFECT_DEFAULT_START[0]] = 0
    maze[PERFECT_DEFAULT_GOAL[1], PERFECT_DEFAULT_GOAL[0]] = 0
    return maze

__all__ += [
    'create_perfect_maze',
    'PERFECT_DEFAULT_START',
    'PERFECT_DEFAULT_GOAL',
]

# Ultra-complex variant (multiple carved regions + bridges + chambers)
ULTRA_DEFAULT_START = (1, 23)
ULTRA_DEFAULT_GOAL = (23, 1)

def create_ultra_maze(seed: int = 123) -> np.ndarray:
    """Generate a denser, higher-branching 25x25 maze.

    Construction strategy:
      1. Start with all walls.
      2. Run DFS backtracker from multiple seeds -> forest merging.
      3. Carve rectangular chambers to increase local openness.
      4. Add random bridge openings between near-parallel corridors.
      5. Ensure start / goal open and far apart.
    """
    rng = np.random.default_rng(seed)
    w = h = 25
    maze = np.ones((h, w), dtype=int)

    def carve_from(sx: int, sy: int):
        stack = [(sx, sy)]
        maze[sy, sx] = 0
        dirs = [(-2,0),(2,0),(0,-2),(0,2)]
        while stack:
            cx, cy = stack[-1]
            rng.shuffle(dirs)
            extended = False
            for dx, dy in dirs:
                nx, ny = cx + dx, cy + dy
                if 1 <= nx < w-1 and 1 <= ny < h-1 and maze[ny, nx] == 1:
                    if maze[cy + dy//2, cx + dx//2] == 1:
                        maze[cy + dy//2, cx + dx//2] = 0
                    maze[ny, nx] = 0
                    stack.append((nx, ny))
                    extended = True
                    break
            if not extended:
                stack.pop()

    # Multiple DFS seeds (forest)
    seeds = [(1,1),(23,23),(1,13),(13,1),(13,23),(23,13)]
    for sx, sy in seeds:
        carve_from(sx, sy)

    # Chambers (open rectangles)
    chambers = [ (5,5,4,4), (15,5,5,4), (5,15,4,5), (15,15,5,5) ]
    for x,y,wc,hc in chambers:
        for yy in range(y, min(y+hc, h-1)):
            for xx in range(x, min(x+wc, w-1)):
                maze[yy, xx] = 0

    # Add random bridges (knock down walls adjacent to two passages)
    for _ in range(120):
        x = rng.integers(2, w-2)
        y = rng.integers(2, h-2)
        if maze[y, x] == 1:
            open_neighbors = sum(maze[y+dy, x+dx] == 0 for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)])
            if 1 <= open_neighbors <= 2 and rng.random() < 0.55:
                maze[y, x] = 0

    # Thin some overly open regions by re-adding sparse walls to avoid trivial corridors
    for _ in range(40):
        x = rng.integers(2, w-2)
        y = rng.integers(2, h-2)
        if maze[y, x] == 0:
            open_neighbors = sum(maze[y+dy, x+dx] == 0 for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)])
            if open_neighbors == 4 and rng.random() < 0.3:
                maze[y, x] = 1

    # Ensure start/goal cells open
    maze[ULTRA_DEFAULT_START[1], ULTRA_DEFAULT_START[0]] = 0
    maze[ULTRA_DEFAULT_GOAL[1], ULTRA_DEFAULT_GOAL[0]] = 0
    return maze

__all__ += [
    'create_ultra_maze',
    'ULTRA_DEFAULT_START',
    'ULTRA_DEFAULT_GOAL',
]

# --- Scaled Ultra (50x50) variant -------------------------------------------------
ULTRA50_DEFAULT_START = (1, 48)
ULTRA50_DEFAULT_GOAL = (48, 1)

def create_ultra_maze_50(seed: int = 456) -> np.ndarray:
    """Generate a larger (50x50) ultra-style maze.

    Scaling strategy:
      * Increase grid to 50.
      * Forest DFS seeds placed on a coarse 4x4 grid + corners to ensure coverage.
      * Chambers scaled (larger rectangular open regions) to create macro-rooms.
      * Number of random bridges / thinning passes scaled ~ area factor (~4x vs 25x25).

    Rationale: Encourage higher branching & multiple dead ends so that geDIG score
    variance increases and backtrack triggers become more frequent.
    """
    rng = np.random.default_rng(seed)
    w = h = 50
    maze = np.ones((h, w), dtype=int)

    def carve_from(sx: int, sy: int):
        stack = [(sx, sy)]
        maze[sy, sx] = 0
        dirs = [(-2,0),(2,0),(0,-2),(0,2)]
        while stack:
            cx, cy = stack[-1]
            rng.shuffle(dirs)
            extended = False
            for dx, dy in dirs:
                nx, ny = cx + dx, cy + dy
                if 1 <= nx < w-1 and 1 <= ny < h-1 and maze[ny, nx] == 1:
                    if maze[cy + dy//2, cx + dx//2] == 1:
                        maze[cy + dy//2, cx + dx//2] = 0
                    maze[ny, nx] = 0
                    stack.append((nx, ny))
                    extended = True
                    break
            if not extended:
                stack.pop()

    # Seed positions: corners + near-center grid
    base_points = [1, 13, 25, 37, 48]
    seeds = []
    for sx in base_points:
        for sy in base_points:
            if (sx + sy) % 2 == 0:  # sparsify
                seeds.append((sx, sy))
    # Ensure corners
    for c in [(1,1),(48,48),(1,48),(48,1)]:
        if c not in seeds:
            seeds.append(c)
    for sx, sy in seeds:
        carve_from(sx, sy)

    # Chambers: bigger rectangles in four quadrants & center ring
    chambers = [
        (6, 6, 8, 7), (34, 6, 8, 7),
        (6, 34, 8, 8), (34, 34, 8, 8),
        (20, 20, 10, 10)
    ]
    for x,y,wc,hc in chambers:
        for yy in range(y, min(y+hc, h-1)):
            for xx in range(x, min(x+wc, w-1)):
                maze[yy, xx] = 0

    # Bridges (scaled up)
    for _ in range(480):
        x = rng.integers(2, w-2)
        y = rng.integers(2, h-2)
        if maze[y, x] == 1:
            open_neighbors = sum(maze[y+dy, x+dx] == 0 for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)])
            if 1 <= open_neighbors <= 2 and rng.random() < 0.55:
                maze[y, x] = 0

    # Thinning overly open (scale factor)
    for _ in range(160):
        x = rng.integers(2, w-2); y = rng.integers(2, h-2)
        if maze[y, x] == 0:
            open_neighbors = sum(maze[y+dy, x+dx] == 0 for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)])
            if open_neighbors == 4 and rng.random() < 0.3:
                maze[y, x] = 1

    # Ensure start/goal
    maze[ULTRA50_DEFAULT_START[1], ULTRA50_DEFAULT_START[0]] = 0
    maze[ULTRA50_DEFAULT_GOAL[1], ULTRA50_DEFAULT_GOAL[0]] = 0
    return maze

__all__ += [
    'create_ultra_maze_50',
    'ULTRA50_DEFAULT_START',
    'ULTRA50_DEFAULT_GOAL'
]

# --- Dead-end enriched 50x50 variant -------------------------------------------
ULTRA50D_DEFAULT_START = ULTRA50_DEFAULT_START
ULTRA50D_DEFAULT_GOAL = ULTRA50_DEFAULT_GOAL

def create_ultra_maze_50_deadends(seed: int = 789, base_seed: int | None = None,
                                  target_deadend_ratio: float = 0.06,
                                  max_attempts: int = 20000) -> np.ndarray:
    """Create a 50x50 maze with enriched short dead-end spurs to provoke backtracking.

    Strategy:
      1. Start from the standard 50x50 ultra maze (moderately open, low dead-end ratio ~1%).
      2. Iteratively carve short spur tunnels (length 1-3) off existing corridors where a wall pocket exists.
      3. Periodically measure dead-end ratio (degree==1 open cells). Stop when target reached or attempts exhausted.

    This keeps global connectivity (only carving) and increases number of shallow cul-de-sacs.
    """
    rng = np.random.default_rng(seed)
    core = create_ultra_maze_50(seed=base_seed or seed)
    h, w = core.shape

    def dead_end_stats(m: np.ndarray):
        open_cells = [(x,y) for y in range(h) for x in range(w) if m[y,x]==0]
        de = 0
        for x,y in open_cells:
            deg = 0
            for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx,ny = x+dx,y+dy
                if 0<=nx<w and 0<=ny<h and m[ny,nx]==0:
                    deg += 1
            if deg == 1:
                de += 1
        return de, len(open_cells)

    def current_ratio(m):
        de, total = dead_end_stats(m)
        return de/total if total else 0.0, de, total

    ratio, de_cnt, total_open = current_ratio(core)
    attempts = 0
    # candidate branch origins: cells with degree 2 or 3 to diversify
    stagnation = 0
    while ratio < target_deadend_ratio and attempts < max_attempts:
        attempts += 1
        y = rng.integers(2, h-2)
        x = rng.integers(2, w-2)
        if core[y, x] != 0:
            continue
        nbrs = []
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x + dx, y + dy
            if core[ny, nx] == 0:
                nbrs.append((nx, ny))
        deg = len(nbrs)
        if deg > 3 or deg < 2:
            continue
        dirs = [(-1,0),(1,0),(0,-1),(0,1)]
        rng.shuffle(dirs)
        carved = False
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            nx2, ny2 = x + 2*dx, y + 2*dy
            if not (1 <= nx2 < w-1 and 1 <= ny2 < h-1):
                continue
            if core[ny, nx] == 1 and core[ny2, nx2] == 1:
                side_clear = 0
                for sdx, sdy in [(-dy, dx), (dy, -dx)]:
                    sx, sy = nx + sdx, ny + sdy
                    if 0 <= sx < w and 0 <= sy < h and core[sy, sx] == 0:
                        side_clear += 1
                if side_clear > 1:
                    continue
                length_choices = [1,2,3,4]
                probs = np.array([0.25,0.35,0.25,0.15])
                length = int(rng.choice(length_choices, p=probs))
                for step in range(length):
                    cx, cy = x + (step+1)*dx, y + (step+1)*dy
                    if not (0 <= cx < w and 0 <= cy < h):
                        break
                    if core[cy, cx] == 0:
                        break
                    core[cy, cx] = 0
                carved = True
                break
        if carved:
            stagnation = 0
        else:
            stagnation += 1
        if (carved and (attempts % 40 == 0)) or (stagnation > 120 and attempts % 10 == 0):
            ratio, de_cnt, total_open = current_ratio(core)
        if stagnation > 200 and deg == 1:
            # allow deg==1 origins after heavy stagnation (already degree filtered earlier)
            pass
    # Final ratio update
    ratio, de_cnt, total_open = current_ratio(core)
    # Guarantee start/goal open
    core[ULTRA50D_DEFAULT_START[1], ULTRA50D_DEFAULT_START[0]] = 0
    core[ULTRA50D_DEFAULT_GOAL[1], ULTRA50D_DEFAULT_GOAL[0]] = 0
    return core

__all__ += [
    'create_ultra_maze_50_deadends',
    'ULTRA50D_DEFAULT_START',
    'ULTRA50D_DEFAULT_GOAL'
]

# === (1) High dead-end dense (perfect) maze variant ==============================
ULTRA50HD_DEFAULT_START = ULTRA50_DEFAULT_START
ULTRA50HD_DEFAULT_GOAL = ULTRA50_DEFAULT_GOAL

def create_ultra_maze_50_dense_deadends(seed: int = 9101, width: int = 50, height: int = 50) -> np.ndarray:
    """Generate a high-dead-end 50x50 perfect maze (DFS backtracker) producing many cul-de-sacs.

    Characteristics:
      * Perfect maze (acyclic) ensures dead-end ratio typically ~35-55% of open cells.
      * Carving grid uses odd coordinates for cells, walls on even lines (classic algorithm adapted to fixed size 50).
    """
    rng = np.random.default_rng(seed)
    w, h = width, height
    maze = np.ones((h, w), dtype=int)
    # Use odd coordinate cells inside border
    cells = [(x, y) for y in range(1, h-1, 2) for x in range(1, w-1, 2)]
    for x, y in cells:
        maze[y, x] = 0
    # DFS stack
    stack = []
    start_cell = (1, 1)
    stack.append(start_cell)
    visited = {start_cell}
    dirs = [(-2,0),(2,0),(0,-2),(0,2)]
    while stack:
        cx, cy = stack[-1]
        rng.shuffle(dirs)
        advanced = False
        for dx, dy in dirs:
            nx, ny = cx + dx, cy + dy
            if 1 <= nx < w-1 and 1 <= ny < h-1 and (nx, ny) not in visited:
                # carve passage
                maze[cy + dy//2, cx + dx//2] = 0
                visited.add((nx, ny))
                stack.append((nx, ny))
                advanced = True
                break
        if not advanced:
            stack.pop()
    # Guarantee start/goal open
    maze[ULTRA50HD_DEFAULT_START[1], ULTRA50HD_DEFAULT_START[0]] = 0
    maze[ULTRA50HD_DEFAULT_GOAL[1], ULTRA50HD_DEFAULT_GOAL[0]] = 0
    return maze

# === (2) Moderate dead-end variant (reopen passages) ============================
ULTRA50MD_DEFAULT_START = ULTRA50HD_DEFAULT_START
ULTRA50MD_DEFAULT_GOAL = ULTRA50HD_DEFAULT_GOAL

def create_ultra_maze_50_moderate_deadends(seed: int = 9202,
                                           target_range: tuple[float,float] = (0.08, 0.12),
                                           max_reopens: int = 5000) -> np.ndarray:
    """Start from dense dead-end perfect maze then reopen corridors to reduce dead-end ratio.

    Reopening strategy:
      * Identify current dead-end cells (degree==1).
      * Randomly pick some and knock down one wall to a neighboring passage two steps away (creating loop / junction).
      * Continue until dead-end ratio falls within target_range or max_reopens reached.

    Notes:
      * Maintains connectivity while introducing loops so geDIG variance includes both cul-de-sacs & branching.
    """
    lo, hi = target_range
    if not (0 < lo < hi < 1):
        raise ValueError('target_range must be (lo, hi) with 0<lo<hi<1')
    rng = np.random.default_rng(seed)
    base = create_ultra_maze_50_dense_deadends(seed=seed)
    h, w = base.shape

    def dead_end_cells(m):
        ends = []
        for y in range(1, h-1):
            for x in range(1, w-1):
                if m[y,x] != 0:
                    continue
                deg = 0
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    if m[y+dy, x+dx] == 0:
                        deg += 1
                if deg == 1:
                    ends.append((x,y))
        return ends

    def ratio(m):
        open_cells = np.count_nonzero(m == 0)
        ends = len(dead_end_cells(m))
        return ends / open_cells if open_cells else 0.0, ends, open_cells

    r, ends_cnt, open_cnt = ratio(base)
    reopens = 0
    # Candidate offset directions for knocking a wall: axial only
    while r > hi and reopens < max_reopens:
        ends = dead_end_cells(base)
        if not ends:
            break
        x, y = ends[rng.integers(0, len(ends))]
        # Determine the direction of existing corridor (the only neighbor open)
        open_dirs = [(dx,dy) for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)] if base[y+dy, x+dx] == 0]
        if not open_dirs:
            continue
        # Try perpendicular or opposite walls to open new loop
        trial_dirs = [d for d in [(-1,0),(1,0),(0,-1),(0,1)] if d not in open_dirs]
        rng.shuffle(trial_dirs)
        opened = False
        for dx, dy in trial_dirs:
            nx, ny = x + dx, y + dy
            if not (1 <= nx < w-1 and 1 <= ny < h-1):
                continue
            if base[ny, nx] == 1:
                # open wall; if behind it there's another wall, that's fine—we just create a side pocket
                base[ny, nx] = 0
                opened = True
                break
        if opened:
            reopens += 1
            if reopens % 200 == 0:
                r, ends_cnt, open_cnt = ratio(base)
        else:
            reopens += 1  # count attempt
    # Final ratio update
    r, ends_cnt, open_cnt = ratio(base)
    # If still above hi, accept; if below lo (overshoot) we stop anyway (simple heuristic)
    base[ULTRA50MD_DEFAULT_START[1], ULTRA50MD_DEFAULT_START[0]] = 0
    base[ULTRA50MD_DEFAULT_GOAL[1], ULTRA50MD_DEFAULT_GOAL[0]] = 0
    return base

__all__ += [
    'create_ultra_maze_50_dense_deadends',
    'ULTRA50HD_DEFAULT_START',
    'ULTRA50HD_DEFAULT_GOAL',
    'create_ultra_maze_50_moderate_deadends',
    'ULTRA50MD_DEFAULT_START',
    'ULTRA50MD_DEFAULT_GOAL'
]
