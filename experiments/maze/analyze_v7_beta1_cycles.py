#!/usr/bin/env python3
"""v7 exploration: characterize the β₁ cycle structure of sleep-time agent graphs.

Question (design premise, before building the DG-normalization ablation):
do the agent's graph cycles have a SIZE SPREAD — i.e., are there large
shortcut-induced cycles distinguishable from small local-wiring cycles?
If yes, size-weighting (design option ①) is meaningful; if all cycles are
tiny/uniform, the β₁-DG signal is weak and the whole line needs rethinking.

Method: project the (query/direction bipartite) graph onto CELLS, split
cell-edges into corridor (spatially adjacent) vs shortcut (non-adjacent =
DG commits), and for each shortcut measure its cycle size = corridor-only
shortest path between endpoints. Report the size distribution + whether
large cycles reach toward the goal (crude discrimination proxy).

NOT confirmatory — exploratory characterization on a few dumped graphs.
"""
from __future__ import annotations
import glob
import json
import sys
from collections import defaultdict

import networkx as nx


def cell(n):
    return (int(n[0]), int(n[1]))


def load_cell_graph(path):
    g = json.load(open(path))
    goal = tuple(g["goal_pos"])
    start = tuple(g["start_pos"])
    # Build cell-level adjacency from spatial edges (endpoints in different cells)
    corridor = nx.Graph()
    shortcuts = []  # (u_cell, v_cell)
    seen = set()
    for a, b in g["edges"]:
        ca, cb = cell(a), cell(b)
        if ca == cb:
            continue  # same-cell query<->direction wiring
        key = tuple(sorted([ca, cb]))
        if key in seen:
            continue
        seen.add(key)
        manh = abs(ca[0] - cb[0]) + abs(ca[1] - cb[1])
        if manh == 1:
            corridor.add_edge(ca, cb)
        else:
            shortcuts.append((ca, cb, manh))
    # ensure corridor has all cells that appear
    for ca, cb, _ in shortcuts:
        corridor.add_node(ca)
        corridor.add_node(cb)
    return corridor, shortcuts, goal, start


def analyze(path):
    corridor, shortcuts, goal, start = load_cell_graph(path)
    ncells = corridor.number_of_nodes()
    ncorr = corridor.number_of_edges()
    recall_manh = sorted(m for _, _, m in shortcuts)  # Manhattan reach of recall edges
    # cycle size for each shortcut = corridor-only shortest path between endpoints + 1
    sizes = []
    goal_reaching = 0  # shortcut whose endpoint is within a few cells of goal
    for u, v, manh in shortcuts:
        try:
            d = nx.shortest_path_length(corridor, u, v)
            sizes.append(d + 1)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            sizes.append(None)  # endpoints not corridor-connected
        gd_u = abs(u[0] - goal[0]) + abs(u[1] - goal[1])
        gd_v = abs(v[0] - goal[0]) + abs(v[1] - goal[1])
        if min(gd_u, gd_v) <= 3:
            goal_reaching += 1
    real = [s for s in sizes if s is not None]
    return {
        "path": path.split("/")[-1],
        "cells": ncells,
        "corridor_edges": ncorr,
        "shortcuts": len(shortcuts),
        "cycle_sizes": sorted(real, reverse=True),
        "size_min": min(real) if real else None,
        "size_max": max(real) if real else None,
        "size_median": sorted(real)[len(real) // 2] if real else None,
        "disconnected_shortcuts": sum(1 for s in sizes if s is None),
        "shortcuts_near_goal": goal_reaching,
        "recall_manhattan": recall_manh,
        "recall_manh_max": max(recall_manh) if recall_manh else 0,
    }


def main():
    paths = sorted(glob.glob("results/graph_persistent_dg/_exploratory_v7_beta1/graph_seed*.json"))
    if len(sys.argv) > 1:
        paths = sys.argv[1:]
    if not paths:
        print("no graph dumps found", file=sys.stderr)
        return 1
    print(f"analyzing {len(paths)} graph(s)\n")
    for p in paths:
        r = analyze(p)
        big = [s for s in r["cycle_sizes"] if s >= 6]
        small = [s for s in r["cycle_sizes"] if s < 6]
        print(f"{r['path']}: cells={r['cells']} corridor={r['corridor_edges']} "
              f"shortcuts={r['shortcuts']} (disconnected={r['disconnected_shortcuts']})")
        print(f"  cycle size: min={r['size_min']} median={r['size_median']} max={r['size_max']} "
              f"| big(≥6)={len(big)} small(<6)={len(small)} | near-goal shortcuts={r['shortcuts_near_goal']}")
        print(f"  recall Manhattan reach: max={r['recall_manh_max']} dist={r['recall_manhattan']}")
        print(f"  top sizes: {r['cycle_sizes'][:12]}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
