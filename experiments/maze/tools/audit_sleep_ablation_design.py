#!/usr/bin/env python3
"""Audit: was there room for sleep to matter, per seed?

For each ablation seed: maze structure (branch points, dead-end corridors,
shortest path), warmup experience (dead-end encounters), eval efficiency
(steps vs shortest possible), and the structural possibility of a
goal-gradient (did warmup reach goal -> does a +1.0 reward exist at all).
Then check v6_perseed: how often did warmup-FAIL flip to eval-OK, in
baseline (8D) vs extended (10D+sleep)?
"""
import json
from collections import deque
from pathlib import Path

ABL = Path("/Users/miyauchikazuyoshi/Documents/GitHub/InsightSpike-AI/experiments/maze/results/graph_persistent_dg/sleep_ablation")
V6 = Path("/Users/miyauchikazuyoshi/Documents/GitHub/InsightSpike-AI/experiments/maze/results/graph_persistent_dg/v6_perseed")


def maze_stats(layout, start, goal):
    n = len(layout)
    # encoding: 0=corridor, 1=wall, 2=start, 3=goal
    open_cells = {(r, c) for r in range(n) for c in range(len(layout[0])) if layout[r][c] != 1}
    def nbrs(p):
        r, c = p
        return [(r+dr, c+dc) for dr, dc in ((0,1),(0,-1),(1,0),(-1,0)) if (r+dr, c+dc) in open_cells]
    deg = {p: len(nbrs(p)) for p in open_cells}
    branch3 = sum(1 for d in deg.values() if d == 3)   # T-junctions
    branch4 = sum(1 for d in deg.values() if d == 4)   # crossroads
    deadends = sum(1 for p, d in deg.items() if d == 1 and p not in (tuple(start), tuple(goal)))
    # BFS shortest path start->goal
    q, seen = deque([(tuple(start), 0)]), {tuple(start)}
    sp = None
    while q:
        p, dist = q.popleft()
        if p == tuple(goal):
            sp = dist
            break
        for nb in nbrs(p):
            if nb not in seen:
                seen.add(nb)
                q.append((nb, dist + 1))
    return dict(open=len(open_cells), t_junctions=branch3, crossroads=branch4,
                deadend_cells=deadends, shortest=sp)


def load(path):
    with open(path) as f:
        return json.load(f)


print("=" * 100)
print("PART 1: ablation seeds 0-29 — maze structure vs warmup experience vs eval efficiency (on-arm data)")
print("=" * 100)
hdr = f"{'seed':>4} {'T-junc':>6} {'cross':>5} {'dead_cells':>10} {'SP*':>4} | {'wu_ok':>5} {'wu_steps':>8} {'wu_deadends':>11} | {'ev_ok':>5} {'ev_steps':>8} {'ev/SP':>6} {'ev_deadends':>11}"
print(hdr)
rows = []
for seed in range(30):
    d = load(ABL / f"on_seed{seed}.json")
    md = d["maze_data"][str(seed)]
    ms = maze_stats(md["layout"], md["start_pos"], md["goal_pos"])
    wu = d["warmup_runs"][0]
    ev = d["runs"][0]
    ratio = ev["steps"] / ms["shortest"] if ms["shortest"] and ev["success"] else float("nan")
    rows.append(dict(seed=seed, **ms, wu_ok=wu["success"], wu_steps=wu["steps"],
                     wu_de=wu.get("dead_end_steps"), ev_ok=ev["success"],
                     ev_steps=ev["steps"], ev_ratio=ratio, ev_de=ev.get("dead_end_steps")))
    print(f"{seed:>4} {ms['t_junctions']:>6} {ms['crossroads']:>5} {ms['deadend_cells']:>10} {ms['shortest']:>4} | "
          f"{str(wu['success']):>5} {wu['steps']:>8} {str(wu.get('dead_end_steps')):>11} | "
          f"{str(ev['success']):>5} {ev['steps']:>8} {ratio:>6.2f} {str(ev.get('dead_end_steps')):>11}")

ok = [r for r in rows if r["ev_ok"]]
print(f"\nSummary (eval-success seeds, n={len(ok)}):")
print(f"  mean eval/shortest ratio: {sum(r['ev_ratio'] for r in ok)/len(ok):.3f}  "
      f"(1.00 = optimal; how much room was left to shorten?)")
print(f"  eval dead-end encounters: total {sum(r['ev_de'] or 0 for r in ok)} across all eval-success runs")
print(f"  warmup dead-end encounters: mean {sum(r['wu_de'] or 0 for r in rows)/len(rows):.1f} per warmup")
perfect = sum(1 for r in ok if r["ev_ratio"] <= 1.0 + 1e-9)
print(f"  eval runs at EXACT shortest path: {perfect}/{len(ok)}")

print()
print("=" * 100)
print("PART 2: v6_perseed — warmup-FAIL -> eval-OK flips, baseline (8D) vs extended (10D+sleep)")
print("=" * 100)
for mode in ("baseline", "extended"):
    files = sorted(V6.glob(f"{mode}_seed*.json"))
    n = flips = wu_fail = ev_ok_total = 0
    lifted_seeds = []
    for fp in files:
        try:
            d = load(fp)
            w = d.get("warmup_runs", [{}])[0]
            e = d.get("runs", [{}])[0]
        except Exception:
            continue
        if "success" not in w or "success" not in e:
            continue
        n += 1
        ev_ok_total += bool(e.get("success"))
        if not w.get("success"):
            wu_fail += 1
            if e.get("success"):
                flips += 1
                lifted_seeds.append(w.get("seed"))
    print(f"  {mode:>9}: n={n}, eval-OK {ev_ok_total}/{n}, warmup-FAIL {wu_fail}, "
          f"of which eval-OK (the 'lifted' cases): {flips}  {('seeds: ' + str(lifted_seeds)) if lifted_seeds else ''}")
print()
print("Interpretation aid: if 'lifted' cases exist mainly in extended, whatever lifts them")
print("cannot be goal-gradient propagation when warmup never reached goal (no +1.0 in graph);")
print("candidates: dim8 raw-reward tagging, dictionary guidance, 10D distance geometry.")
