"""Branch completion & geDIG threshold exploratory study.

Runs a moderately complex maze with branches + a loop to:
- Capture geDIG distribution during branch exploration vs branch completion vs loop (shortcut) creation.
- Inspect episode similarity ranking (distance margin between best and second option) at branch completion (after dead-end fully explored).
- Propose tentative thresholds for:
  * wiring (positive structural change)
  * backtrack (negative spike)
  * shortcut detection (loop closure)
- Evaluate optional multi-hop geDIG decay usefulness.

This is exploratory; not a formal benchmark.
"""
from navigation.maze_navigator import MazeNavigator
from core.gedig_evaluator import GeDIGEvaluator
import numpy as np
import networkx as nx
from statistics import mean

# 15x15 maze with multiple branches and an intentional loop corridor.
# 0 = path, 1 = wall.
MAZE_COMPLEX = np.array([
 [0,0,0,0,0,1,0,0,0,0,1,0,0,0,0],
 [1,1,1,0,1,1,0,1,1,0,1,0,1,1,0],
 [0,0,0,0,0,0,0,0,1,0,0,0,0,1,0],
 [0,1,1,1,1,1,1,0,1,1,1,1,0,1,0],
 [0,0,0,0,0,0,1,0,0,0,0,1,0,0,0],
 [0,1,1,1,1,0,1,1,1,1,0,1,1,1,0],
 [0,0,0,0,1,0,0,0,0,1,0,0,0,1,0],
 [0,1,1,0,1,1,1,1,0,1,1,1,0,1,0],
 [0,0,0,0,0,0,0,1,0,0,0,1,0,1,0],
 [0,1,1,1,1,1,0,1,1,1,0,1,0,1,0],
 [0,0,0,0,0,0,0,0,0,1,0,0,0,1,0],
 [0,1,1,1,1,1,1,1,0,1,1,1,0,1,0],
 [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
 [0,1,1,1,1,1,1,1,1,1,0,1,1,1,0],
 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
], dtype=int)

START = (0,0)
GOAL = (14,14)
RUNS = 5
MAX_STEPS = 1500

def graph_density(g: nx.Graph) -> float:
    n = g.number_of_nodes()
    if n <= 1: return 0.0
    e = g.number_of_edges()
    return e / (n*(n-1)/2)

records = []
branch_completion_gedigs = []
branch_explore_gedigs = []
shortcut_gedigs = []
completion_margins = []  # distance(second) - distance(first)
completion_unvisited_ratio = []  # fraction of options with visit_count==0
multi_hop_samples = []

for seed in range(RUNS):
    np.random.seed(seed)
    nav = MazeNavigator(MAZE_COMPLEX, START, GOAL, temperature=0.1, wiring_strategy='simple')
    nav.run(MAX_STEPS)

    # Map step -> gedig (assumes one per step after first diff)
    gedig_map = {}
    for idx, val in enumerate(nav.gedig_history):
        # gedig_history index aligns with step after snapshot? approximate
        gedig_map[idx+1] = val

    # Iterate events
    for ev in nav.event_log:
        if ev['type'] == 'branch_entry':
            # Mark start of branch segment
            pass
        if ev['type'] == 'branch_completion':
            step = ev['step']
            gval = gedig_map.get(step, 0.0)
            branch_completion_gedigs.append(gval)
            # Retrieve last decision options (at or before step)
            if nav.decision_history:
                last_dec = nav.decision_history[-1]
                opts = last_dec['options']['options'] if 'options' in last_dec else {}
                distances = [info.get('distance') for info in opts.values() if 'distance' in info]
                if len(distances) >= 2:
                    ds = sorted(distances)
                    completion_margins.append(ds[1]-ds[0])
                # unvisited proxy: visit_count==0
                unv = [1 for info in opts.values() if info.get('visit_count') == 0 and not info.get('is_wall')]
                total = sum(1 for info in opts.values() if not info.get('is_wall'))
                if total>0:
                    completion_unvisited_ratio.append(len(unv)/total)
        if ev['type'] == 'branch_analysis':
            analysis = ev['message']
            gval = analysis.get('gedig_1hop', 0.0)
            # Heuristic: if not shortcut treat as exploration mid-branch
            if not analysis.get('is_shortcut'):
                branch_explore_gedigs.append(gval)
            else:
                shortcut_gedigs.append(gval)
                # multi-hop sample
                evaluator = GeDIGEvaluator()
                mh = evaluator.calculate_multihop(nav.graph_manager.graph_history[-2], nav.graph_manager.graph_history[-1])
                multi_hop_samples.append(list(mh.values()))

# Aggregate
def pct(values, p):
    if not values: return None
    arr = sorted(values)
    k = int((p/100)* (len(arr)-1))
    return arr[k]

summary = {
    'runs': RUNS,
    'branch_completion_count': len(branch_completion_gedigs),
    'branch_explore_samples': len(branch_explore_gedigs),
    'shortcut_samples': len(shortcut_gedigs),
    'mean_completion_gedig': round(mean(branch_completion_gedigs),4) if branch_completion_gedigs else None,
    'mean_explore_gedig': round(mean(branch_explore_gedigs),4) if branch_explore_gedigs else None,
    'mean_shortcut_gedig': round(mean(shortcut_gedigs),4) if shortcut_gedigs else None,
    'p10_explore_gedig': pct(branch_explore_gedigs,10),
    'p90_explore_gedig': pct(branch_explore_gedigs,90),
    'p10_completion_gedig': pct(branch_completion_gedigs,10),
    'p90_completion_gedig': pct(branch_completion_gedigs,90),
    'shortcut_min': min(shortcut_gedigs) if shortcut_gedigs else None,
    'shortcut_max': max(shortcut_gedigs) if shortcut_gedigs else None,
    'mean_completion_margin': round(mean(completion_margins),4) if completion_margins else None,
    'mean_completion_unvisited_ratio': round(mean(completion_unvisited_ratio),4) if completion_unvisited_ratio else None,
    'multi_hop_mean_decay_first3': None,
}

if multi_hop_samples:
    first3 = [s[:3] for s in multi_hop_samples if len(s)>=3]
    if first3:
        avg_first = np.mean([v[0] for v in first3])
        avg_second = np.mean([v[1] for v in first3])
        avg_third = np.mean([v[2] for v in first3])
        summary['multi_hop_mean_decay_first3'] = [round(avg_first,4), round(avg_second,4), round(avg_third,4)]

print("=== Branch & geDIG Threshold Study Summary ===")
for k,v in summary.items():
    print(f"{k}: {v}")

# Heuristic threshold suggestions
if branch_explore_gedigs:
    wiring_threshold = pct(branch_explore_gedigs, 60)  # above median structural additions
else:
    wiring_threshold = 0.3
if branch_completion_gedigs:
    backtrack_threshold = min(pct(branch_completion_gedigs,10), pct(branch_explore_gedigs,10) if branch_explore_gedigs else 0) - 0.01
else:
    backtrack_threshold = -0.2
if shortcut_gedigs:
    shortcut_trigger = max(max(shortcut_gedigs), -0.01)  # negative region boundary
    negative_band = (min(shortcut_gedigs), max(shortcut_gedigs))
else:
    shortcut_trigger = -0.05
    negative_band = None

print("\nSuggested Thresholds:")
print(f" wiring_threshold ≈ {round(wiring_threshold,3)}")
print(f" backtrack_threshold ≈ {round(backtrack_threshold,3)}")
print(f" shortcut_negative_band ≈ {negative_band}")

if summary['multi_hop_mean_decay_first3']:
    d = summary['multi_hop_mean_decay_first3']
    print(f" multi-hop decay sample (1hop→2hop→3hop): {d}")
    if abs(d[1]) < abs(d[0])*0.75 and abs(d[2]) < abs(d[1])*0.75:
        print(" multi-hop adds smooth temporal smoothing; could average first 2 hops for stability.")
