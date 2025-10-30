"""Quick sanity run to verify including wall episodes doesn't degrade basic behavior.

Runs a tiny synthetic maze multiple times and reports:
- total runs
- goal success count
- average steps
- wall_selected events count
- wall_selected rate per steps

This is a lightweight script; not a formal test.
"""
from navigation.maze_navigator import MazeNavigator
import numpy as np

# Simple 5x5 maze (0 = path, 1 = wall)
# Start (0,0), Goal (4,4)
MAZE = np.array([
    [0,0,0,0,0],
    [0,1,1,1,0],
    [0,0,0,1,0],
    [0,1,0,0,0],
    [0,0,0,1,0],
], dtype=int)

RUNS = 10
MAX_STEPS = 400

success = 0
steps_acc = 0
wall_selected_events = 0

for seed in range(RUNS):
    np.random.seed(seed)
    nav = MazeNavigator(MAZE, (0,0), (4,4), temperature=0.1, wiring_strategy='simple')
    reached = nav.run(max_steps=MAX_STEPS)
    if reached:
        success += 1
    steps_acc += nav.step_count
    # count wall_selected events
    wall_selected_events += sum(1 for e in nav.event_log if e['type'] == 'wall_selected')

avg_steps = steps_acc / RUNS
wall_rate = wall_selected_events / max(1, steps_acc)

print("Runs:", RUNS)
print("Success:", success)
print("Avg steps:", round(avg_steps,2))
print("Wall selections:", wall_selected_events)
print("Wall selection rate per step:", f"{wall_rate*100:.4f}%")

# Basic expectation sanity thresholds (heuristic):
# - success should be >= 8/10 (stochastic)
# - wall selection rate should be effectively 0 (<0.05%)
if success < 8:
    print("[WARN] Success count lower than expected; investigate.")
if wall_rate > 0.0005:
    print("[WARN] Wall selection rate unusually high.")
