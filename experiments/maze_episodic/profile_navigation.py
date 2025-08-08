#!/usr/bin/env python3
"""Profile navigation to find bottlenecks"""

from pure_episodic_donut import PureEpisodicDonutNavigator
from pure_episodic_navigator import create_complex_maze
import time
import cProfile
import pstats

# Small maze for profiling
maze = create_complex_maze(5, seed=42)
nav = PureEpisodicDonutNavigator(maze)

# Profile navigation steps
def profile_navigation():
    for _ in range(10):
        nav.navigate_step()

# Basic timing
print("Timing individual operations:")

# 1. Action evaluation
start = time.time()
scores = nav.gedig_evaluate_actions()
print(f"Action evaluation: {time.time() - start:.4f}s")

# 2. Donut search
query = nav.episodes[0]['embedding']
start = time.time()
results = nav.donut_search(query)
print(f"Donut search: {time.time() - start:.4f}s")

# 3. Multi-hop episodes
start = time.time()
episodes = nav.get_n_hop_episodes_donut(nav.position, 3)
print(f"Multi-hop search: {time.time() - start:.4f}s")

# 4. Message passing
if episodes:
    start = time.time()
    updated = nav.deep_message_pass(episodes[:10])
    print(f"Message passing (10 episodes): {time.time() - start:.4f}s")

# Full profiling
print("\n\nDetailed profiling of 10 navigation steps:")
profiler = cProfile.Profile()
profiler.enable()
profile_navigation()
profiler.disable()

# Print stats
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
print("\nTop 10 time-consuming functions:")
stats.print_stats(10)