#!/usr/bin/env python3
"""Test with clean start and debug multi-hop"""

from pure_episodic_donut import PureEpisodicDonutNavigator
from pure_episodic_navigator import create_complex_maze

# Create new instance (should be clean)
maze = create_complex_maze(5, seed=42)
nav = PureEpisodicDonutNavigator(maze)

print("Initial state check:")
print(f"Episodes: {len(nav.episodes)}")
print(f"Graph nodes: {nav.episode_graph.number_of_nodes()}")
print(f"Graph edges: {nav.episode_graph.number_of_edges()}")
print(f"Visit counts: {len(nav.visit_counts)}")

# Check multi-hop functionality
print("\n\nMulti-hop test from (2,1):")
for n_hops in [1, 2, 3]:
    episodes = nav.get_n_hop_episodes_donut((2, 1), n_hops)
    print(f"{n_hops}-hop: found {len(episodes)} episodes")
    
    # Show episode distribution by hop
    hop_counts = {}
    for ep in episodes:
        hop = ep.get('hop', 0)
        hop_counts[hop] = hop_counts.get(hop, 0) + 1
    print(f"  Distribution: {hop_counts}")

# Navigate a few steps
print("\n\nNavigating 5 steps...")
for i in range(5):
    nav.navigate_step()
    print(f"Step {i+1}: position={nav.position}")

# Check multi-hop again
print("\n\nMulti-hop test after navigation:")
for n_hops in [1, 2, 3]:
    episodes = nav.get_n_hop_episodes_donut(nav.position, n_hops)
    print(f"{n_hops}-hop: found {len(episodes)} episodes")
    
# Check graph connectivity
print(f"\n\nGraph analysis:")
print(f"Nodes: {nav.episode_graph.number_of_nodes()}")
print(f"Edges: {nav.episode_graph.number_of_edges()}")
print(f"Average degree: {2 * nav.episode_graph.number_of_edges() / nav.episode_graph.number_of_nodes():.2f}")

# Check if multi-hop is actually being used
import networkx as nx
if nav.episode_graph.number_of_nodes() > 5:
    # Get connected components
    components = list(nx.connected_components(nav.episode_graph))
    print(f"Connected components: {len(components)}")
    print(f"Largest component size: {max(len(c) for c in components)}")