"""
geDIG概念図の作成 - グラフのbefore/afterを示す
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-GUI environments

# Font settings
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(14, 8))

# Left: Before (disconnected knowledge)
ax1 = plt.subplot(2, 3, 1)
G1 = nx.Graph()
# Three independent clusters
cluster1 = [(0, 0), (1, 0), (1, 1)]
cluster2 = [(3, 2), (4, 2), (4, 3)]
cluster3 = [(2, -2), (3, -2)]

pos1 = {}
nodes1 = []
for i, clusters in enumerate([cluster1, cluster2, cluster3]):
    for j, (x, y) in enumerate(clusters):
        node_id = f"c{i}_{j}"
        pos1[node_id] = (x, y)
        nodes1.append(node_id)
        G1.add_node(node_id)

# Edges within clusters
G1.add_edges_from([("c0_0", "c0_1"), ("c0_1", "c0_2")])
G1.add_edges_from([("c1_0", "c1_1"), ("c1_1", "c1_2")])
G1.add_edge("c2_0", "c2_1")

nx.draw_networkx_nodes(G1, pos1, node_color='lightcoral', node_size=500, ax=ax1)
nx.draw_networkx_edges(G1, pos1, width=2, alpha=0.5, ax=ax1)
ax1.set_title("Before: Disconnected Clusters", fontsize=12, fontweight='bold')
ax1.axis('off')

# Center: New Query
ax2 = plt.subplot(2, 3, 2)
ax2.text(0.5, 0.7, "New Query", ha='center', fontsize=14, fontweight='bold')
ax2.text(0.5, 0.3, "How are concepts\nA, B, C related?", ha='center', fontsize=11)
# Down arrow
arrow = FancyArrowPatch((0.5, 0.1), (0.5, -0.1),
                       mutation_scale=30, color='green', linewidth=2)
ax2.add_patch(arrow)
ax2.set_xlim(0, 1)
ax2.set_ylim(-0.2, 1)
ax2.axis('off')

# Right: After (integrated knowledge)
ax3 = plt.subplot(2, 3, 3)
G2 = nx.Graph()
# Integrated graph (connected by query node)
pos2 = dict(pos1)
pos2["query"] = (2, 0.5)  # Query node at center
G2.add_nodes_from(G1.nodes())
G2.add_edges_from(G1.edges())
G2.add_node("query")
# Query connects each cluster
G2.add_edges_from([("query", "c0_1"), ("query", "c1_0"), ("query", "c2_0")])

# Draw nodes (query in different color)
node_colors = ['gold' if n == "query" else 'lightblue' for n in G2.nodes()]
nx.draw_networkx_nodes(G2, pos2, node_color=node_colors, node_size=500, ax=ax3)
nx.draw_networkx_edges(G2, pos2, width=2, alpha=0.5, ax=ax3)
# Highlight edges from query
query_edges = [("query", "c0_1"), ("query", "c1_0"), ("query", "c2_0")]
nx.draw_networkx_edges(G2, pos2, edgelist=query_edges, width=3, 
                      edge_color='green', style='--', ax=ax3)
ax3.set_title("After: Integrated Knowledge Graph", fontsize=12, fontweight='bold')
ax3.axis('off')

# Bottom: Metrics explanation
ax4 = plt.subplot(2, 3, 4)
ax4.text(0.5, 0.8, "EPC (Edit-Path Cost)", fontsize=11, fontweight='bold', ha='center')
# Use bullet point character
ax4.text(0.5, 0.5, "• Nodes: 8 → 9 (+1)", ha='center')
ax4.text(0.5, 0.3, "• Edges: 5 → 8 (+3)", ha='center')
ax4.text(0.5, 0.1, "Structural Integration", ha='center', color='blue', fontweight='bold')
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.axis('off')

ax5 = plt.subplot(2, 3, 5)
ax5.text(0.5, 0.8, "IG (Information Gain)", fontsize=11, fontweight='bold', ha='center')
ax5.text(0.5, 0.5, "• Entropy Variance: High → Low", ha='center')
ax5.text(0.5, 0.3, "• Inter-cluster Information Flow", ha='center')
ax5.text(0.5, 0.1, "Information Organization", ha='center', color='red', fontweight='bold')
ax5.set_xlim(0, 1)
ax5.set_ylim(0, 1)
ax5.axis('off')

ax6 = plt.subplot(2, 3, 6)
# Use math mode for Delta symbol
ax6.text(0.5, 0.8, r"geDIG = $-\Delta$GED $-$ $\Delta$IG", fontsize=11, fontweight='bold', ha='center')
ax6.text(0.5, 0.5, "Structure + Information", ha='center')
ax6.text(0.5, 0.3, "= Insight Moment", ha='center', fontsize=12, fontweight='bold', color='purple')
# Threshold line
ax6.axhline(y=0.15, color='green', linestyle='--', linewidth=1)
ax6.text(0.8, 0.17, r"$\theta$ = -0.5", fontsize=9)
ax6.set_xlim(0, 1)
ax6.set_ylim(0, 1)
ax6.axis('off')

plt.suptitle("geDIG Framework: Insight Detection through Graph Structure Changes", 
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('figures/fig5_concept_new.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/fig5_concept_new.png', dpi=300, bbox_inches='tight')

print("Generated new concept figure:")
print("- figures/fig5_concept_new.pdf")
print("- figures/fig5_concept_new.png")