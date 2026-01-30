
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def create_insight_visualization():
    # Setup the figure - 3 panels
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))
    fig.patch.set_facecolor('white')  # White background
    
    # Common style settings
    node_color_Cluster1 = '#38bdf8'  # Light Blue (Newtonian)
    node_color_Cluster2 = '#f472b6'  # Pink (Electromagnetism)
    node_color_Insight = '#fbbf24'   # Amber (Insight/Spark)
    edge_color_default = '#64748b'   # Slate 500 (darker for white bg)
    edge_color_insight = '#fbbf24'   # Amber
    text_color = '#1e293b'           # Slate 800 (Dark text)

    # Layout for G1 (Common)
    pos = {}
    # Left Cluster - Newtonian
    pos["Absolute(T)"] = np.array([-0.8, 0.5])
    pos["Galilean"] = np.array([-0.6, 0.2])
    pos["Mass (m)"] = np.array([-0.8, -0.3])
    pos["Energy (E)"] = np.array([-0.5, -0.5])
    pos["Force (F)"] = np.array([-0.6, 0.0])
    pos["Accel (a)"] = np.array([-0.4, -0.2])
    
    # Right Cluster - Electromagnetism
    pos["c=const"] = np.array([0.8, 0.5])
    pos["Light Wave"] = np.array([0.6, 0.2])
    pos["Maxwell"] = np.array([0.5, -0.1])
    pos["E-Field"] = np.array([0.7, -0.4])
    pos["B-Field"] = np.array([0.9, -0.2])

    # Insight Node (Center)
    pos["INSIGHT"] = np.array([0.0, 0.1])

    # --- Panel 1: The Paradox (disconnected) ---
    G1 = nx.Graph()
    cluster1_nodes = ["Mass (m)", "Force (F)", "Accel (a)", "Absolute(T)", "Galilean", "Energy (E)"]
    G1.add_nodes_from(cluster1_nodes)
    G1.add_edges_from([
        ("Mass (m)", "Force (F)"), ("Mass (m)", "Accel (a)"), ("Force (F)", "Accel (a)"),
        ("Absolute(T)", "Galilean"), ("Galilean", "Accel (a)"),
        ("Force (F)", "Energy (E)"), ("Accel (a)", "Energy (E)")
    ])
    
    cluster2_nodes = ["E-Field", "B-Field", "Maxwell", "Light Wave", "c=const"]
    G1.add_nodes_from(cluster2_nodes)
    G1.add_edges_from([
        ("E-Field", "B-Field"), ("E-Field", "Maxwell"),
        ("B-Field", "Maxwell"), ("Maxwell", "Light Wave"),
        ("Light Wave", "c=const")
    ])

    ax1.set_title("1. The Paradox (Contradiction)", color=text_color, fontsize=16, pad=20, fontweight='bold')
    nx.draw_networkx_nodes(G1, pos, nodelist=cluster1_nodes, node_color=node_color_Cluster1, node_size=1500, alpha=0.9, ax=ax1, edgecolors='black', linewidths=1)
    nx.draw_networkx_nodes(G1, pos, nodelist=cluster2_nodes, node_color=node_color_Cluster2, node_size=1500, alpha=0.9, ax=ax1, edgecolors='black', linewidths=1)
    nx.draw_networkx_edges(G1, pos, edge_color=edge_color_default, width=2, alpha=0.6, ax=ax1)
    nx.draw_networkx_labels(G1, pos, font_size=9, font_color='black', font_weight='bold', ax=ax1)
    
    # Gap Annotation
    ax1.annotate("GAP / CONTRADICTION\n(c is const vs relative)", 
                 xy=(0, 0.35), xytext=(0, 0.35), 
                 ha='center', va='center', color='white', fontsize=11, fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.5", fc="#ef4444", ec="none", alpha=0.8))
    
    ax1.text(0, -0.8, "High Structural Tension\n($F \\gg 0$)", ha='center', color=text_color, fontsize=12, style='italic')
    ax1.set_axis_off()

    # --- Panel 2: The Insight (Reconstruction) ---
    G2 = G1.copy()
    G2.add_node("INSIGHT", label="Constancy of\nLight Speed")
    
    # The reconstruction edges (The "Space-Time" fix)
    insight_edges = [
        ("INSIGHT", "Absolute(T)"),  # Redefines Time
        ("INSIGHT", "c=const"),        # Matches Electromagnetism
        ("INSIGHT", "Galilean")     # Modifies to Lorentz
    ]
    G2.add_edges_from(insight_edges)

    ax2.set_title("2. The Insight (Topological Reconstruction)", color=text_color, fontsize=16, pad=20, fontweight='bold')
    
    # Shift nodes slightly towards center in Panel 2
    pos2 = pos.copy()
    for n in cluster1_nodes: pos2[n] = pos2[n] * 0.9 + np.array([0.05, 0])
    for n in cluster2_nodes: pos2[n] = pos2[n] * 0.9 + np.array([-0.05, 0])
    pos2["INSIGHT"] = np.array([0.0, 0.2])

    nx.draw_networkx_nodes(G2, pos2, nodelist=cluster1_nodes, node_color=node_color_Cluster1, node_size=1500, alpha=0.6, ax=ax2, edgecolors='gray', linewidths=1)
    nx.draw_networkx_nodes(G2, pos2, nodelist=cluster2_nodes, node_color=node_color_Cluster2, node_size=1500, alpha=0.6, ax=ax2, edgecolors='gray', linewidths=1)
    nx.draw_networkx_nodes(G2, pos2, nodelist=["INSIGHT"], node_color=node_color_Insight, node_size=3000, alpha=1.0, ax=ax2, edgecolors='black', linewidths=2)

    nx.draw_networkx_edges(G2, pos2, edgelist=G1.edges(), edge_color=edge_color_default, width=2, alpha=0.3, ax=ax2)
    nx.draw_networkx_edges(G2, pos2, edgelist=insight_edges, edge_color=edge_color_insight, style='dashed', width=3, alpha=0.9, ax=ax2)
    
    nx.draw_networkx_labels(G2, pos2, font_size=9, font_color='black', font_weight='bold', ax=ax2)
    
    ax2.text(0, -0.8, "New Axiom Connects Clusters\n(Cost Reduced)", ha='center', color=text_color, fontsize=12, style='italic')
    ax2.set_axis_off()

    # --- Panel 3: The Discovery (E=mc^2) ---
    G3 = G2.copy()
    
    # The consequential edges (The "Mass-Energy" realization)
    discovery_edges = [
        ("INSIGHT", "Mass (m)"),
        ("INSIGHT", "Energy (E)")
    ]
    G3.add_edges_from(discovery_edges)
    
    # The direct link (E=mc^2)
    emc2_edge = ("Mass (m)", "Energy (E)")
    G3.add_edge(*emc2_edge)

    ax3.set_title("3. The Discovery ($E=mc^2$)", color=text_color, fontsize=16, pad=20, fontweight='bold')
    
    # Even tighter layout in Panel 3
    pos3 = pos2.copy()
    pos3["Mass (m)"] = np.array([-0.3, -0.2])   # Pull Mass to center
    pos3["Energy (E)"] = np.array([0.3, -0.2])  # Pull Energy to center

    nx.draw_networkx_nodes(G3, pos3, nodelist=cluster1_nodes, node_color=node_color_Cluster1, node_size=1500, alpha=0.5, ax=ax3, edgecolors='gray', linewidths=1)
    nx.draw_networkx_nodes(G3, pos3, nodelist=cluster2_nodes, node_color=node_color_Cluster2, node_size=1500, alpha=0.5, ax=ax3, edgecolors='gray', linewidths=1)
    nx.draw_networkx_nodes(G3, pos3, nodelist=["INSIGHT"], node_color=node_color_Insight, node_size=3000, alpha=1.0, ax=ax3, edgecolors='black', linewidths=2)

    # Background edges
    bg_edges = [e for e in G3.edges() if e != emc2_edge and e not in discovery_edges]
    nx.draw_networkx_edges(G3, pos3, edgelist=bg_edges, edge_color=edge_color_default, width=1, alpha=0.2, ax=ax3)
    
    # Path highlight
    path_edges = discovery_edges
    nx.draw_networkx_edges(G3, pos3, edgelist=path_edges, edge_color='#f43f5e', style='dotted', width=2, alpha=0.8, ax=ax3)
    
    # The Big Result
    nx.draw_networkx_edges(G3, pos3, edgelist=[emc2_edge], edge_color='#f43f5e', width=5, alpha=1.0, ax=ax3)
    
    nx.draw_networkx_labels(G3, pos3, font_size=9, font_color='black', font_weight='bold', ax=ax3)

    # Label E=mc2
    edge_center = (pos3["Mass (m)"] + pos3["Energy (E)"]) / 2
    ax3.text(edge_center[0], edge_center[1]-0.15, r"$E=mc^2$", 
             ha='center', va='top', fontsize=22, color='#f43f5e', fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='#f43f5e', boxstyle='round,pad=0.3', linewidth=2))

    ax3.text(0, -0.8, "Topological Path Found!\n(Information Gain Maximized)", ha='center', color=text_color, fontsize=12, style='italic')
    ax3.set_axis_off()

    # Title for whole figure
    plt.suptitle("The Origin of Insight: 3 Stages of Discovery", 
                 fontsize=22, color=text_color, y=0.98, fontweight='bold')

    output_path = "docs/research/vis_insight_topology_3step.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    create_insight_visualization()
