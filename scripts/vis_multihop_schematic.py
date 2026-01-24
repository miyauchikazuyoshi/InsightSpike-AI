
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def create_multihop_schematic(output_path):
    """
    Creates a schematic visualization of subgraphs connecting via a multi-hop path.
    """
    G = nx.Graph()
    
    # Subgraph 1 (Local Context A)
    cluster_a = range(5)
    G.add_edges_from([(0,1), (0,2), (1,3), (2,4), (1,2)])
    pos = {
        0: (0, 0), 1: (0.5, 0.8), 2: (0.5, -0.8), 
        3: (1.2, 1.2), 4: (1.2, -1.2)
    }
    
    # Subgraph 2 (Local Context B)
    cluster_b = range(5, 10)
    offset_x = 4.0
    G.add_edges_from([(5,6), (5,7), (6,8), (7,9), (6,7)])
    pos.update({
        5: (offset_x, 0), 6: (offset_x-0.5, 0.8), 7: (offset_x-0.5, -0.8),
        8: (offset_x-1.2, 1.2), 9: (offset_x-1.2, -1.2)
    })
    
    # Multi-hop Path (The Bridge/Insight)
    # Connecting node 0 (center of A) to node 5 (center of B) via intermediate nodes
    path_nodes = [10, 11]
    G.add_edges_from([(0, 10), (10, 11), (11, 5)])
    pos.update({
        10: (1.3, 0), 11: (2.7, 0)
    })
    
    plt.figure(figsize=(10, 4))
    
    # Draw Subgraphs (Clusters)
    nx.draw_networkx_nodes(G, pos, nodelist=cluster_a, node_color='#aaccff', node_size=500, label='Subgraph A')
    nx.draw_networkx_nodes(G, pos, nodelist=cluster_b, node_color='#ffccaa', node_size=500, label='Subgraph B')
    nx.draw_networkx_edges(G, pos, edgelist=[(u,v) for u,v in G.edges() if u in cluster_a and v in cluster_a], edge_color='#88aaee', width=2)
    nx.draw_networkx_edges(G, pos, edgelist=[(u,v) for u,v in G.edges() if u in cluster_b and v in cluster_b], edge_color='#eeaa88', width=2)
    
    # Draw Multi-hop Path (The "Insight")
    nx.draw_networkx_nodes(G, pos, nodelist=path_nodes, node_color='#dddddd', node_size=300, node_shape='s')
    nx.draw_networkx_edges(G, pos, edgelist=[(0,10), (10,11), (11,5)], edge_color='#ff5555', width=3, style='dashed')
    
    # Annotations
    plt.text(0, -1.8, "Local Subgraph A\n(Known Context)", ha='center', fontsize=12, fontweight='bold', color='#4466aa')
    plt.text(4.0, -1.8, "Local Subgraph B\n(Target Context)", ha='center', fontsize=12, fontweight='bold', color='#aa6644')
    plt.text(2.0, 0.3, "Multi-hop Inference\n(Insight / Abduction)", ha='center', fontsize=10, color='#cc3333', backgroundcolor='white')
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Graph saved to {output_path}")

if __name__ == "__main__":
    output_file = "/Users/miyauchikazuyoshi/Documents/GitHub/InsightSpike-AI/docs/paper/jsai2026/option_ab_merged/figs/multihop_schematic.png"
    create_multihop_schematic(output_file)
