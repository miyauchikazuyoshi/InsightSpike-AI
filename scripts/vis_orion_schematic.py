
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def create_multihop_schematic():
    """
    Creates a schematic illustrating multi-hop connections.
    """
    G = nx.Graph()
    
    # Define nodes for local subgraphs and the path
    # Subgraph 1 (Start)
    G.add_edges_from([(1, 2), (1, 3), (2, 3)])
    # Subgraph 2 (End)
    G.add_edges_from([(8, 9), (8, 10), (9, 10)])
    
    # Connector path (Multi-hop)
    path_nodes = [3, 4, 5, 6, 7, 8]
    nx.add_path(G, path_nodes)
    
    # Positions
    pos = {
        1: (0, 1), 2: (0, -1), 3: (1, 0),
        4: (2, 0), 5: (3, 0), 6: (4, 0), 7: (5, 0),
        8: (6, 0), 9: (7, 1), 10: (7, -1)
    }
    
    plt.figure(figsize=(10, 4))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500)
    plt.title("Multi-hop Subgraph Connection Schematic")
    plt.savefig('docs/paper/jsai2026/option_ab_merged/figs/multihop_schematic.png')
    print("Generated multihop_schematic.png")

def create_orion_schematic():
    """
    Creates an 'Orion-style' schematic with star-like nodes and dark background.
    """
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='black')
    ax.set_facecolor('black')
    
    # Nodes with star positions (roughly)
    pos = {
        'Query': (0, 0),
        'Target': (4, 2),
        'S1': (1, 1), 'S2': (1, -1), # 1-hop
        'S3': (2.5, 1.5), 'S4': (2.5, 0), # 2-hop path
        'D1': (-2, 1), 'D2': (-2, -1) # Distractors
    }
    
    # Edges
    edges_core = [('Query', 'S1'), ('Query', 'S2')]
    edges_path = [('S1', 'S3'), ('S3', 'Target')]
    edges_noise = [('Query', 'D1'), ('Query', 'D2')]
    
    # Draw logic
    nx.draw_networkx_nodes(nx.Graph(), pos, nodelist=['Query'], node_color='gold', node_size=500, node_shape='*')
    nx.draw_networkx_nodes(nx.Graph(), pos, nodelist=['Target'], node_color='cyan', node_size=300, node_shape='o')
    nx.draw_networkx_nodes(nx.Graph(), pos, nodelist=['S1', 'S2', 'S3', 'S4'], node_color='white', node_size=100, alpha=0.8)
    nx.draw_networkx_nodes(nx.Graph(), pos, nodelist=['D1', 'D2'], node_color='gray', node_size=80, alpha=0.5)
    
    nx.draw_networkx_edges(nx.Graph(edges_core), pos, edge_color='gold', alpha=0.8, width=1.5)
    nx.draw_networkx_edges(nx.Graph(edges_path), pos, edge_color='cyan', style='dashed', alpha=0.6, width=1.5)
    nx.draw_networkx_edges(nx.Graph(edges_noise), pos, edge_color='gray', alpha=0.3)
    
    plt.title("Constraint-Satisfaction (Orion) Path", color='white')
    plt.axis('off')
    plt.savefig('docs/paper/jsai2026/option_ab_merged/figs/orion_schematic.png')
    print("Generated orion_schematic.png")



def create_academic_orion_radar():
    """
    Creates the 'Academic Orion Radar' schematic (v6).
    Adapts the 'Orion Radar' (organic layout + concentric logic) to a 
    White Background for paper compatibility (Print-friendly).
    """
    plt.style.use('default') 
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
    ax.set_facecolor('white')
    
    # 1. Draw Concentric Rings (Subtle guides)
    # Inner Orbit (0-hop)
    circle_0hop = plt.Circle((0, 0), 1.5, color='#B2EBF2', alpha=0.5, linestyle='-', linewidth=1.5, fill=False)
    ax.add_patch(circle_0hop)
    
    # Outer Orbit (Multi-hop)
    circle_mhop = plt.Circle((0, 0), 3.5, color='#FFCCBC', alpha=0.4, linestyle='-', linewidth=1.0, fill=False)
    ax.add_patch(circle_mhop)
    
    # 2. Define Positions (Same Organic Randomness logic)
    np.random.seed(42) # Keep same organic shape as dark version
    pos = {}
    
    # Center: Query
    query_node = 'Q'
    pos[query_node] = (0, 0)
    
    # Ring 1: 0-hop Neighbors
    neighbors = ['n1', 'n2', 'n3', 'n4', 'n5']
    radius_1 = 1.3
    for i, node in enumerate(neighbors):
        angle = (2 * np.pi / len(neighbors)) * i + np.random.uniform(-0.2, 0.2)
        r = radius_1 + np.random.uniform(-0.2, 0.2)
        pos[node] = (r * np.cos(angle), r * np.sin(angle))
        
    # Ring 2: Distant Clusters
    cluster_a_center = (-3.2, 0.5)
    cluster_b_center = (3.2, -0.8)
    
    cluster_a_nodes = ['A_hub', 'A1', 'A2', 'A3']
    pos['A_hub'] = cluster_a_center
    for n in ['A1', 'A2', 'A3']:
        pos[n] = (cluster_a_center[0] + np.random.uniform(-0.6, 0.6), 
                  cluster_a_center[1] + np.random.uniform(-0.6, 0.6))

    cluster_b_nodes = ['B_hub', 'B1', 'B2', 'B3']
    pos['B_hub'] = cluster_b_center
    for n in ['B1', 'B2', 'B3']:
        pos[n] = (cluster_b_center[0] + np.random.uniform(-0.6, 0.6), 
                  cluster_b_center[1] + np.random.uniform(-0.6, 0.6))

    # 3. Draw Background Edges (Grey for context)
    edges_bg = []
    for n in ['A1','A2','A3']: edges_bg.append(('A_hub', n))
    edges_bg.append(('A1', 'A2'))
    for n in ['B1','B2','B3']: edges_bg.append(('B_hub', n))
    edges_bg.append(('B2', 'B3'))
    
    nx.draw_networkx_edges(
        nx.Graph(edges_bg), pos, ax=ax, edge_color='#9E9E9E', width=1.0, alpha=0.6
    )

    # 4. Draw Metric Edges
    
    # 0-hop Edges (Teal/Blue - Structure Check)
    edges_0hop = [(query_node, n) for n in neighbors]
    nx.draw_networkx_edges(
        nx.Graph(edges_0hop), pos, ax=ax, 
        edge_color='#00838F', width=2.0, alpha=0.7, style=':'
    )
    
    # Multi-hop Bridges (Red - Insight)
    bridge_edges = [(neighbors[2], 'A_hub'), (neighbors[4], 'B_hub')]
    nx.draw_networkx_edges(
        nx.Graph(bridge_edges), pos, ax=ax, 
        edge_color='#C62828', width=3.0, alpha=1.0 # Strong Red
    )

    # 5. Draw Nodes
    # Query (Red Star)
    nx.draw_networkx_nodes(nx.Graph(), pos, nodelist=[query_node], node_color='#D32F2F', node_size=800, node_shape='*', label='Query', ax=ax)
    
    # Neighbors (Teal Circles)
    nx.draw_networkx_nodes(nx.Graph(), pos, nodelist=neighbors, node_color='white', edgecolors='#00838F', linewidths=2, node_size=200, node_shape='o', ax=ax)
    
    # Clusters (Grey Clouds)
    nx.draw_networkx_nodes(nx.Graph(), pos, nodelist=cluster_a_nodes + cluster_b_nodes, node_color='#EEEEEE', edgecolors='#9E9E9E', node_size=120, ax=ax)
    # Hubs bigger
    nx.draw_networkx_nodes(nx.Graph(), pos, nodelist=['A_hub', 'B_hub'], node_color='#E0E0E0', edgecolors='#616161', node_size=250, ax=ax)

    # 6. Annotations (Clean Academic)
    
    # Ring Labels
    ax.text(0, 1.8, "0-hop Orbit (Structure)", ha='center', va='bottom', color='#006064', fontsize=10, fontweight='bold')
    ax.text(0, 3.8, "Multi-hop Reach (Insight)", ha='center', va='bottom', color='#BF360C', fontsize=10, fontweight='bold')
    
    # Definitions
    # SP
    ax.text(3.5, -3.5, 
            r"$\bf{\Delta SP \gg 0}$" + "\n(Shortcut Found)", 
            ha='right', va='bottom', fontsize=12, color='#C62828', 
            bbox=dict(boxstyle="round", fc="white", ec="#C62828", alpha=1.0))
            
    # Entropy
    ax.text(-3.5, -3.5, 
            r"$\bf{\Delta IG \uparrow}$" + "\n(Uncertainty Drop)", 
            ha='left', va='bottom', fontsize=12, color='#006064', 
            bbox=dict(boxstyle="round", fc="white", ec="#006064", alpha=1.0))

    # Central Label
    ax.text(0, -0.6, "Query", ha='center', va='top', color='#D32F2F', fontsize=10, fontweight='bold')

    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-4.5, 4.5)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('docs/paper/jsai2026/option_ab_merged/figs/academic_orion_radar.png', dpi=300, bbox_inches='tight')
    print("Generated academic_orion_radar.png")

if __name__ == "__main__":
    create_academic_orion_radar()
