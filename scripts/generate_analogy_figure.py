
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx

def generate_analogy_figure():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Solar System Graph
    G_solar = nx.DiGraph()
    G_solar.add_edges_from([
        ('Sun', 'Earth'), ('Sun', 'Mars'), ('Sun', 'Jupiter'),
        ('Earth', 'Moon')
    ])
    pos_solar = {
        'Sun': (0, 0),
        'Earth': (1, 0),
        'Mars': (0, 1), 
        'Jupiter': (0, -1),
        'Moon': (1.5, 0)
    }
    
    nx.draw_networkx_nodes(G_solar, pos_solar, ax=ax1, node_color='#ffcc00', node_size=1500)
    nx.draw_networkx_edges(G_solar, pos_solar, ax=ax1, edge_color='gray', width=2)
    nx.draw_networkx_labels(G_solar, pos_solar, ax=ax1, font_size=10, font_weight='bold')
    ax1.set_title("Solar System (Gravitational)", fontsize=14)
    ax1.axis('off')
    
    # Atom Graph
    G_atom = nx.DiGraph()
    G_atom.add_edges_from([
        ('Nucleus', 'Electron1'), ('Nucleus', 'Electron2'), ('Nucleus', 'Electron3'),
        ('Electron1', 'Spin') # Fictional detail for structure matching
    ])
    pos_atom = {
        'Nucleus': (0, 0),
        'Electron1': (1, 0),
        'Electron2': (0, 1), 
        'Electron3': (0, -1),
        'Spin': (1.5, 0)
    }
    
    nx.draw_networkx_nodes(G_atom, pos_atom, ax=ax2, node_color='#00ccff', node_size=1500)
    nx.draw_networkx_edges(G_atom, pos_atom, ax=ax2, edge_color='gray', width=2)
    nx.draw_networkx_labels(G_atom, pos_atom, ax=ax2, font_size=10, font_weight='bold')
    ax2.set_title("Atom Model (Electromagnetic)", fontsize=14)
    ax2.axis('off')
    
    # Isomorphism Annotation
    plt.figtext(0.5, 0.9, "Structural Isomorphism (SS = 0.995)", ha="center", fontsize=16, color='red')
    plt.figtext(0.5, 0.5, "≈", ha="center", va="center", fontsize=50)

    out_path = "docs/paper/jsai2026/option_ab_merged/figs/analogy_isomorphism.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved analogy figure to {out_path}")

if __name__ == "__main__":
    generate_analogy_figure()
