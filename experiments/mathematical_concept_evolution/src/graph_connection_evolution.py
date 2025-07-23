#!/usr/bin/env python3
"""
Create knowledge graph connection evolution visualization
Shows how mathematical concepts connect and split over learning phases
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.patches as mpatches

def create_graph_connection_evolution():
    """Create a visualization showing evolution of concept connections"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Mathematical Concept Graph Evolution', fontsize=18, fontweight='bold')
    
    # Define node colors
    colors = {
        'phase1': '#FFD700',      # Gold
        'phase2': '#87CEEB',      # SkyBlue  
        'phase3': '#FF69B4',      # HotPink
        'split': '#FF0000',       # Red
        'query': '#00FF00',       # Green
        'spike': '#FF4500'        # OrangeRed
    }
    
    # Phase 1: Elementary Foundation
    ax1.set_title('Phase 1: Elementary Foundation', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    G1 = nx.Graph()
    pos1 = {
        'number': (0, 0),
        'addition': (-1, -1),
        'subtraction': (1, -1),
        'multiplication': (-1, 1),
        'division': (1, 1)
    }
    
    G1.add_nodes_from(pos1.keys())
    G1.add_edges_from([
        ('number', 'addition'),
        ('number', 'subtraction'),
        ('number', 'multiplication'),
        ('number', 'division'),
        ('addition', 'subtraction'),
        ('multiplication', 'division')
    ])
    
    nx.draw_networkx_nodes(G1, pos1, node_color=colors['phase1'], 
                          node_size=1000, edgecolors='black', linewidths=2, ax=ax1)
    nx.draw_networkx_labels(G1, pos1, font_size=10, font_weight='bold', ax=ax1)
    nx.draw_networkx_edges(G1, pos1, width=2, alpha=0.6, ax=ax1)
    
    ax1.text(0, -2, 'Base Knowledge Graph\n5 nodes, 6 edges', 
             ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", 
             facecolor='lightgray', alpha=0.7))
    
    # Phase 2: Middle School - Adding Complexity
    ax2.set_title('Phase 2: Relational Concepts Added', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    G2 = nx.Graph()
    # Expand positions
    pos2 = {
        # Original nodes
        'number': (0, 0),
        'addition': (-1.5, -1),
        'subtraction': (1.5, -1),
        'multiplication': (-1.5, 1),
        'division': (1.5, 1),
        # New nodes
        'negative': (-2.5, 0),
        'fraction': (2.5, 0),
        'function': (0, 2),
        'algebraic': (0, -2)
    }
    
    # Color mapping
    node_colors = []
    for node in pos2.keys():
        if node in ['negative', 'fraction', 'function', 'algebraic']:
            node_colors.append(colors['phase2'])
        else:
            node_colors.append(colors['phase1'])
    
    G2.add_nodes_from(pos2.keys())
    # Original edges
    G2.add_edges_from([
        ('number', 'addition'), ('number', 'subtraction'),
        ('number', 'multiplication'), ('number', 'division'),
        ('addition', 'subtraction'), ('multiplication', 'division')
    ])
    # New edges
    G2.add_edges_from([
        ('negative', 'subtraction'), ('negative', 'number'),
        ('fraction', 'division'), ('fraction', 'number'),
        ('function', 'number'), ('function', 'algebraic'),
        ('algebraic', 'addition'), ('algebraic', 'multiplication')
    ])
    
    nx.draw_networkx_nodes(G2, pos2, node_color=node_colors,
                          node_size=800, edgecolors='black', linewidths=2, ax=ax2)
    nx.draw_networkx_labels(G2, pos2, font_size=9, font_weight='bold', ax=ax2)
    nx.draw_networkx_edges(G2, pos2, width=2, alpha=0.6, ax=ax2)
    
    ax2.text(0, -2.8, 'Query: "What is the fundamental nature of reality?..."', 
             ha='center', fontsize=10, style='italic')
    
    # Phase 3: Concept Conflicts and Splits
    ax3.set_title('Phase 3: Conceptual Conflicts Detected', fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    G3 = nx.Graph()
    # Show splitting nodes
    pos3 = {
        # Non-split nodes
        'addition': (-1.5, -1),
        'subtraction': (1.5, -1),
        'division': (1.5, 1),
        'negative': (-2.5, 0),
        'fraction': (2.5, 0),
        'algebraic': (0, -2),
        # Nodes that will split
        'number!': (0, 0),
        'multiplication!': (-1.5, 1),
        'function!': (0, 2)
    }
    
    # Color mapping with conflict highlighting
    node_colors = []
    for node in pos3.keys():
        if '!' in node:
            node_colors.append(colors['split'])
        elif node in ['negative', 'fraction', 'algebraic']:
            node_colors.append(colors['phase2'])
        else:
            node_colors.append(colors['phase1'])
    
    G3.add_nodes_from(pos3.keys())
    # Show stressed connections
    G3.add_edges_from([
        ('number!', 'addition'), ('number!', 'subtraction'),
        ('number!', 'multiplication!'), ('number!', 'division'),
        ('addition', 'subtraction'), ('multiplication!', 'division'),
        ('negative', 'subtraction'), ('negative', 'number!'),
        ('fraction', 'division'), ('fraction', 'number!'),
        ('function!', 'number!'), ('function!', 'algebraic'),
        ('algebraic', 'addition'), ('algebraic', 'multiplication!')
    ])
    
    # Draw with special effects for conflicted nodes
    nx.draw_networkx_nodes(G3, pos3, node_color=node_colors,
                          node_size=800, edgecolors='black', linewidths=2, ax=ax3)
    
    # Labels without exclamation marks
    labels3 = {k: k.replace('!', '') for k in pos3.keys()}
    nx.draw_networkx_labels(G3, pos3, labels3, font_size=9, font_weight='bold', ax=ax3)
    
    # Draw edges with different styles
    normal_edges = [(u, v) for u, v in G3.edges() if '!' not in u and '!' not in v]
    conflict_edges = [(u, v) for u, v in G3.edges() if '!' in u or '!' in v]
    
    nx.draw_networkx_edges(G3, pos3, edgelist=normal_edges, width=2, alpha=0.6, ax=ax3)
    nx.draw_networkx_edges(G3, pos3, edgelist=conflict_edges, width=3, alpha=0.8, 
                          edge_color='red', style='--', ax=ax3)
    
    # Add conflict annotations
    ax3.text(0, 0.3, 'CONFLICT!', ha='center', va='bottom', color='red', 
             fontweight='bold', fontsize=8)
    ax3.text(-1.5, 1.3, 'CONFLICT!', ha='center', va='bottom', color='red', 
             fontweight='bold', fontsize=8)
    ax3.text(0, 2.3, 'CONFLICT!', ha='center', va='bottom', color='red', 
             fontweight='bold', fontsize=8)
    
    # Phase 4: After Splits - Final State
    ax4.set_title('Phase 4: After Memory Splits', fontsize=14, fontweight='bold')
    ax4.axis('off')
    
    G4 = nx.DiGraph()  # Directed graph to show relationships
    
    # Final positions with split nodes
    pos4 = {
        # Non-split nodes
        'addition': (-2, -1.5),
        'subtraction': (2, -1.5),
        'division': (2, 1),
        'negative': (-3, 0),
        'fraction': (3, 0),
        'algebraic': (0, -2.5),
        # Split nodes
        'num_basic': (-0.5, 0.5),
        'num_adv': (0.5, 0.5),
        'mult_basic': (-2.5, 1.5),
        'mult_adv': (-1.5, 1.5),
        'func_basic': (-0.5, 2.5),
        'func_adv': (0.5, 2.5)
    }
    
    # Color mapping for final state
    node_colors4 = []
    node_sizes4 = []
    for node in pos4.keys():
        if '_basic' in node:
            node_colors4.append(colors['phase1'])
            node_sizes4.append(700)
        elif '_adv' in node:
            node_colors4.append(colors['phase3'])
            node_sizes4.append(700)
        elif node in ['negative', 'fraction', 'algebraic']:
            node_colors4.append(colors['phase2'])
            node_sizes4.append(800)
        else:
            node_colors4.append(colors['phase1'])
            node_sizes4.append(800)
    
    G4.add_nodes_from(pos4.keys())
    
    # Add directed edges showing relationships
    G4.add_edges_from([
        # Basic number connections
        ('num_basic', 'addition'), ('num_basic', 'subtraction'),
        # Advanced number connections
        ('num_adv', 'negative'), ('num_adv', 'fraction'), ('num_adv', 'algebraic'),
        # Multiplication splits
        ('mult_basic', 'addition'), ('mult_adv', 'algebraic'),
        # Function splits
        ('func_basic', 'num_basic'), ('func_adv', 'algebraic'),
        # Regular connections
        ('addition', 'subtraction'), ('division', 'fraction'),
        ('negative', 'subtraction'), ('algebraic', 'func_adv')
    ])
    
    nx.draw_networkx_nodes(G4, pos4, node_color=node_colors4, node_size=node_sizes4,
                          edgecolors='black', linewidths=2, ax=ax4)
    
    # Custom labels
    labels4 = {
        'num_basic': 'number\n(basic)',
        'num_adv': 'number\n(adv)',
        'mult_basic': 'mult\n(basic)',
        'mult_adv': 'mult\n(adv)',
        'func_basic': 'func\n(basic)',
        'func_adv': 'func\n(adv)'
    }
    for node in pos4.keys():
        if node not in labels4:
            labels4[node] = node
    
    nx.draw_networkx_labels(G4, pos4, labels4, font_size=8, font_weight='bold', ax=ax4)
    nx.draw_networkx_edges(G4, pos4, width=2, alpha=0.6, arrows=True, 
                          arrowsize=15, ax=ax4)
    
    # Add spike detection annotation
    spike_bbox = FancyBboxPatch((1.5, 2.3), 1, 0.4, boxstyle="round,pad=0.1",
                               facecolor=colors['spike'], edgecolor='black', linewidth=2)
    ax4.add_patch(spike_bbox)
    ax4.text(2, 2.5, 'SPIKE!', ha='center', va='center', color='white', 
             fontweight='bold', fontsize=10)
    
    # Legend
    legend_elements = [
        mpatches.Patch(color=colors['phase1'], label='Phase 1: Elementary'),
        mpatches.Patch(color=colors['phase2'], label='Phase 2: Relational'),
        mpatches.Patch(color=colors['phase3'], label='Phase 3: Advanced'),
        mpatches.Patch(color=colors['split'], label='Conflict Node'),
        mpatches.Patch(color=colors['spike'], label='Spike Detected')
    ]
    ax4.legend(handles=legend_elements, loc='lower left', fontsize=8)
    
    # Summary text
    ax4.text(0, -3.2, 'Final: 12 nodes (6 split), Complex hierarchical structure', 
             ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", 
             facecolor='lightgray', alpha=0.7))
    
    plt.tight_layout()
    
    # Save
    from pathlib import Path
    output_file = Path(__file__).parent.parent / "graph_connection_evolution.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n📊 Graph connection evolution saved to: {output_file}")

if __name__ == "__main__":
    create_graph_connection_evolution()