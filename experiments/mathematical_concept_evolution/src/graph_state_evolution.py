#!/usr/bin/env python3
"""
Visualize the evolution of graph states showing splits
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import networkx as nx
import numpy as np


def create_graph_state_evolution():
    """Create visualization showing how the knowledge graph evolves with splits"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Knowledge Graph Evolution: Episodic Memory Splits', fontsize=16, fontweight='bold')
    
    # Colors
    colors = {
        'phase1': '#FFE4B5',  # Moccasin
        'phase2': '#98FB98',  # PaleGreen
        'phase3': '#DDA0DD',  # Plum
        'split': '#FF6347',   # Tomato
        'normal': '#87CEEB'   # SkyBlue
    }
    
    # --- Phase 1: Initial State ---
    ax1 = axes[0, 0]
    ax1.set_title('Phase 1: Elementary Concepts', fontsize=12, fontweight='bold')
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.axis('off')
    
    # Create simple graph
    G1 = nx.Graph()
    nodes_phase1 = ['number', 'addition', 'subtraction', 'multiplication', 'division']
    positions1 = {
        'number': (0, 1),
        'addition': (-1, 0),
        'subtraction': (1, 0),
        'multiplication': (-0.5, -1),
        'division': (0.5, -1)
    }
    
    G1.add_nodes_from(nodes_phase1)
    G1.add_edges_from([
        ('number', 'addition'),
        ('number', 'subtraction'),
        ('addition', 'multiplication'),
        ('subtraction', 'division'),
        ('multiplication', 'division')
    ])
    
    # Draw graph
    nx.draw_networkx_nodes(G1, positions1, ax=ax1, node_color=colors['phase1'], 
                          node_size=1500, edgecolors='black', linewidths=2)
    nx.draw_networkx_labels(G1, positions1, ax=ax1, font_size=9)
    nx.draw_networkx_edges(G1, positions1, ax=ax1, width=2, alpha=0.6)
    
    ax1.text(0, -1.8, '5 nodes, 5 edges', ha='center', fontweight='bold')
    
    # --- Phase 2: After Adding New Concepts ---
    ax2 = axes[0, 1]
    ax2.set_title('Phase 2: Middle School Concepts Added', fontsize=12, fontweight='bold')
    ax2.set_xlim(-2.5, 2.5)
    ax2.set_ylim(-2.5, 2.5)
    ax2.axis('off')
    
    # Expanded graph
    G2 = nx.Graph()
    nodes_phase2_old = nodes_phase1
    nodes_phase2_new = ['negative_numbers', 'fraction_ratio', 'function_basic', 'algebraic']
    
    positions2 = {
        'number': (0, 1.5),
        'addition': (-1, 0.5),
        'subtraction': (1, 0.5),
        'multiplication': (-0.5, -0.5),
        'division': (0.5, -0.5),
        'negative_numbers': (-2, 0),
        'fraction_ratio': (2, 0),
        'function_basic': (0, 0),
        'algebraic': (0, -1.5)
    }
    
    G2.add_nodes_from(nodes_phase2_old + nodes_phase2_new)
    G2.add_edges_from([
        ('number', 'addition'), ('number', 'subtraction'),
        ('addition', 'multiplication'), ('subtraction', 'division'),
        ('multiplication', 'division'), ('negative_numbers', 'subtraction'),
        ('fraction_ratio', 'division'), ('function_basic', 'algebraic'),
        ('algebraic', 'multiplication'), ('function_basic', 'number')
    ])
    
    # Draw with different colors
    nx.draw_networkx_nodes(G2, positions2, nodelist=nodes_phase2_old, ax=ax2, 
                          node_color=colors['phase1'], node_size=1200, 
                          edgecolors='black', linewidths=2)
    nx.draw_networkx_nodes(G2, positions2, nodelist=nodes_phase2_new, ax=ax2, 
                          node_color=colors['phase2'], node_size=1200, 
                          edgecolors='black', linewidths=2)
    nx.draw_networkx_labels(G2, positions2, ax=ax2, font_size=8)
    nx.draw_networkx_edges(G2, positions2, ax=ax2, width=2, alpha=0.6)
    
    ax2.text(0, -2.3, '9 nodes, 10 edges', ha='center', fontweight='bold')
    
    # --- Phase 3: Conflicts Trigger Splits ---
    ax3 = axes[1, 0]
    ax3.set_title('Phase 3: Conceptual Conflicts → Memory Splits', fontsize=12, fontweight='bold')
    ax3.set_xlim(-3, 3)
    ax3.set_ylim(-3, 3)
    ax3.axis('off')
    
    # Show splits happening
    split_nodes = {
        'multiplication': {'basic': (-1.5, -1), 'advanced': (-0.5, -1.5)},
        'function_basic': {'basic': (0.5, 0.5), 'advanced': (1.5, 0)},
        'number': {'basic': (-0.5, 2), 'advanced': (0.5, 2)}
    }
    
    # Draw original nodes that will split
    for node, splits in split_nodes.items():
        if node in positions2:
            # Original node with warning
            circle = Circle(positions2[node], 0.3, color=colors['split'], alpha=0.3)
            ax3.add_patch(circle)
            ax3.text(positions2[node][0], positions2[node][1] + 0.5, 
                    'CONFLICT!', ha='center', fontsize=8, color='red', fontweight='bold')
    
    # Draw the split process
    for node, splits in split_nodes.items():
        if node in positions2:
            orig_pos = positions2[node]
            for split_type, new_pos in splits.items():
                # Arrow from original to split
                arrow = FancyArrowPatch(orig_pos, new_pos,
                                      connectionstyle="arc3,rad=0.3",
                                      arrowstyle='->',
                                      mutation_scale=20,
                                      color=colors['split'],
                                      linewidth=2,
                                      linestyle='--')
                ax3.add_patch(arrow)
    
    ax3.text(0, -2.8, 'Detecting conflicts...', ha='center', fontweight='bold', color='red')
    
    # --- Phase 4: Final State After Splits ---
    ax4 = axes[1, 1]
    ax4.set_title('Phase 4: Final State with Split Episodes', fontsize=12, fontweight='bold')
    ax4.set_xlim(-3, 3)
    ax4.set_ylim(-3, 3)
    ax4.axis('off')
    
    # Final graph with split nodes
    G4 = nx.Graph()
    
    # All nodes including split ones
    final_positions = {
        'addition': (-1, 0.5),
        'subtraction': (1, 0.5),
        'division': (0.5, -0.5),
        'negative_numbers': (-2.5, 0),
        'fraction_ratio': (2.5, 0),
        'algebraic': (0, -2),
        'mult_basic': (-2, -1),
        'mult_advanced': (-1, -1.5),
        'func_basic': (1, 1),
        'func_advanced': (2, 0.5),
        'num_basic': (-0.5, 2),
        'num_advanced': (0.5, 2)
    }
    
    # Group nodes by type
    unchanged_nodes = ['addition', 'subtraction', 'division', 'negative_numbers', 
                      'fraction_ratio', 'algebraic']
    basic_nodes = ['mult_basic', 'func_basic', 'num_basic']
    advanced_nodes = ['mult_advanced', 'func_advanced', 'num_advanced']
    
    # Draw nodes by group
    nx.draw_networkx_nodes(G4, final_positions, nodelist=unchanged_nodes, ax=ax4,
                          node_color=colors['normal'], node_size=1000,
                          edgecolors='black', linewidths=2)
    nx.draw_networkx_nodes(G4, final_positions, nodelist=basic_nodes, ax=ax4,
                          node_color=colors['phase1'], node_size=1000,
                          edgecolors='black', linewidths=2)
    nx.draw_networkx_nodes(G4, final_positions, nodelist=advanced_nodes, ax=ax4,
                          node_color=colors['phase3'], node_size=1000,
                          edgecolors='black', linewidths=2)
    
    # Labels
    labels = {
        'mult_basic': 'mult\nbasic',
        'mult_advanced': 'mult\nadv',
        'func_basic': 'func\nbasic',
        'func_advanced': 'func\nadv',
        'num_basic': 'num\nbasic',
        'num_advanced': 'num\nadv'
    }
    for node in unchanged_nodes:
        labels[node] = node[:4]
    
    nx.draw_networkx_labels(G4, final_positions, labels, ax=ax4, font_size=7)
    
    # Draw some connections
    connections = [
        ('num_basic', 'addition'), ('num_advanced', 'negative_numbers'),
        ('mult_basic', 'addition'), ('mult_advanced', 'algebraic'),
        ('func_basic', 'algebraic'), ('func_advanced', 'algebraic')
    ]
    
    for u, v in connections:
        ax4.plot([final_positions[u][0], final_positions[v][0]], 
                [final_positions[u][1], final_positions[v][1]], 
                'k-', alpha=0.3, linewidth=1)
    
    ax4.text(0, -2.8, '14 nodes (3 splits), complex structure', ha='center', fontweight='bold')
    
    # Add legend
    legend_elements = [
        patches.Patch(color=colors['phase1'], label='Basic/Elementary'),
        patches.Patch(color=colors['phase2'], label='Middle School'),
        patches.Patch(color=colors['phase3'], label='Advanced'),
        patches.Patch(color=colors['split'], label='Conflict/Split'),
        patches.Patch(color=colors['normal'], label='Unchanged')
    ]
    ax4.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(-0.1, 1), fontsize=8)
    
    plt.tight_layout()
    
    # Save
    output_file = "graph_state_evolution.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n📊 Graph state evolution saved to: {output_file}")
    
    # Create detailed split diagram
    create_split_detail_diagram()


def create_split_detail_diagram():
    """Create detailed diagram showing what happens during a split"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Episode Memory Split Operation Detail', fontsize=14, fontweight='bold')
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Original episode
    orig_box = FancyBboxPatch((1, 2.5), 3, 1.5,
                              boxstyle="round,pad=0.1",
                              facecolor='lightblue',
                              edgecolor='black',
                              linewidth=2)
    ax.add_patch(orig_box)
    ax.text(2.5, 3.5, 'Original Episode', ha='center', fontweight='bold')
    ax.text(2.5, 3, 'Multiplication = repeated addition', ha='center')
    ax.text(2.5, 2.7, 'Phase 1 understanding', ha='center', fontsize=9, style='italic')
    
    # Conflict
    conflict_box = FancyBboxPatch((4.5, 3.5), 1.5, 0.8,
                                  boxstyle="round,pad=0.05",
                                  facecolor='#FF6347',
                                  edgecolor='red',
                                  linewidth=2)
    ax.add_patch(conflict_box)
    ax.text(5.25, 3.9, 'CONFLICT', ha='center', fontweight='bold', color='white')
    
    ax.text(5.25, 2.8, 'New information:', ha='center', fontweight='bold')
    ax.text(5.25, 2.4, 'Multiplication = scaling', ha='center')
    ax.text(5.25, 2, 'Cannot reconcile with', ha='center', fontsize=9)
    ax.text(5.25, 1.7, 'existing understanding', ha='center', fontsize=9)
    
    # Split arrow
    split_arrow = FancyArrowPatch((6, 3.5), (7, 3.5),
                                 connectionstyle="arc3,rad=0",
                                 arrowstyle='->',
                                 mutation_scale=25,
                                 color='red',
                                 linewidth=3)
    ax.add_patch(split_arrow)
    ax.text(6.5, 4, 'SPLIT', ha='center', fontweight='bold', color='red')
    
    # Result: Two episodes
    basic_box = FancyBboxPatch((7.5, 4), 2.5, 1.2,
                               boxstyle="round,pad=0.1",
                               facecolor='#FFE4B5',
                               edgecolor='black',
                               linewidth=2)
    ax.add_patch(basic_box)
    ax.text(8.75, 4.8, 'Episode 1: Basic', ha='center', fontweight='bold')
    ax.text(8.75, 4.4, 'Multiplication = repeated', ha='center', fontsize=9)
    ax.text(8.75, 4.1, 'addition (elementary)', ha='center', fontsize=9)
    
    advanced_box = FancyBboxPatch((7.5, 1.5), 2.5, 1.2,
                                  boxstyle="round,pad=0.1",
                                  facecolor='#DDA0DD',
                                  edgecolor='black',
                                  linewidth=2)
    ax.add_patch(advanced_box)
    ax.text(8.75, 2.3, 'Episode 2: Advanced', ha='center', fontweight='bold')
    ax.text(8.75, 1.9, 'Multiplication = scaling', ha='center', fontsize=9)
    ax.text(8.75, 1.6, 'operation (advanced)', ha='center', fontsize=9)
    
    # Benefits
    ax.text(5, 0.8, 'Benefits of Split:', ha='center', fontweight='bold', fontsize=11)
    ax.text(5, 0.4, '• Both understandings preserved • Context-appropriate retrieval • No information loss',
           ha='center', fontsize=9)
    
    plt.tight_layout()
    
    # Save
    output_file = "episode_split_detail.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Episode split detail saved to: {output_file}")


if __name__ == "__main__":
    create_graph_state_evolution()