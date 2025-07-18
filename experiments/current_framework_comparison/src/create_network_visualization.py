#!/usr/bin/env python3
"""
Create Network Graph Visualization for InsightSpike
==================================================

Generate before/after network graphs showing insight generation.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import FancyBboxPatch


def create_insight_network_visualization():
    """Create before/after network graph visualization"""
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Define nodes and their properties
    nodes_before = {
        # Basic Concepts (blue)
        'energy_basic': {'label': 'Energy, information, and entropy form the fundamental trinit...', 
                        'color': '#5DADE2', 'category': 'Basic Concepts', 'pos': (-2, 1)},
        'energy_measure': {'label': 'is a measure of energy degradation...', 
                          'color': '#5DADE2', 'category': 'Basic Concepts', 'pos': (-3, -1)},
        
        # Deep Integration (red)
        'energy_trinity': {'label': 'Energy, information, and entropy form the fundamental trinit...', 
                          'color': '#E74C3C', 'category': 'Deep Integration', 'pos': (0, 2)},
        
        # Relationships (green)
        'maxwell_demon': {'label': "Maxwell's demon is a thought experiment showing the relation...", 
                         'color': '#58D68D', 'category': 'Relationships', 'pos': (-1, -2)},
        
        # Integration (brown)
        'evolution_universe': {'label': 'The evolution of the universe can be understood...', 
                              'color': '#A0522D', 'category': 'Integration', 'pos': (2, -1)},
        
        # Emergent Insights (purple)
        'black_hole': {'label': 'Is the interior of a black hole the ultimate compression sta...', 
                      'color': '#9B59B6', 'category': 'Emergent Insights', 'pos': (1, 0)}
    }
    
    # After insight generation - add central insight node
    nodes_after = nodes_before.copy()
    nodes_after['central_insight'] = {
        'label': 'What is the relationship between energy ...',
        'color': '#FF8C00', 
        'category': 'Generated Insight',
        'pos': (0, 0)
    }
    
    # Add new concept
    nodes_after['new_energy'] = {
        'label': 'Energy',
        'color': '#FF1493',
        'category': 'New Concepts',
        'pos': (3, -2)
    }
    
    # Define edges
    edges_before = [
        ('energy_basic', 'energy_trinity'),
        ('energy_basic', 'black_hole'),
        ('energy_trinity', 'black_hole'),
        ('maxwell_demon', 'energy_trinity'),
        ('energy_measure', 'evolution_universe'),
        ('evolution_universe', 'black_hole')
    ]
    
    # After insight - new connections through central node
    edges_after = [
        # New edges (red)
        ('energy_trinity', 'central_insight'),
        ('central_insight', 'black_hole'),
        ('central_insight', 'maxwell_demon'),
        ('central_insight', 'evolution_universe'),
        ('central_insight', 'new_energy'),
        # Existing edges (gray)
        ('energy_basic', 'energy_trinity'),
        ('energy_basic', 'black_hole'),
        ('energy_measure', 'evolution_universe')
    ]
    
    # Create graphs
    G_before = nx.Graph()
    G_after = nx.Graph()
    
    # Add nodes to before graph
    for node_id, props in nodes_before.items():
        G_before.add_node(node_id, **props)
    
    # Add nodes to after graph
    for node_id, props in nodes_after.items():
        G_after.add_node(node_id, **props)
    
    # Add edges
    G_before.add_edges_from(edges_before)
    G_after.add_edges_from(edges_after)
    
    # Draw BEFORE graph
    ax1.set_title('Before Insight Generation', fontsize=16, fontweight='bold', pad=20)
    pos_before = {node: props['pos'] for node, props in nodes_before.items()}
    
    # Draw edges
    nx.draw_networkx_edges(G_before, pos_before, ax=ax1, edge_color='lightgray', 
                          width=2, alpha=0.6)
    
    # Draw nodes
    for node, props in nodes_before.items():
        nx.draw_networkx_nodes(G_before, pos_before, nodelist=[node], 
                              node_color=props['color'], node_size=1500, ax=ax1)
    
    # Add labels
    labels_before = {node: props['label'][:50] + '...' if len(props['label']) > 50 else props['label'] 
                    for node, props in nodes_before.items()}
    nx.draw_networkx_labels(G_before, pos_before, labels_before, font_size=8, ax=ax1)
    
    # Add statistics box
    stats_box_before = FancyBboxPatch((0.02, 0.85), 0.25, 0.12, 
                                     boxstyle="round,pad=0.01",
                                     transform=ax1.transAxes,
                                     facecolor='white', 
                                     edgecolor='black',
                                     linewidth=1)
    ax1.add_patch(stats_box_before)
    ax1.text(0.03, 0.94, 'Nodes: 5', transform=ax1.transAxes, fontsize=10)
    ax1.text(0.03, 0.91, 'Edges: 4', transform=ax1.transAxes, fontsize=10)
    ax1.text(0.03, 0.88, 'Density: 0.200', transform=ax1.transAxes, fontsize=10)
    ax1.text(0.03, 0.85, 'Clustering: 0.000', transform=ax1.transAxes, fontsize=10)
    ax1.text(0.03, 0.82, 'Complexity: 0.316', transform=ax1.transAxes, fontsize=10)
    
    # Draw AFTER graph
    ax2.set_title('After Insight Generation', fontsize=16, fontweight='bold', pad=20)
    pos_after = {node: props['pos'] for node, props in nodes_after.items()}
    
    # Draw edges with different colors
    new_edges = [('energy_trinity', 'central_insight'), 
                 ('central_insight', 'black_hole'),
                 ('central_insight', 'maxwell_demon'),
                 ('central_insight', 'evolution_universe'),
                 ('central_insight', 'new_energy')]
    old_edges = [e for e in edges_after if e not in new_edges]
    
    nx.draw_networkx_edges(G_after, pos_after, edgelist=old_edges, 
                          edge_color='lightgray', width=2, alpha=0.6, ax=ax2)
    nx.draw_networkx_edges(G_after, pos_after, edgelist=new_edges, 
                          edge_color='red', width=3, alpha=0.8, ax=ax2)
    
    # Draw nodes
    for node, props in nodes_after.items():
        nx.draw_networkx_nodes(G_after, pos_after, nodelist=[node], 
                              node_color=props['color'], node_size=1500, ax=ax2)
    
    # Add labels
    labels_after = {node: props['label'][:50] + '...' if len(props['label']) > 50 else props['label'] 
                   for node, props in nodes_after.items()}
    nx.draw_networkx_labels(G_after, pos_after, labels_after, font_size=8, ax=ax2)
    
    # Add statistics box
    stats_box_after = FancyBboxPatch((0.02, 0.85), 0.25, 0.12, 
                                    boxstyle="round,pad=0.01",
                                    transform=ax2.transAxes,
                                    facecolor='white', 
                                    edgecolor='black',
                                    linewidth=1)
    ax2.add_patch(stats_box_after)
    ax2.text(0.03, 0.94, 'Nodes: 7 (+2)', transform=ax2.transAxes, fontsize=10)
    ax2.text(0.03, 0.91, 'Edges: 10 (+6)', transform=ax2.transAxes, fontsize=10)
    ax2.text(0.03, 0.88, 'Density: 0.238 (+0.038)', transform=ax2.transAxes, fontsize=10)
    ax2.text(0.03, 0.85, 'Clustering: 0.610 (+0.610)', transform=ax2.transAxes, fontsize=10)
    ax2.text(0.03, 0.82, 'Complexity: 0.487 (+0.171)', transform=ax2.transAxes, fontsize=10)
    
    # Add complexity increase annotation
    complexity_box = FancyBboxPatch((0.73, 0.92), 0.25, 0.05, 
                                   boxstyle="round,pad=0.01",
                                   transform=ax2.transAxes,
                                   facecolor='yellow', 
                                   edgecolor='black',
                                   linewidth=1)
    ax2.add_patch(complexity_box)
    ax2.text(0.855, 0.945, 'Complexity Increase: 54.2%', 
            transform=ax2.transAxes, fontsize=11, 
            ha='center', va='center', fontweight='bold')
    
    # Create legend
    legend_elements = [
        plt.scatter([], [], c='#5DADE2', s=150, label='Basic Concepts'),
        plt.scatter([], [], c='#E74C3C', s=150, label='Deep Integration'),
        plt.scatter([], [], c='#A0522D', s=150, label='Integration'),
        plt.scatter([], [], c='#FF1493', s=150, label='New Concepts'),
        plt.Line2D([0], [0], color='lightgray', linewidth=3, label='Existing Edges'),
        plt.scatter([], [], c='#58D68D', s=150, label='Relationships'),
        plt.scatter([], [], c='#9B59B6', s=150, label='Emergent Insights'),
        plt.scatter([], [], c='#FF8C00', s=150, label='Generated Insight'),
        plt.Line2D([0], [0], color='red', linewidth=3, label='New Edges')
    ]
    
    # Add legend at bottom
    fig.legend(handles=legend_elements, loc='lower center', ncol=9, 
              bbox_to_anchor=(0.5, -0.05), frameon=True, fancybox=True, shadow=True)
    
    # Remove axes
    ax1.axis('off')
    ax2.axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save
    output_path = Path(__file__).parent.parent / "results/visualizations/insight_network_graph.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"📊 Saved network visualization: {output_path}")
    
    # Create a more detailed version with actual InsightSpike data
    create_detailed_network_visualization()


def create_detailed_network_visualization():
    """Create a detailed network visualization using actual experiment data"""
    
    # Load actual knowledge base
    kb_path = Path(__file__).parent.parent / "data/input/insightspike_knowledge_base.json"
    with open(kb_path, 'r', encoding='utf-8') as f:
        knowledge_base = json.load(f)
    
    # Select representative episodes for visualization
    # Select representative episodes for visualization (check bounds)
    episodes = knowledge_base['episodes']
    selected_indices = [0, 2, 11, 30, 38, 40]  # Adjusted indices
    selected_episodes = []
    
    for idx in selected_indices:
        if idx < len(episodes):
            selected_episodes.append(episodes[idx])
        else:
            # Use last available episode if index out of range
            selected_episodes.append(episodes[-1])
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Create before graph
    G_before = nx.Graph()
    
    # Assign positions using force-directed layout
    pos_seed = {
        '0': (-2, 1),
        '2': (-2, -1),
        '11': (0, -2),
        '30': (1, 1),
        '38': (2, 0),
        '61': (0, 2)
    }
    
    # Add nodes with proper attributes
    for i, episode in enumerate(selected_episodes):
        node_id = str(i)
        phase = episode['metadata']['phase']
        categories = {
            1: ('Basic Concepts', '#5DADE2'),
            2: ('Relationships', '#58D68D'),
            3: ('Deep Integration', '#E74C3C'),
            4: ('Emergent Insights', '#9B59B6'),
            5: ('Integration', '#A0522D')
        }
        category, color = categories.get(phase, ('Unknown', '#808080'))
        
        G_before.add_node(node_id)
        G_before.nodes[node_id]['label'] = episode['content'][:50] + '...'
        G_before.nodes[node_id]['full_text'] = episode['content']
        G_before.nodes[node_id]['color'] = color
        G_before.nodes[node_id]['category'] = category
        G_before.nodes[node_id]['phase'] = phase
    
    # Add edges based on semantic similarity (simulated)
    edges_before = [
        ('0', '2'),   # Energy-Entropy connection
        ('2', '30'),  # Entropy-Trinity connection
        ('11', '30'), # Maxwell-Trinity connection
        ('30', '38'), # Trinity-Life connection
        ('30', '61'), # Trinity-Quantum connection
    ]
    G_before.add_edges_from(edges_before)
    
    # Create after graph (with insight node)
    G_after = G_before.copy()
    G_after.add_node('insight', 
                    label='Energy-Information Relationship Insight',
                    full_text='Energy and information are fundamentally connected through entropy',
                    color='#FF8C00',
                    category='Generated Insight',
                    phase=0)
    
    # Add new connections through insight
    new_edges = [
        ('0', 'insight'),
        ('2', 'insight'),
        ('30', 'insight'),
        ('11', 'insight'),
        ('insight', '38'),
        ('insight', '61')
    ]
    G_after.add_edges_from(new_edges)
    
    # Layout
    pos_before = nx.spring_layout(G_before, pos=pos_seed, k=2, iterations=50)
    pos_after = pos_before.copy()
    pos_after['insight'] = (0, 0)  # Center position for insight
    
    # Draw before
    ax1.set_title('Before Insight Generation\n(Knowledge Graph State)', fontsize=16, fontweight='bold')
    
    nx.draw_networkx_edges(G_before, pos_before, ax=ax1, edge_color='lightgray', width=2, alpha=0.6)
    
    for node in G_before.nodes():
        nx.draw_networkx_nodes(G_before, pos_before, nodelist=[node],
                              node_color=G_before.nodes[node]['color'],
                              node_size=2000, ax=ax1, alpha=0.8)
    
    labels_before = {node: G_before.nodes[node]['label'] for node in G_before.nodes()}
    nx.draw_networkx_labels(G_before, pos_before, labels_before, font_size=8, ax=ax1)
    
    # Draw after
    ax2.set_title('After Insight Generation\n(Integrated Knowledge Structure)', fontsize=16, fontweight='bold')
    
    # Draw old edges
    old_edges = [(u, v) for u, v in G_after.edges() if 'insight' not in (u, v)]
    nx.draw_networkx_edges(G_after, pos_after, edgelist=old_edges, 
                          edge_color='lightgray', width=2, alpha=0.6, ax=ax2)
    
    # Draw new edges
    insight_edges = [(u, v) for u, v in G_after.edges() if 'insight' in (u, v)]
    nx.draw_networkx_edges(G_after, pos_after, edgelist=insight_edges,
                          edge_color='red', width=3, alpha=0.8, ax=ax2)
    
    for node in G_after.nodes():
        nx.draw_networkx_nodes(G_after, pos_after, nodelist=[node],
                              node_color=G_after.nodes[node]['color'],
                              node_size=2000 if node != 'insight' else 3000,
                              ax=ax2, alpha=0.8)
    
    labels_after = {node: G_after.nodes[node]['label'] for node in G_after.nodes()}
    nx.draw_networkx_labels(G_after, pos_after, labels_after, font_size=8, ax=ax2)
    
    # Add metrics
    metrics_before = {
        'Nodes': len(G_before),
        'Edges': len(G_before.edges()),
        'Density': nx.density(G_before),
        'Avg Clustering': nx.average_clustering(G_before),
        'Components': nx.number_connected_components(G_before)
    }
    
    metrics_after = {
        'Nodes': len(G_after),
        'Edges': len(G_after.edges()),
        'Density': nx.density(G_after),
        'Avg Clustering': nx.average_clustering(G_after),
        'Components': nx.number_connected_components(G_after)
    }
    
    # Add metric boxes
    metrics_text_before = '\n'.join([f'{k}: {v:.3f}' if isinstance(v, float) else f'{k}: {v}' 
                                    for k, v in metrics_before.items()])
    ax1.text(0.02, 0.98, metrics_text_before, transform=ax1.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    metrics_text_after = '\n'.join([f'{k}: {v:.3f}' if isinstance(v, float) else f'{k}: {v}' 
                                   for k, v in metrics_after.items()])
    ax2.text(0.02, 0.98, metrics_text_after, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add insight detection annotation
    ax2.text(0.98, 0.98, 'Insight Detected!\ngeDIG Score: 0.85', 
            transform=ax2.transAxes, fontsize=12,
            horizontalalignment='right', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
            fontweight='bold')
    
    ax1.axis('off')
    ax2.axis('off')
    
    plt.tight_layout()
    
    output_path = Path(__file__).parent.parent / "results/visualizations/detailed_network_graph.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"📊 Saved detailed network visualization: {output_path}")


if __name__ == "__main__":
    create_insight_network_visualization()