#!/usr/bin/env python3
"""
Visualize Query Transformation as Graph Images
Shows how queries evolve through knowledge graphs
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation
from datetime import datetime

from src.insightspike.core.query_transformation import QueryState, QueryTransformationHistory
import torch


def create_knowledge_graph():
    """Create a sample knowledge graph"""
    G = nx.Graph()
    
    # Add nodes with positions
    nodes = {
        'thermodynamics': {'pos': (0, 1), 'category': 'physics'},
        'information_theory': {'pos': (2, 1), 'category': 'mathematics'},
        'entropy': {'pos': (1, 0), 'category': 'concept'},
        'physics': {'pos': (-1, 2), 'category': 'field'},
        'mathematics': {'pos': (3, 2), 'category': 'field'},
        'boltzmann': {'pos': (0, -1), 'category': 'scientist'},
        'shannon': {'pos': (2, -1), 'category': 'scientist'},
    }
    
    for node_id, attrs in nodes.items():
        G.add_node(node_id, **attrs)
    
    # Add edges
    edges = [
        ('thermodynamics', 'physics'),
        ('thermodynamics', 'entropy'),
        ('thermodynamics', 'boltzmann'),
        ('information_theory', 'mathematics'),
        ('information_theory', 'entropy'),
        ('information_theory', 'shannon'),
        ('entropy', 'boltzmann'),
        ('entropy', 'shannon'),
    ]
    G.add_edges_from(edges)
    
    return G


def visualize_static_transformation():
    """Create static visualization of query transformation"""
    G = create_knowledge_graph()
    
    # Create figure with subplots for different stages
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle('Query Transformation through Knowledge Graph', fontsize=16)
    
    # Define query states
    query = "How are thermodynamic entropy and information entropy related?"
    stages = ['Initial', 'Exploring', 'Transforming', 'Insight']
    colors = ['yellow', 'orange', 'darkorange', 'green']
    confidences = [0.1, 0.4, 0.7, 0.9]
    
    # Query node connections at each stage
    query_connections = [
        [],  # Initial: no connections
        ['entropy'],  # Exploring: found entropy
        ['entropy', 'thermodynamics', 'information_theory'],  # Transforming: connecting concepts
        ['entropy', 'thermodynamics', 'information_theory', 'boltzmann', 'shannon']  # Insight: full understanding
    ]
    
    for i, (ax, stage, color, confidence, connections) in enumerate(
        zip(axes, stages, colors, confidences, query_connections)
    ):
        ax.set_title(f'{stage} (Confidence: {confidence:.0%})', fontsize=12)
        
        # Get node positions
        pos = nx.get_node_attributes(G, 'pos')
        
        # Add query node at center
        query_pos = (1, 1.5)
        pos['QUERY'] = query_pos
        
        # Draw base graph
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                              node_size=1000, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color='gray', 
                              alpha=0.5, width=1, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
        
        # Draw query node with stage-specific color
        nx.draw_networkx_nodes(G.subgraph(['QUERY']), pos={'QUERY': query_pos},
                              node_color=color, node_size=1500, ax=ax,
                              node_shape='s')  # Square for query
        ax.text(query_pos[0], query_pos[1], 'QUERY', ha='center', va='center',
                fontsize=10, fontweight='bold')
        
        # Draw query connections
        for node in connections:
            if node in G:
                # Connection strength based on stage
                width = 2 + i  # Increasing width
                alpha = 0.3 + 0.2 * i  # Increasing opacity
                ax.plot([query_pos[0], pos[node][0]], 
                       [query_pos[1], pos[node][1]],
                       color=color, linewidth=width, alpha=alpha)
        
        # Add absorbed concepts text
        if connections:
            absorbed_text = f"Absorbed: {', '.join(connections[:2])}"
            if len(connections) > 2:
                absorbed_text += f" +{len(connections)-2}"
            ax.text(0.5, -0.1, absorbed_text, transform=ax.transAxes,
                   ha='center', fontsize=8, style='italic')
        
        ax.set_xlim(-2, 4)
        ax.set_ylim(-2, 3)
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save the plot
    output_file = f"query_transformation_static_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Static visualization saved to: {output_file}")
    
    # plt.show()  # Comment out for headless operation


def visualize_transformation_metrics():
    """Visualize the transformation metrics over time"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
    fig.suptitle('Query Transformation Metrics', fontsize=16)
    
    # Simulated transformation data
    cycles = np.arange(0, 10)
    confidence = 1 / (1 + np.exp(-0.8 * (cycles - 5)))  # Sigmoid growth
    transformation_magnitude = np.cumsum(np.random.exponential(0.3, 10))
    transformation_magnitude = transformation_magnitude / transformation_magnitude[-1]
    insights = np.cumsum([0, 0, 0, 1, 0, 1, 1, 0, 1, 1])
    
    # Confidence trajectory
    ax1.plot(cycles, confidence, 'b-', linewidth=2, marker='o')
    ax1.fill_between(cycles, 0, confidence, alpha=0.3)
    ax1.set_ylabel('Confidence', fontsize=12)
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Query Confidence Evolution')
    
    # Color zones
    ax1.axhspan(0, 0.33, alpha=0.2, color='yellow', label='Initial')
    ax1.axhspan(0.33, 0.66, alpha=0.2, color='orange', label='Exploring')
    ax1.axhspan(0.66, 1.0, alpha=0.2, color='green', label='Insight')
    ax1.legend(loc='right')
    
    # Transformation magnitude
    ax2.plot(cycles, transformation_magnitude, 'r-', linewidth=2, marker='s')
    ax2.fill_between(cycles, 0, transformation_magnitude, alpha=0.3, color='red')
    ax2.set_ylabel('Transformation Magnitude', fontsize=12)
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Query Embedding Change')
    
    # Insights discovered
    ax3.bar(cycles, insights, color=['yellow' if i < 3 else 'orange' if i < 6 else 'green' 
                                     for i in range(len(cycles))])
    ax3.set_xlabel('Transformation Cycle', fontsize=12)
    ax3.set_ylabel('Cumulative Insights', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_title('Insights Discovered')
    
    # Add insight annotations
    insight_cycles = [3, 5, 6, 8, 9]
    insight_texts = ['Entropy connection', 'Mathematical link', 'S = k ln W', 
                     'Universal principle', 'Complete understanding']
    for cycle, text in zip(insight_cycles, insight_texts):
        ax3.annotate(text, xy=(cycle, insights[cycle]), 
                    xytext=(cycle, insights[cycle] + 0.5),
                    ha='center', fontsize=8, 
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))
    
    plt.tight_layout()
    
    # Save the plot
    output_file = f"query_transformation_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Metrics visualization saved to: {output_file}")
    
    # plt.show()  # Comment out for headless operation


def visualize_query_embedding_space():
    """Visualize query movement in embedding space"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax.set_title('Query Evolution in Embedding Space', fontsize=16)
    
    # Create transformation history
    history = QueryTransformationHistory("How are entropies related?")
    
    # Simulate query evolution in 2D space (using PCA projection of embeddings)
    np.random.seed(42)
    n_steps = 20
    
    # Start position
    x, y = [0], [0]
    colors = ['yellow']  # Initial color
    sizes = [50]  # Initial size
    
    for i in range(n_steps):
        # Simulate movement towards insight (upper right)
        dx = np.random.normal(0.1, 0.05) + 0.02 * i
        dy = np.random.normal(0.1, 0.05) + 0.01 * i
        
        x.append(x[-1] + dx)
        y.append(y[-1] + dy)
        
        # Color based on stage
        if i < 5:
            colors.append('yellow')
            sizes.append(50)
        elif i < 12:
            colors.append('orange')
            sizes.append(70)
        else:
            colors.append('green')
            sizes.append(100)
    
    # Plot trajectory
    for i in range(1, len(x)):
        ax.plot([x[i-1], x[i]], [y[i-1], y[i]], 
               color='gray', alpha=0.3, linewidth=1)
    
    # Plot points (convert sizes list to numpy array)
    scatter = ax.scatter(x, y, c=colors, s=np.array(sizes), alpha=0.7, edgecolors='black')
    
    # Add stage annotations
    ax.annotate('START', xy=(x[0], y[0]), xytext=(x[0]-0.3, y[0]-0.3),
               fontsize=12, fontweight='bold',
               arrowprops=dict(arrowstyle='->', color='black'))
    ax.annotate('INSIGHT!', xy=(x[-1], y[-1]), xytext=(x[-1]+0.3, y[-1]+0.3),
               fontsize=12, fontweight='bold', color='green',
               arrowprops=dict(arrowstyle='->', color='green'))
    
    # Add concept regions
    circle1 = patches.Circle((0.5, 0.5), 0.5, fill=False, 
                           edgecolor='blue', linestyle='--', label='Thermodynamics')
    circle2 = patches.Circle((1.5, 1.2), 0.5, fill=False, 
                           edgecolor='red', linestyle='--', label='Information Theory')
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    
    # Add absorbed concepts as text along the path
    concept_points = [5, 10, 15]
    concepts = ['entropy', 'disorder', 'S = k ln W']
    for idx, concept in zip(concept_points, concepts):
        ax.text(x[idx], y[idx] + 0.1, concept, fontsize=8, 
               ha='center', style='italic', color='darkblue')
    
    ax.set_xlabel('Embedding Dimension 1 (PCA)', fontsize=12)
    ax.set_ylabel('Embedding Dimension 2 (PCA)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Equal aspect ratio
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    # Save the plot
    output_file = f"query_embedding_evolution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Embedding space visualization saved to: {output_file}")
    
    # plt.show()  # Comment out for headless operation


def main():
    """Run all visualizations"""
    print("🎨 Creating Query Transformation Visualizations...\n")
    
    print("1. Creating static transformation stages...")
    visualize_static_transformation()
    
    print("\n2. Creating transformation metrics...")
    visualize_transformation_metrics()
    
    print("\n3. Creating embedding space evolution...")
    visualize_query_embedding_space()
    
    print("\n✨ All visualizations complete!")


if __name__ == "__main__":
    main()