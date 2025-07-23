#!/usr/bin/env python3
"""
Visualize Knowledge Graph with Query Injection
=============================================
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
from sentence_transformers import SentenceTransformer
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

def create_knowledge_graph_visualization():
    """Create comprehensive visualization of knowledge graph with query injection"""
    
    # Load knowledge base
    kb_path = Path(__file__).parent.parent / "data" / "input" / "knowledge_base_100.json"
    with open(kb_path) as f:
        kb_data = json.load(f)
    
    # Initialize embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Select representative nodes (20 nodes across phases)
    selected_items = []
    for phase in range(1, 6):
        phase_items = [item for item in kb_data['knowledge_items'] if item['phase'] == phase]
        selected_items.extend(phase_items[:4])  # 4 items per phase
    
    # Create graph
    G = nx.Graph()
    embeddings = {}
    phase_colors = {
        1: '#E8F4FD',  # Light blue - Foundational
        2: '#B8E0D2',  # Light green - Relational  
        3: '#F7DC6F',  # Yellow - Integrative
        4: '#F8C471',  # Orange - Exploratory
        5: '#F1948A'   # Red - Transcendent
    }
    
    # Add nodes with embeddings
    for item in selected_items:
        node_id = item['id']
        text = item['text']
        embedding = model.encode(text)
        embeddings[node_id] = embedding
        
        G.add_node(node_id, 
                  text=text[:40] + '...' if len(text) > 40 else text,
                  phase=item['phase'],
                  category=item['category'])
    
    # Add edges based on similarity
    threshold = 0.4
    for i, item1 in enumerate(selected_items):
        for j, item2 in enumerate(selected_items[i+1:], i+1):
            similarity = np.dot(embeddings[item1['id']], embeddings[item2['id']]) / \
                        (np.linalg.norm(embeddings[item1['id']]) * np.linalg.norm(embeddings[item2['id']]))
            
            if similarity > threshold:
                G.add_edge(item1['id'], item2['id'], weight=similarity)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # Main title
    fig.suptitle('Knowledge Graph Evolution with Query Injection', fontsize=20, fontweight='bold')
    
    # Define test queries
    queries = [
        "How does information theory relate to thermodynamics?",
        "What is the fundamental nature of reality?",
        "Can consciousness emerge from quantum processes?"
    ]
    
    # Create 4 subplots: base graph + 3 queries
    for idx in range(4):
        ax = plt.subplot(2, 2, idx + 1)
        
        # Use hierarchical layout
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        
        if idx == 0:
            # Base knowledge graph
            ax.set_title('Base Knowledge Graph', fontsize=14, fontweight='bold')
            
            # Draw nodes by phase
            for phase in range(1, 6):
                phase_nodes = [n for n in G.nodes() if G.nodes[n]['phase'] == phase]
                nx.draw_networkx_nodes(G, pos, nodelist=phase_nodes,
                                     node_color=phase_colors[phase],
                                     node_size=500,
                                     alpha=0.9,
                                     ax=ax)
            
            # Draw edges
            nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
            
            # Draw labels
            labels = {n: f"{n}" for n in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
            
        else:
            # Query injection visualization
            query = queries[idx - 1]
            ax.set_title(f'Query: "{query[:50]}..."', fontsize=12, fontweight='bold')
            
            # Get query embedding
            query_embedding = model.encode(query)
            
            # Find relevant nodes
            relevances = {}
            for node_id, node_embedding in embeddings.items():
                similarity = np.dot(query_embedding, node_embedding) / \
                           (np.linalg.norm(query_embedding) * np.linalg.norm(node_embedding))
                relevances[node_id] = similarity
            
            # Sort by relevance
            sorted_nodes = sorted(relevances.items(), key=lambda x: x[1], reverse=True)
            top_nodes = [n[0] for n in sorted_nodes[:5]]  # Top 5 relevant nodes
            
            # Draw all nodes with varying alpha based on relevance
            for node in G.nodes():
                if node in top_nodes:
                    # Highlight relevant nodes
                    nx.draw_networkx_nodes(G, pos, nodelist=[node],
                                         node_color='red',
                                         node_size=800,
                                         alpha=0.9,
                                         ax=ax)
                else:
                    # Fade irrelevant nodes
                    nx.draw_networkx_nodes(G, pos, nodelist=[node],
                                         node_color=phase_colors[G.nodes[node]['phase']],
                                         node_size=300,
                                         alpha=0.3,
                                         ax=ax)
            
            # Draw edges with emphasis on connections between relevant nodes
            for edge in G.edges():
                if edge[0] in top_nodes and edge[1] in top_nodes:
                    # Strong connection
                    nx.draw_networkx_edges(G, pos, edgelist=[edge],
                                         width=3, alpha=0.8, edge_color='red', ax=ax)
                elif edge[0] in top_nodes or edge[1] in top_nodes:
                    # Weak connection
                    nx.draw_networkx_edges(G, pos, edgelist=[edge],
                                         width=1, alpha=0.4, edge_color='gray', ax=ax)
                else:
                    # Background connection
                    nx.draw_networkx_edges(G, pos, edgelist=[edge],
                                         width=0.5, alpha=0.1, edge_color='lightgray', ax=ax)
            
            # Draw labels for relevant nodes
            relevant_labels = {n: f"{n}" for n in top_nodes}
            nx.draw_networkx_labels(G, pos, relevant_labels, font_size=10, font_weight='bold', ax=ax)
            
            # Add spike detection indicator
            spike_detected = len([e for e in G.edges() if e[0] in top_nodes and e[1] in top_nodes]) > 3
            if spike_detected:
                ax.text(0.95, 0.95, '✨ SPIKE!', transform=ax.transAxes,
                       fontsize=14, fontweight='bold', color='red',
                       ha='right', va='top',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        ax.axis('off')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color=phase_colors[1], label='Phase 1: Foundational'),
        mpatches.Patch(color=phase_colors[2], label='Phase 2: Relational'),
        mpatches.Patch(color=phase_colors[3], label='Phase 3: Integrative'),
        mpatches.Patch(color=phase_colors[4], label='Phase 4: Exploratory'),
        mpatches.Patch(color=phase_colors[5], label='Phase 5: Transcendent'),
        mpatches.Patch(color='red', label='Query-Relevant Node')
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=6, fontsize=12, 
              bbox_to_anchor=(0.5, -0.05))
    
    plt.tight_layout()
    
    # Save visualization
    output_path = Path(__file__).parent.parent / "results" / "visualizations"
    output_path.mkdir(exist_ok=True, parents=True)
    
    plt.savefig(output_path / "knowledge_graph_query_injection.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_path / "knowledge_graph_query_injection.pdf", format='pdf', bbox_inches='tight')
    
    print(f"Visualization saved to {output_path}")
    
    # Create additional detailed visualization for paper
    create_detailed_spike_visualization()

def create_detailed_spike_visualization():
    """Create detailed visualization showing spike detection mechanism"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left: Graph structure metrics
    ax1.set_title('Graph Metrics Evolution', fontsize=14, fontweight='bold')
    
    # Simulate metric evolution
    time_steps = np.arange(0, 10, 0.1)
    connectivity = 0.2 + 0.6 * (1 - np.exp(-time_steps/3))
    phase_diversity = 0.1 + 0.8 * (1 - np.exp(-time_steps/4))
    spike_score = connectivity * 0.5 + phase_diversity * 0.5
    
    ax1.plot(time_steps, connectivity, 'b-', linewidth=2, label='Connectivity Ratio')
    ax1.plot(time_steps, phase_diversity, 'g-', linewidth=2, label='Phase Diversity')
    ax1.plot(time_steps, spike_score, 'r-', linewidth=3, label='Spike Score')
    ax1.axhline(y=0.5, color='k', linestyle='--', alpha=0.5, label='Spike Threshold')
    
    # Mark spike detection point
    spike_point = np.where(spike_score > 0.5)[0][0]
    ax1.scatter(time_steps[spike_point], spike_score[spike_point], 
               color='red', s=200, marker='*', zorder=5)
    ax1.annotate('Spike Detected!', 
                xy=(time_steps[spike_point], spike_score[spike_point]),
                xytext=(time_steps[spike_point]+1, spike_score[spike_point]+0.1),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=12, fontweight='bold')
    
    ax1.set_xlabel('Query Processing Steps', fontsize=12)
    ax1.set_ylabel('Metric Value', fontsize=12)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right')
    
    # Right: Reward function landscape
    ax2.set_title('geDIG Reward Function Landscape', fontsize=14, fontweight='bold')
    
    # Create 2D reward landscape
    delta_ged = np.linspace(0, 10, 100)
    delta_ig = np.linspace(0, 10, 100)
    X, Y = np.meshgrid(delta_ged, delta_ig)
    
    # Calculate reward (F = w1*ΔGED - kT*ΔIG)
    w1 = 1.0
    kT = 1.0
    F = w1 * X - kT * Y
    
    # Create contour plot
    contour = ax2.contourf(X, Y, F, levels=20, cmap='RdBu_r', alpha=0.8)
    contour_lines = ax2.contour(X, Y, F, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    
    # Add optimal path
    optimal_ged = np.linspace(0, 8, 50)
    optimal_ig = optimal_ged * 0.6  # Slightly less IG than GED
    ax2.plot(optimal_ged, optimal_ig, 'g-', linewidth=3, label='Optimal Insight Path')
    
    # Mark some points
    ax2.scatter([2, 5, 8], [1, 3, 5], color='yellow', s=200, marker='*', 
               edgecolor='black', linewidth=2, zorder=5, label='Insight Spikes')
    
    ax2.set_xlabel('ΔGED (Structure Change)', fontsize=12)
    ax2.set_ylabel('ΔIG (Information Gain)', fontsize=12)
    ax2.legend(loc='upper left')
    
    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax2)
    cbar.set_label('Reward F', fontsize=12)
    
    plt.tight_layout()
    
    # Save
    output_path = Path(__file__).parent.parent / "results" / "visualizations"
    plt.savefig(output_path / "spike_detection_mechanism.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_path / "spike_detection_mechanism.pdf", format='pdf', bbox_inches='tight')

if __name__ == "__main__":
    print("Generating knowledge graph visualizations...")
    create_knowledge_graph_visualization()
    print("Done!")