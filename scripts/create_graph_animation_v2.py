#!/usr/bin/env python3
"""
Create an animated visualization using matplotlib's animation API
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import numpy as np
from pathlib import Path

def create_animation():
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.patch.set_facecolor('#1e1e1e')
    
    # Create initial graph
    G = nx.Graph()
    nodes = {
        'Thermodynamics': (0, 2),
        'Information Theory': (2, 2),
        'Biology': (-1, 0),
        'Physics': (1, 0),
        'Systems Theory': (3, 0),
    }
    
    for node, pos in nodes.items():
        G.add_node(node, pos=pos)
    
    initial_edges = [
        ('Thermodynamics', 'Physics'),
        ('Information Theory', 'Systems Theory'),
        ('Biology', 'Systems Theory'),
        ('Physics', 'Systems Theory')
    ]
    G.add_edges_from(initial_edges)
    
    pos = nx.get_node_attributes(G, 'pos')
    
    # Animation data
    animation_states = [
        {
            'title': 'Initial Knowledge Graph',
            'highlight': [],
            'new_nodes': [],
            'new_edges': [],
            'ged': 0,
            'ig': 0,
            'message': 'Disconnected domains'
        },
        {
            'title': 'Query Injection',
            'highlight': ['Thermodynamics', 'Information Theory'],
            'new_nodes': [],
            'new_edges': [],
            'ged': -0.2,
            'ig': 0.1,
            'message': 'Query activates nodes'
        },
        {
            'title': 'Message Passing',
            'highlight': ['Physics', 'Systems Theory'],
            'new_nodes': [],
            'new_edges': [],
            'ged': -0.5,
            'ig': 0.3,
            'message': 'Messages propagate'
        },
        {
            'title': 'New Concepts Emerge',
            'highlight': [],
            'new_nodes': ['Entropy'],
            'new_edges': [],
            'ged': -0.7,
            'ig': 0.4,
            'message': 'Entropy emerges!'
        },
        {
            'title': 'Insight Spike!',
            'highlight': ['Entropy'],
            'new_nodes': ['Entropy'],
            'new_edges': [('Thermodynamics', 'Entropy'), ('Information Theory', 'Entropy')],
            'ged': -0.92,
            'ig': 0.56,
            'message': 'NEW CONNECTIONS!'
        }
    ]
    
    # Add new nodes to graph
    G.add_node('Entropy', pos=(1, 1))
    
    def animate(frame):
        # Clear axes
        ax1.clear()
        ax2.clear()
        
        # Style
        for ax in [ax1, ax2]:
            ax.set_facecolor('#2a2a2a')
            ax.axis('off')
        
        state = animation_states[frame]
        
        # Title
        ax1.set_title(state['title'], color='white', fontsize=14, pad=20)
        
        # Draw graph
        all_nodes = list(nodes.keys()) + state['new_nodes']
        node_colors = ['#ff6b6b' if n in state['highlight'] else '#4ecdc4' if n in state['new_nodes'] else '#666666' for n in all_nodes]
        
        nx.draw_networkx_nodes(G, pos, nodelist=all_nodes, node_color=node_colors, node_size=1000, ax=ax1)
        
        # Draw edges
        all_edges = initial_edges + state['new_edges']
        edge_colors = ['#ffd93d' if e in state['new_edges'] else '#333333' for e in all_edges]
        edge_widths = [3 if e in state['new_edges'] else 1 for e in all_edges]
        
        nx.draw_networkx_edges(G, pos, edgelist=all_edges, edge_color=edge_colors, width=edge_widths, ax=ax1)
        
        # Labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_color='white', ax=ax1)
        
        # Metrics panel
        ax2.text(0.5, 0.8, 'geDIG Metrics', ha='center', fontsize=16, color='white', weight='bold')
        ax2.text(0.2, 0.6, f'ΔGED: {state["ged"]:.2f}', fontsize=14, color='#ff6b6b' if state['ged'] < -0.5 else 'white')
        ax2.text(0.2, 0.5, f'ΔIG: {state["ig"]:.2f}', fontsize=14, color='#ffd93d' if state['ig'] > 0.3 else 'white')
        ax2.text(0.5, 0.3, state['message'], ha='center', fontsize=12, color='#ffd93d', weight='bold')
        
        if frame == 4:  # Spike frame
            ax2.text(0.5, 0.1, '⚡ SPIKE! ⚡', ha='center', fontsize=16, color='red', weight='bold')
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=len(animation_states), interval=2000, repeat=True)
    
    # Save as GIF
    output_path = Path(__file__).parent.parent / "gedig_animation_v2.gif"
    anim.save(str(output_path), writer='pillow', fps=0.5)
    
    plt.close()
    
    print(f"✅ Animation created: {output_path}")
    print(f"📏 Size: {output_path.stat().st_size / 1024:.1f} KB")
    
    return output_path

if __name__ == "__main__":
    create_animation()