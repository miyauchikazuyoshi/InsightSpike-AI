#!/usr/bin/env python3
"""
Create an animated visualization of InsightSpike's GNN message passing
Shows how a query propagates through the knowledge graph and transforms into insight
"""

import matplotlib
matplotlib.use('Agg')
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path

def create_knowledge_graph():
    """Create a sample knowledge graph"""
    G = nx.Graph()
    
    # Add nodes with positions
    nodes = {
        'Thermodynamics': (0, 2),
        'Information Theory': (2, 2),
        'Biology': (-1, 0),
        'Physics': (1, 0),
        'Systems Theory': (3, 0),
        'Energy': (0.5, 1),  # Emerges during message passing
        'Entropy': (1.5, 1)  # Emerges during message passing
    }
    
    for node, pos in nodes.items():
        G.add_node(node, pos=pos)
    
    # Add edges (initial connections)
    initial_edges = [
        ('Thermodynamics', 'Physics'),
        ('Information Theory', 'Systems Theory'),
        ('Biology', 'Systems Theory'),
        ('Physics', 'Systems Theory')
    ]
    
    G.add_edges_from(initial_edges)
    
    return G, nodes

def animate_gedig_process():
    """Create animation showing GNN message passing and insight detection"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=100)
    fig.patch.set_facecolor('#1e1e1e')
    
    # Create knowledge graph
    G, node_positions = create_knowledge_graph()
    pos = nx.get_node_attributes(G, 'pos')
    
    # Animation parameters
    frames = []
    
    # Frame 1: Initial graph
    frames.append({
        'title': 'Initial Knowledge Graph',
        'highlight_nodes': [],
        'highlight_edges': [],
        'new_nodes': [],
        'new_edges': [],
        'message': 'Disconnected knowledge domains',
        'ged': 0,
        'ig': 0,
        'query': None
    })
    
    # Frame 2: Query injection
    frames.append({
        'title': 'Query: "How are thermodynamic and information entropy related?"',
        'highlight_nodes': ['Thermodynamics', 'Information Theory'],
        'highlight_edges': [],
        'new_nodes': [],
        'new_edges': [],
        'message': 'Query activates relevant nodes',
        'ged': 0,
        'ig': 0,
        'query': 'active',
        'query_flow': [('Thermodynamics', 0.8), ('Information Theory', 0.8)]
    })
    
    # Frame 3: Message passing round 1
    frames.append({
        'title': 'GNN Message Passing - Round 1',
        'highlight_nodes': ['Thermodynamics', 'Physics', 'Information Theory', 'Systems Theory'],
        'highlight_edges': [('Thermodynamics', 'Physics'), ('Information Theory', 'Systems Theory')],
        'new_nodes': [],
        'new_edges': [],
        'message': 'Messages propagate through graph',
        'ged': -0.2,
        'ig': 0.1,
        'query': 'propagating',
        'query_flow': [('Physics', 0.6), ('Systems Theory', 0.6)]
    })
    
    # Frame 4: Emergence of new nodes
    frames.append({
        'title': 'Emergent Concepts Detected',
        'highlight_nodes': ['Energy', 'Entropy'],
        'highlight_edges': [],
        'new_nodes': ['Energy', 'Entropy'],
        'new_edges': [],
        'message': 'New conceptual nodes emerge',
        'ged': -0.5,
        'ig': 0.3,
        'query': 'transforming'
    })
    
    # Frame 5: New connections form
    frames.append({
        'title': 'Insight Spike! New Connections Form',
        'highlight_nodes': ['Energy', 'Entropy', 'Thermodynamics', 'Information Theory'],
        'highlight_edges': [],
        'new_nodes': ['Energy', 'Entropy'],
        'new_edges': [
            ('Thermodynamics', 'Entropy'),
            ('Information Theory', 'Entropy'),
            ('Energy', 'Entropy'),
            ('Physics', 'Energy'),
            ('Thermodynamics', 'Energy')
        ],
        'message': 'Entropy unifies thermodynamics & information!',
        'ged': -0.92,
        'ig': 0.56,
        'query': 'insight',
        'spike': True
    })
    
    # Frame 6: Final integrated graph
    frames.append({
        'title': 'Knowledge Graph Restructured',
        'highlight_nodes': [],
        'highlight_edges': [],
        'new_nodes': ['Energy', 'Entropy'],
        'new_edges': [
            ('Thermodynamics', 'Entropy'),
            ('Information Theory', 'Entropy'),
            ('Energy', 'Entropy'),
            ('Physics', 'Energy'),
            ('Thermodynamics', 'Energy')
        ],
        'message': 'New understanding integrated',
        'ged': -0.92,
        'ig': 0.56,
        'query': 'complete'
    })
    
    # Create frames
    images = []
    
    for frame_idx, frame in enumerate(frames):
        # Clear axes
        ax1.clear()
        ax2.clear()
        
        # Style axes
        for ax in [ax1, ax2]:
            ax.set_facecolor('#2a2a2a')
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        
        # Left panel: Graph visualization
        ax1.set_title(frame['title'], color='white', fontsize=14, pad=20)
        
        # Add existing nodes
        existing_nodes = [n for n in G.nodes() if n not in frame['new_nodes']]
        node_colors = []
        node_sizes = []
        
        for node in existing_nodes:
            if node in frame['highlight_nodes']:
                node_colors.append('#ff6b6b')
                node_sizes.append(1500)
            else:
                node_colors.append('#4ecdc4')
                node_sizes.append(800)
        
        nx.draw_networkx_nodes(G, pos, nodelist=existing_nodes, 
                              node_color=node_colors, node_size=node_sizes,
                              ax=ax1, alpha=0.9)
        
        # Add new nodes with special style
        if frame['new_nodes']:
            new_pos = {n: pos[n] for n in frame['new_nodes']}
            nx.draw_networkx_nodes(G, new_pos, nodelist=frame['new_nodes'],
                                  node_color='#ffd93d', node_size=1200,
                                  node_shape='s', ax=ax1, alpha=0.9)
        
        # Draw edges
        existing_edges = [e for e in G.edges() if e not in frame['new_edges']]
        edge_colors = []
        edge_widths = []
        
        for edge in existing_edges:
            if edge in frame['highlight_edges'] or edge[::-1] in frame['highlight_edges']:
                edge_colors.append('#ff6b6b')
                edge_widths.append(3)
            else:
                edge_colors.append('#666666')
                edge_widths.append(1)
        
        nx.draw_networkx_edges(G, pos, edgelist=existing_edges,
                              edge_color=edge_colors, width=edge_widths,
                              ax=ax1, alpha=0.7)
        
        # Draw new edges
        if frame['new_edges']:
            nx.draw_networkx_edges(G, pos, edgelist=frame['new_edges'],
                                  edge_color='#ffd93d', width=4,
                                  ax=ax1, alpha=0.8, style='dashed')
        
        # Draw labels
        labels = {n: n.replace(' ', '\\n') for n in G.nodes() if n not in frame['new_nodes']}
        nx.draw_networkx_labels(G, pos, labels, font_size=10, 
                               font_color='white', ax=ax1)
        
        if frame['new_nodes']:
            new_labels = {n: n for n in frame['new_nodes']}
            nx.draw_networkx_labels(G, pos, new_labels, font_size=11,
                                   font_color='black', font_weight='bold', ax=ax1)
        
        # Add query flow animation
        if 'query_flow' in frame:
            for node, strength in frame['query_flow']:
                circle = plt.Circle(pos[node], 0.3 * strength, 
                                  color='yellow', alpha=0.3)
                ax1.add_patch(circle)
        
        # Right panel: Metrics and explanation
        ax2.text(0.5, 0.9, 'geDIG Metrics', ha='center', va='top',
                fontsize=16, color='white', weight='bold', transform=ax2.transAxes)
        
        # GED bar
        ged_value = frame['ged']
        ged_color = '#4ecdc4' if ged_value >= 0 else '#ff6b6b'
        ax2.barh(0.7, abs(ged_value), height=0.1, color=ged_color, alpha=0.8)
        ax2.text(0.1, 0.7, 'ΔGED:', va='center', color='white', fontsize=12)
        ax2.text(0.9, 0.7, f'{ged_value:.2f}', va='center', ha='right', 
                color='white', fontsize=12, weight='bold')
        
        # IG bar
        ig_value = frame['ig']
        ig_color = '#ffd93d' if ig_value > 0.2 else '#4ecdc4'
        ax2.barh(0.5, ig_value, height=0.1, color=ig_color, alpha=0.8)
        ax2.text(0.1, 0.5, 'ΔIG:', va='center', color='white', fontsize=12)
        ax2.text(0.9, 0.5, f'{ig_value:.2f}', va='center', ha='right',
                color='white', fontsize=12, weight='bold')
        
        # Message
        ax2.text(0.5, 0.3, frame['message'], ha='center', va='center',
                fontsize=14, color='#ffd93d', weight='bold', 
                transform=ax2.transAxes, wrap=True)
        
        # Spike indicator
        if frame.get('spike'):
            spike_box = FancyBboxPatch((0.1, 0.05), 0.8, 0.15,
                                      boxstyle="round,pad=0.05",
                                      facecolor='#ff6b6b', edgecolor='white',
                                      linewidth=2)
            ax2.add_patch(spike_box)
            ax2.text(0.5, 0.125, '⚡ INSIGHT SPIKE DETECTED! ⚡',
                    ha='center', va='center', fontsize=14,
                    color='white', weight='bold', transform=ax2.transAxes)
        
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        
        # Save frame
        plt.tight_layout()
        
        # Convert to image for GIF
        fig.canvas.draw()
        # Use buffer_rgba for compatibility
        buf = fig.canvas.buffer_rgba()
        image = np.frombuffer(buf, dtype='uint8')
        w, h = fig.canvas.get_width_height()
        image = image.reshape((h, w, 4))
        # Convert RGBA to RGB
        image = image[:, :, :3]
        images.append(image)
        
        # Add new edges/nodes to graph for next frame
        for node in frame['new_nodes']:
            if node not in G:
                G.add_node(node, pos=pos[node])
        for edge in frame['new_edges']:
            if edge not in G.edges():
                G.add_edge(*edge)
    
    # Save as GIF with PIL
    from PIL import Image
    output_path = Path(__file__).parent.parent / "gedig_animation.gif"
    
    # Convert numpy arrays to PIL Images
    pil_images = []
    for img_array in images:
        # Ensure correct dtype
        img_array = img_array.astype('uint8')
        pil_img = Image.fromarray(img_array)
        pil_images.append(pil_img)
    
    # Save animated GIF
    if pil_images:
        pil_images[0].save(
            output_path,
            save_all=True,
            append_images=pil_images[1:],
            duration=2000,  # 2 seconds per frame
            loop=0
        )
    
    plt.close()
    
    print(f"✅ Graph animation created: {output_path}")
    print(f"📏 Size: {output_path.stat().st_size / 1024:.1f} KB")
    print(f"🖼️  Frames: {len(images)}")
    print(f"📊 Frame sizes: {[img.shape for img in images[:3]]}")
    print(f"🎨 PIL images: {len(pil_images)} frames")
    
    return output_path

if __name__ == "__main__":
    animate_gedig_process()