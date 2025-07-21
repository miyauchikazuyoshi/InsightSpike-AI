#!/usr/bin/env python3
"""
Visualize InsightSpike Graph Connections
========================================

Creates visual representations of knowledge graph and insight detection.
"""

import os
import sys
import json
import math
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
from matplotlib.animation import FuncAnimation
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sentence_transformers import SentenceTransformer


class GraphVisualizer:
    """Visualize knowledge graph and insight detection"""
    
    def __init__(self):
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.knowledge_graph = {}
        self.embeddings = {}
        
    def build_graph(self, knowledge_items: List[str]) -> nx.DiGraph:
        """Build knowledge graph with real embeddings"""
        G = nx.DiGraph()
        
        # Add nodes with embeddings
        for i, text in enumerate(knowledge_items):
            embedding = self.embed_model.encode(text).tolist()
            self.embeddings[i] = embedding
            
            # Shorten text for visualization
            short_text = text[:40] + "..." if len(text) > 40 else text
            G.add_node(i, text=short_text, full_text=text, embedding=embedding)
        
        # Add edges based on similarity
        for i in range(len(knowledge_items)):
            for j in range(i + 1, len(knowledge_items)):
                similarity = self._cosine_similarity(
                    self.embeddings[i], 
                    self.embeddings[j]
                )
                if similarity > 0.5:  # Threshold for connection
                    G.add_edge(i, j, weight=similarity)
                    G.add_edge(j, i, weight=similarity)  # Bidirectional
        
        return G
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        return dot_product / (norm1 * norm2 + 1e-8)
    
    def find_relevant_nodes(self, G: nx.DiGraph, question: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Find nodes relevant to the question"""
        q_embedding = self.embed_model.encode(question).tolist()
        
        relevant = []
        for node in G.nodes():
            node_embedding = G.nodes[node]['embedding']
            similarity = self._cosine_similarity(q_embedding, node_embedding)
            if similarity > 0.3:
                relevant.append((node, similarity))
        
        relevant.sort(key=lambda x: x[1], reverse=True)
        return relevant[:top_k]
    
    def detect_spike(self, G: nx.DiGraph, relevant_nodes: List[Tuple[int, float]]) -> Tuple[bool, float, List[Tuple[int, int]]]:
        """Detect insight spike based on connectivity"""
        if len(relevant_nodes) < 3:
            return False, 0.0, []
        
        # Find connections between relevant nodes
        connections = []
        total_weight = 0
        
        for i, (node1, _) in enumerate(relevant_nodes):
            for j, (node2, _) in enumerate(relevant_nodes[i+1:], i+1):
                if G.has_edge(node1, node2):
                    weight = G[node1][node2]['weight']
                    connections.append((node1, node2))
                    total_weight += weight
        
        # Calculate spike score
        max_possible = len(relevant_nodes) * (len(relevant_nodes) - 1) / 2
        connectivity_ratio = len(connections) / max_possible if max_possible > 0 else 0
        avg_weight = total_weight / len(connections) if connections else 0
        
        spike_score = connectivity_ratio * avg_weight
        has_spike = spike_score > 0.3
        
        return has_spike, spike_score, connections
    
    def visualize_static_graph(self, G: nx.DiGraph, question: str, output_path: str):
        """Create static visualization of insight detection"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Find relevant nodes and detect spike
        relevant_nodes = self.find_relevant_nodes(G, question)
        has_spike, spike_score, connections = self.detect_spike(G, relevant_nodes)
        
        relevant_node_ids = [n for n, _ in relevant_nodes]
        
        # Left: Full knowledge graph
        ax1.set_title("Knowledge Graph", fontsize=14, fontweight='bold')
        pos1 = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Node colors
        node_colors = []
        for node in G.nodes():
            if node in relevant_node_ids[:3]:  # Top 3 relevant
                node_colors.append('#ff6b6b')  # Red for highly relevant
            elif node in relevant_node_ids:
                node_colors.append('#ffd93d')  # Yellow for relevant
            else:
                node_colors.append('#95e1d3')  # Light green for others
        
        # Draw full graph
        nx.draw_networkx_nodes(G, pos1, node_color=node_colors, 
                              node_size=800, ax=ax1, alpha=0.9)
        nx.draw_networkx_edges(G, pos1, edge_color='gray', 
                              width=1, alpha=0.3, ax=ax1)
        
        # Labels for full graph
        labels = {n: f"{n}: {G.nodes[n]['text'][:15]}..." for n in G.nodes()}
        nx.draw_networkx_labels(G, pos1, labels, font_size=8, ax=ax1)
        
        # Right: Insight detection visualization
        ax2.set_title(f"Insight Detection: {'✨ SPIKE!' if has_spike else '📝 No Spike'}", 
                     fontsize=14, fontweight='bold')
        
        # Create subgraph of relevant nodes
        subgraph = G.subgraph(relevant_node_ids).copy()
        
        if len(subgraph.nodes()) > 0:
            pos2 = nx.circular_layout(subgraph)
            
            # Draw relevant nodes larger
            nx.draw_networkx_nodes(subgraph, pos2, 
                                  node_color=['#ff6b6b' if i < 3 else '#ffd93d' 
                                            for i in range(len(subgraph))],
                                  node_size=1500, ax=ax2, alpha=0.9)
            
            # Highlight spike connections
            edge_colors = []
            edge_widths = []
            for u, v in subgraph.edges():
                if (u, v) in connections or (v, u) in connections:
                    edge_colors.append('#ff6b6b')
                    edge_widths.append(3)
                else:
                    edge_colors.append('gray')
                    edge_widths.append(1)
            
            nx.draw_networkx_edges(subgraph, pos2, edge_color=edge_colors,
                                  width=edge_widths, ax=ax2, alpha=0.8)
            
            # Detailed labels
            detailed_labels = {n: G.nodes[n]['text'] for n in subgraph.nodes()}
            nx.draw_networkx_labels(subgraph, pos2, detailed_labels, 
                                   font_size=9, ax=ax2, bbox=dict(boxstyle="round,pad=0.3", 
                                                                   facecolor="white", 
                                                                   alpha=0.8))
        
        # Add question and spike info
        ax2.text(0.5, -0.15, f"Question: {question}", 
                transform=ax2.transAxes, ha='center', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.5))
        
        ax2.text(0.5, -0.25, f"Spike Score: {spike_score:.3f}", 
                transform=ax2.transAxes, ha='center', fontsize=10,
                color='red' if has_spike else 'gray', fontweight='bold')
        
        # Clean up axes
        ax1.axis('off')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return has_spike, spike_score
    
    def create_animation(self, G: nx.DiGraph, question: str, output_path: str):
        """Create animated visualization of insight detection"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Find relevant nodes
        relevant_nodes = self.find_relevant_nodes(G, question)
        has_spike, spike_score, connections = self.detect_spike(G, relevant_nodes)
        relevant_node_ids = [n for n, _ in relevant_nodes]
        
        # Use consistent layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        def animate(frame):
            ax.clear()
            ax.axis('off')
            
            if frame < 30:
                # Phase 1: Show full graph
                ax.set_title("Knowledge Graph", fontsize=16, fontweight='bold')
                
                nx.draw_networkx_nodes(G, pos, node_color='#95e1d3',
                                      node_size=600, ax=ax, alpha=0.7)
                nx.draw_networkx_edges(G, pos, edge_color='gray',
                                      width=1, alpha=0.3, ax=ax)
                
                labels = {n: f"{n}" for n in G.nodes()}
                nx.draw_networkx_labels(G, pos, labels, font_size=10, ax=ax)
                
            elif frame < 60:
                # Phase 2: Highlight relevant nodes
                ax.set_title(f"Finding Relevant Concepts for:\n'{question}'", 
                           fontsize=14, fontweight='bold')
                
                # Fade effect
                fade = (frame - 30) / 30
                
                node_colors = []
                node_sizes = []
                for node in G.nodes():
                    if node in relevant_node_ids[:3]:
                        node_colors.append('#ff6b6b')
                        node_sizes.append(600 + 400 * fade)
                    elif node in relevant_node_ids:
                        node_colors.append('#ffd93d')
                        node_sizes.append(600 + 200 * fade)
                    else:
                        node_colors.append('#95e1d3')
                        node_sizes.append(600 - 200 * fade)
                
                nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                                      node_size=node_sizes, ax=ax, alpha=0.8)
                nx.draw_networkx_edges(G, pos, edge_color='gray',
                                      width=1, alpha=0.2, ax=ax)
                
                labels = {n: f"{n}" for n in G.nodes()}
                nx.draw_networkx_labels(G, pos, labels, font_size=10, ax=ax)
                
            else:
                # Phase 3: Show connections and spike
                ax.set_title(f"Insight Detection: {'✨ SPIKE DETECTED!' if has_spike else '📝 No Spike'}\n(Score: {spike_score:.3f})", 
                           fontsize=16, fontweight='bold',
                           color='red' if has_spike else 'gray')
                
                # Draw all nodes faded
                nx.draw_networkx_nodes(G, pos, node_color='lightgray',
                                      node_size=400, ax=ax, alpha=0.3)
                
                # Highlight relevant nodes
                relevant_positions = {n: pos[n] for n in relevant_node_ids}
                nx.draw_networkx_nodes(G, pos, nodelist=relevant_node_ids[:3],
                                      node_color='#ff6b6b', node_size=1200,
                                      ax=ax, alpha=0.9)
                if len(relevant_node_ids) > 3:
                    nx.draw_networkx_nodes(G, pos, nodelist=relevant_node_ids[3:],
                                          node_color='#ffd93d', node_size=800,
                                          ax=ax, alpha=0.9)
                
                # Animate connections
                pulse = 0.5 + 0.5 * math.sin((frame - 60) * 0.2)
                
                for u, v in connections:
                    if u in relevant_positions and v in relevant_positions:
                        x = [pos[u][0], pos[v][0]]
                        y = [pos[u][1], pos[v][1]]
                        ax.plot(x, y, 'r-', linewidth=3 * pulse, alpha=0.8)
                
                # Labels for relevant nodes
                relevant_labels = {n: G.nodes[n]['text'][:20] + "..." 
                                 for n in relevant_node_ids}
                nx.draw_networkx_labels(G, pos, relevant_labels, 
                                       font_size=9, ax=ax,
                                       bbox=dict(boxstyle="round,pad=0.3",
                                               facecolor="white", alpha=0.8))
        
        anim = FuncAnimation(fig, animate, frames=90, interval=50, repeat=True)
        anim.save(output_path, writer='pillow', fps=20)
        plt.close()


def run_visualization():
    """Run visualization of insight detection"""
    print("Creating InsightSpike Visualizations...")
    
    # Knowledge base
    knowledge_items = [
        "Energy is the capacity to do work.",
        "Information is defined as the reduction of uncertainty.",
        "Information and entropy have a deep mathematical relationship.",
        "The second law of thermodynamics and Shannon's information theory share the same mathematical structure.",
        "Energy, information, and entropy form the fundamental trinity of the universe.",
        "Life is a dissipative structure that locally decreases entropy.",
        "Consciousness might be quantified by Integrated Information Theory.",
        "Evolution is a process of increasing information processing capability.",
        "Energy, information, and consciousness are different aspects of the same reality.",
        "All physical laws reduce to laws of information conservation and transformation."
    ]
    
    # Test questions
    questions = [
        "How are energy and information fundamentally related?",
        "Can consciousness be understood through information theory?",
        "How does life organize information against entropy?"
    ]
    
    # Create visualizer
    viz = GraphVisualizer()
    
    # Build graph
    print("\nBuilding knowledge graph...")
    G = viz.build_graph(knowledge_items)
    print(f"Graph created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Create output directory
    output_dir = Path(__file__).parent.parent / "results" / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate visualizations for each question
    results = []
    for i, question in enumerate(questions):
        print(f"\nVisualizing Question {i+1}: {question}")
        
        # Static visualization
        static_path = output_dir / f"insight_detection_q{i+1}.png"
        has_spike, spike_score = viz.visualize_static_graph(G, question, str(static_path))
        print(f"  Static visualization saved: {static_path}")
        print(f"  Spike: {'Yes' if has_spike else 'No'} (score: {spike_score:.3f})")
        
        # Animated visualization (only for first question)
        if i == 0:
            anim_path = output_dir / f"insight_detection_animation_q{i+1}.gif"
            viz.create_animation(G, question, str(anim_path))
            print(f"  Animation saved: {anim_path}")
        
        results.append({
            'question': question,
            'has_spike': has_spike,
            'spike_score': spike_score,
            'visualization': str(static_path.name)
        })
    
    # Create combined visualization
    print("\nCreating combined visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()
    
    # First subplot: Full graph
    ax = axes[0]
    ax.set_title("Complete Knowledge Graph", fontsize=14, fontweight='bold')
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    nx.draw_networkx_nodes(G, pos, node_color='#95e1d3', node_size=600, ax=ax, alpha=0.8)
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=1, alpha=0.4, ax=ax)
    labels = {n: f"{n}" for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=10, ax=ax)
    ax.axis('off')
    
    # Other subplots: Each question
    for i, (question, result) in enumerate(zip(questions, results)):
        ax = axes[i+1]
        relevant_nodes = viz.find_relevant_nodes(G, question)
        has_spike, spike_score, connections = viz.detect_spike(G, relevant_nodes)
        relevant_node_ids = [n for n, _ in relevant_nodes]
        
        ax.set_title(f"Q{i+1}: {'✨ SPIKE!' if has_spike else '📝 No Spike'} (score: {spike_score:.3f})", 
                    fontsize=12, fontweight='bold')
        
        # Highlight relevant nodes
        node_colors = []
        for node in G.nodes():
            if node in relevant_node_ids[:3]:
                node_colors.append('#ff6b6b')
            elif node in relevant_node_ids:
                node_colors.append('#ffd93d')
            else:
                node_colors.append('#e0e0e0')
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=500, ax=ax, alpha=0.8)
        
        # Highlight connections
        for u, v in G.edges():
            if (u, v) in connections or (v, u) in connections:
                x = [pos[u][0], pos[v][0]]
                y = [pos[u][1], pos[v][1]]
                ax.plot(x, y, 'r-', linewidth=2, alpha=0.8)
            else:
                x = [pos[u][0], pos[v][0]]
                y = [pos[u][1], pos[v][1]]
                ax.plot(x, y, 'gray', linewidth=0.5, alpha=0.2)
        
        ax.text(0.5, -0.1, question[:50] + "...", 
               transform=ax.transAxes, ha='center', fontsize=9)
        ax.axis('off')
    
    plt.tight_layout()
    combined_path = output_dir / "insight_detection_combined.png"
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Combined visualization saved: {combined_path}")
    
    # Save results
    results_file = output_dir / "visualization_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'knowledge_items': len(knowledge_items),
            'graph_edges': G.number_of_edges(),
            'results': results
        }, f, indent=2)
    
    print(f"\nResults saved: {results_file}")
    print("\n✅ Visualizations complete!")


if __name__ == "__main__":
    run_visualization()