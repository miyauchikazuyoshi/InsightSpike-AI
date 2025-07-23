#!/usr/bin/env python3
"""
Create Insight Detection Animation from Latest Results
=====================================================

Creates animated GIF showing graph reorganization during insight detection.
"""

import os
import sys
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sentence_transformers import SentenceTransformer


class InsightAnimationCreator:
    """Create animated visualization of insight detection"""
    
    def __init__(self, result_file: str):
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.load_results(result_file)
        
    def load_results(self, result_file: str):
        """Load experiment results"""
        with open(result_file, 'r') as f:
            self.data = json.load(f)
            
        # Find the best insight detection result
        best_result = None
        best_confidence = 0
        
        for result in self.data['detailed_results']:
            if result['has_spike'] and result['spike_confidence'] > best_confidence:
                best_result = result
                best_confidence = result['spike_confidence']
                
        self.selected_result = best_result
        print(f"Selected question: {best_result['question']}")
        print(f"Spike confidence: {best_confidence:.2%}")
        
    def extract_concepts(self, response: str) -> List[str]:
        """Extract concepts from the response"""
        concepts = []
        
        # Parse the structured response
        lines = response.split('\n')
        for line in lines:
            if ':' in line and any(phase in line for phase in ['Foundational', 'Relational', 'Integrative', 'Exploratory', 'Transcendent']):
                # Extract concepts after the colon
                content = line.split(':', 1)[1].strip()
                # Split by commas or periods
                for part in content.replace('...', '').split(','):
                    concept = part.strip()
                    if len(concept) > 10 and len(concept) < 100:
                        concepts.append(concept)
        
        # Limit to top 10 concepts for visualization
        return concepts[:10]
        
    def create_animation(self, output_file: str):
        """Create the animated GIF"""
        if not self.selected_result:
            print("No suitable result found for animation")
            return
            
        # Extract concepts from the response
        concepts = self.extract_concepts(self.selected_result['response'])
        question = self.selected_result['question']
        confidence = self.selected_result['spike_confidence']
        
        # Create embeddings
        embeddings = {}
        texts = [question] + concepts
        for i, text in enumerate(texts):
            embeddings[i] = self.embed_model.encode(text)
            
        # Calculate similarities
        G = nx.Graph()
        for i in range(len(texts)):
            G.add_node(i, text=texts[i][:30] + "..." if len(texts[i]) > 30 else texts[i])
            
        # Add edges based on similarity
        for i in range(len(texts)):
            for j in range(i+1, len(texts)):
                sim = np.dot(embeddings[i], embeddings[j]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
                if sim > 0.3:  # Lower threshold to show more connections
                    G.add_edge(i, j, weight=sim)
        
        # Create animation
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Animation parameters
        frames = 60
        spike_frame = 30  # When spike is detected
        
        def animate(frame):
            ax.clear()
            
            # Title changes when spike is detected
            if frame < spike_frame:
                ax.set_title(f"Processing: {question[:60]}...", fontsize=14, fontweight='bold')
                color_map = ['lightblue'] + ['lightgray'] * (len(texts) - 1)
                edge_color = 'gray'
                edge_width = 1
            else:
                ax.set_title(f"✨ INSIGHT DETECTED! (Confidence: {confidence:.1%}) ✨", 
                           fontsize=16, fontweight='bold', color='darkgreen')
                # Color nodes by connection strength
                color_map = ['gold'] + ['lightgreen'] * (len(texts) - 1)
                edge_color = 'darkgreen'
                edge_width = 3
            
            # Dynamic layout - converges when insight is detected
            if frame < spike_frame:
                # Start with random layout
                pos = nx.spring_layout(G, k=2, iterations=frame+1, seed=42)
            else:
                # Converge to organized layout
                pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
            
            # Draw the graph
            nx.draw_networkx_nodes(G, pos, node_color=color_map, 
                                 node_size=2000, alpha=0.9, ax=ax)
            
            # Draw edges with varying width based on progress
            if frame >= spike_frame - 10:
                # Gradually show edges
                edge_alpha = min(1.0, (frame - spike_frame + 10) / 20)
                nx.draw_networkx_edges(G, pos, edge_color=edge_color, 
                                     width=edge_width, alpha=edge_alpha, ax=ax)
            
            # Draw labels
            labels = {i: G.nodes[i]['text'] for i in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
            
            # Add insight indicators
            if frame >= spike_frame:
                # Add sparkles around the graph
                for _ in range(5):
                    x = np.random.uniform(-1.5, 1.5)
                    y = np.random.uniform(-1.5, 1.5)
                    ax.scatter(x, y, s=100, c='gold', marker='*', alpha=0.7)
            
            # Add metrics display
            if frame >= spike_frame:
                metrics_text = f"Connectivity: {self.selected_result['metrics']['connectivity_ratio']:.2f}\n"
                metrics_text += f"Phase Diversity: {self.selected_result['metrics']['phase_diversity']:.2f}\n"
                metrics_text += f"Nodes: {int(self.selected_result['metrics']['node_count'])}"
                ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, 
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
            ax.axis('off')
            
        # Create animation
        anim = FuncAnimation(fig, animate, frames=frames, interval=100, repeat=True)
        
        # Save as GIF
        writer = PillowWriter(fps=10)
        anim.save(output_file, writer=writer)
        plt.close()
        
        print(f"Animation saved to: {output_file}")


def main():
    """Create animation from latest results"""
    # Find latest result file
    results_dir = Path(__file__).parent.parent / "results" / "outputs"
    result_files = list(results_dir.glob("comprehensive_results_*.json"))
    
    if not result_files:
        print("No result files found")
        return
        
    latest_result = max(result_files, key=lambda x: x.stat().st_mtime)
    print(f"Using result file: {latest_result}")
    
    # Create animation
    creator = InsightAnimationCreator(str(latest_result))
    
    # Output file
    output_dir = Path(__file__).parent.parent / "results" / "visualizations"
    output_dir.mkdir(exist_ok=True, parents=True)
    output_file = output_dir / "insight_detection_animation_latest.gif"
    
    creator.create_animation(str(output_file))


if __name__ == "__main__":
    main()