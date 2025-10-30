#!/usr/bin/env python3
"""
Create brain 4-layer structure analogy diagram for geDIG/InsightSpike-AI
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import numpy as np

def create_brain_analogy_diagram():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 10))
    
    # Title
    fig.suptitle('Brain-Inspired Architecture: From Neuroscience to InsightSpike-AI', 
                fontsize=16, fontweight='bold')
    
    # Colors
    colors = {
        'sensory': '#FFE4B5',      # Moccasin
        'episodic': '#87CEEB',     # SkyBlue
        'semantic': '#98FB98',     # PaleGreen
        'insight': '#FFB6C1',      # LightPink
        'connection': '#4169E1',   # RoyalBlue
        'spike': '#FF4500'         # OrangeRed
    }
    
    # Left: Brain Architecture
    ax1.set_title('Brain: Hierarchical Processing', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    
    # Brain layers (top-down display matching AI layers)
    brain_layers = [
        {'name': 'Cerebellum\n(Error Processing)', 'y': 8.5, 'color': colors['sensory'], 
         'function': 'Error monitoring'},
        {'name': 'Hippocampus\n(Memory)', 'y': 6, 'color': colors['episodic'],
         'function': 'Experience storage'},
        {'name': 'Prefrontal Cortex\n(Reasoning)', 'y': 3.5, 'color': colors['semantic'],
         'function': 'Graph reasoning'},
        {'name': 'Broca\'s/Wernicke\'s\n(Language)', 'y': 1, 'color': colors['insight'], 
         'function': 'Language output'}
    ]
    
    # Draw brain layers
    for i, layer in enumerate(brain_layers):
        # Main box
        box = FancyBboxPatch((1, layer['y']-0.4), 8, 0.8,
                            boxstyle="round,pad=0.1",
                            facecolor=layer['color'],
                            edgecolor='black',
                            linewidth=2)
        ax1.add_patch(box)
        ax1.text(5, layer['y'], layer['name'], ha='center', va='center',
                fontsize=11, fontweight='bold')
        ax1.text(5, layer['y']-0.6, layer['function'], ha='center', va='top',
                fontsize=9, style='italic')
        
        # Connections (top-down arrows)
        if i < len(brain_layers) - 1:
            arrow = FancyArrowPatch((5, layer['y']-0.4), (5, brain_layers[i+1]['y']+0.4),
                                  connectionstyle="arc3,rad=0",
                                  arrowstyle='->',
                                  mutation_scale=20,
                                  color=colors['connection'],
                                  linewidth=2)
            ax1.add_patch(arrow)
    
    # Hippocampal replay annotation (moved to hippocampus position)
    replay_box = FancyBboxPatch((9.2, 5.5), 1.5, 1,
                               boxstyle="round,pad=0.05",
                               facecolor='yellow',
                               edgecolor='red',
                               linewidth=1.5,
                               alpha=0.8)
    ax1.add_patch(replay_box)
    ax1.text(9.95, 6, 'Replay\n+\nPruning', ha='center', va='center',
            fontsize=8, color='red', fontweight='bold')
    
    # Right: InsightSpike-AI Architecture
    ax2.set_title('InsightSpike-AI: Computational Implementation', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    
    # AI layers (top-down display, Layer 1-4)
    ai_layers = [
        {'name': 'Layer 1\n(Error Monitor)', 'y': 8.5, 'color': colors['sensory'],
         'function': 'Error detection'},
        {'name': 'Layer 2\n(Memory Manager)', 'y': 6, 'color': colors['episodic'],
         'function': 'Knowledge storage'},
        {'name': 'Layer 3\n(Graph Reasoner)', 'y': 3.5, 'color': colors['semantic'],
         'function': 'ΔEPC - λ·ΔIG'},
        {'name': 'Layer 4\n(Language Interface)', 'y': 1, 'color': colors['insight'],
         'function': 'Prompt → LLM decoding'}
    ]
    
    # Draw AI layers
    for i, layer in enumerate(ai_layers):
        # Main box
        box = FancyBboxPatch((1, layer['y']-0.4), 8, 0.8,
                            boxstyle="round,pad=0.1",
                            facecolor=layer['color'],
                            edgecolor='black',
                            linewidth=2)
        ax2.add_patch(box)
        ax2.text(5, layer['y'], layer['name'], ha='center', va='center',
                fontsize=11, fontweight='bold')
        ax2.text(5, layer['y']-0.6, layer['function'], ha='center', va='top',
                fontsize=9, style='italic')
        
        # Connections (top-down arrows)
        if i < len(ai_layers) - 1:
            arrow = FancyArrowPatch((5, layer['y']-0.4), (5, ai_layers[i+1]['y']+0.4),
                                  connectionstyle="arc3,rad=0",
                                  arrowstyle='->',
                                  mutation_scale=20,
                                  color=colors['connection'],
                                  linewidth=2)
            ax2.add_patch(arrow)
    
    # Spike detection annotation (at Layer 3 - Graph Reasoner)
    spike_burst = []
    for angle in np.linspace(0, 2*np.pi, 16):
        x = 9.5 + 0.3 * np.cos(angle)
        y = 3.5 + 0.3 * np.sin(angle)
        spike_burst.append(FancyArrowPatch((9.5, 3.5), (x, y),
                                         arrowstyle='-',
                                         color=colors['spike'],
                                         linewidth=2))
    for spike in spike_burst:
        ax2.add_patch(spike)
    ax2.text(9.5, 3.5, 'SPIKE!', ha='center', va='center',
            fontsize=10, color='white', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['spike']))
    
    # Add LLM decoding component
    llm_box = FancyBboxPatch((1, -1), 8, 0.8,
                            boxstyle="round,pad=0.1",
                            facecolor='#E6E6FA',  # Lavender
                            edgecolor='black',
                            linewidth=2)
    ax2.add_patch(llm_box)
    ax2.text(5, -0.6, 'LLM Decoder', ha='center', va='center',
            fontsize=11, fontweight='bold')
    ax2.text(5, -1.2, 'Prompt generation → Response synthesis', ha='center', va='top',
            fontsize=9, style='italic')
    
    # Arrow from Layer 4 to LLM
    arrow_to_llm = FancyArrowPatch((5, 0.6), (5, -0.2),
                                  connectionstyle="arc3,rad=0",
                                  arrowstyle='->',
                                  mutation_scale=20,
                                  color=colors['connection'],
                                  linewidth=2)
    ax2.add_patch(arrow_to_llm)
    
    # Extend y-axis limits to show LLM
    ax2.set_ylim(-2, 10)
    
    # Key insights box at bottom
    insight_text = """Key Insights:
• Top-down processing: From integration to perception (insight-driven search)
• Hippocampal replay ≈ Graph structure reorganization (ΔEPC)
• Synaptic pruning ≈ Information compression (ΔIG)
• LLM decodes detected insights into natural language responses"""
    
    fig.text(0.5, 0.02, insight_text, ha='center', va='bottom',
            fontsize=10, bbox=dict(boxstyle="round,pad=0.5", 
            facecolor='lightyellow', edgecolor='gray'))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig('brain_ai_analogy_diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("Brain-AI analogy diagram saved as brain_ai_analogy_diagram.png")

if __name__ == "__main__":
    create_brain_analogy_diagram()