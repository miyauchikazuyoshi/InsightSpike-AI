#!/usr/bin/env python3
"""
Create geDIG framework conceptual diagram
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

def create_framework_diagram():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(6, 7.5, 'geDIG Framework: From Brain to Implementation', 
            fontsize=18, fontweight='bold', ha='center')
    
    # Colors
    colors = {
        'brain': '#FFE4B5',      # Moccasin
        'math': '#87CEEB',       # SkyBlue
        'implementation': '#98FB98',  # PaleGreen
        'arrow': '#4169E1'       # RoyalBlue
    }
    
    # 1. Brain Science (Left)
    brain_box = FancyBboxPatch((0.5, 4), 3, 2.5,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['brain'],
                               edgecolor='black',
                               linewidth=2)
    ax.add_patch(brain_box)
    ax.text(2, 5.8, 'Brain Science', fontsize=14, fontweight='bold', ha='center')
    ax.text(2, 5.3, 'Hippocampal Replay', fontsize=11, ha='center')
    ax.text(2, 4.8, '• Synaptic pruning', fontsize=9, ha='center')
    ax.text(2, 4.4, '• Prediction error reduction', fontsize=9, ha='center')
    
    # 2. Mathematical Model (Center)
    math_box = FancyBboxPatch((4.5, 4), 3, 2.5,
                              boxstyle="round,pad=0.1",
                              facecolor=colors['math'],
                              edgecolor='black',
                              linewidth=2)
    ax.add_patch(math_box)
    ax.text(6, 5.8, 'Mathematical Model', fontsize=14, fontweight='bold', ha='center')
    ax.text(6, 5.2, r'$\mathcal{F} = w_1\Delta GED - kT\cdot\Delta IG$', 
            fontsize=12, ha='center')
    ax.text(6, 4.7, 'Structural change', fontsize=9, ha='center')
    ax.text(6, 4.3, 'Information compression', fontsize=9, ha='center')
    
    # 3. Implementation (Right)
    impl_box = FancyBboxPatch((8.5, 4), 3, 2.5,
                              boxstyle="round,pad=0.1",
                              facecolor=colors['implementation'],
                              edgecolor='black',
                              linewidth=2)
    ax.add_patch(impl_box)
    ax.text(10, 5.8, 'Implementation', fontsize=14, fontweight='bold', ha='center')
    ax.text(10, 5.3, 'InsightSpike-AI', fontsize=11, ha='center', style='italic')
    ax.text(10, 4.8, '• Graph structure analysis', fontsize=9, ha='center')
    ax.text(10, 4.4, '• Real-time detection (45ms)', fontsize=9, ha='center')
    
    # Arrows
    arrow1 = FancyArrowPatch((3.5, 5.25), (4.5, 5.25),
                            connectionstyle="arc3,rad=0",
                            arrowstyle='->',
                            mutation_scale=25,
                            color=colors['arrow'],
                            linewidth=3)
    ax.add_patch(arrow1)
    ax.text(4, 5.5, 'Inspiration', fontsize=9, ha='center', color=colors['arrow'])
    
    arrow2 = FancyArrowPatch((7.5, 5.25), (8.5, 5.25),
                            connectionstyle="arc3,rad=0",
                            arrowstyle='->',
                            mutation_scale=25,
                            color=colors['arrow'],
                            linewidth=3)
    ax.add_patch(arrow2)
    ax.text(8, 5.5, 'Realization', fontsize=9, ha='center', color=colors['arrow'])
    
    # Key Innovation Box
    innovation_box = FancyBboxPatch((2, 1.5), 8, 1.5,
                                   boxstyle="round,pad=0.1",
                                   facecolor='lightyellow',
                                   edgecolor='red',
                                   linewidth=2,
                                   linestyle='--')
    ax.add_patch(innovation_box)
    ax.text(6, 2.7, 'Key Innovation', fontsize=12, fontweight='bold', ha='center', color='red')
    ax.text(6, 2.2, 'First framework to use edit-path cost (ΔEPC) as intrinsic reward', 
            fontsize=10, ha='center')
    ax.text(6, 1.8, 'Enables detection of multi-concept integration ("insights")', 
            fontsize=10, ha='center')
    
    # Results indicators
    ax.text(1, 0.8, '85% Accuracy', fontsize=10, fontweight='bold', color='green')
    ax.text(6, 0.8, '100% on Hard Questions', fontsize=10, fontweight='bold', color='green')
    ax.text(10.5, 0.8, '45ms Processing', fontsize=10, fontweight='bold', color='green')
    
    plt.tight_layout()
    plt.savefig('gedig_framework_diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("Framework diagram saved as gedig_framework_diagram.png")

def create_results_visualization():
    """Create visualization of experimental results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Accuracy by difficulty
    difficulties = ['Easy', 'Medium', 'Hard']
    accuracies = [60, 90, 100]
    colors_diff = ['#FFA07A', '#FFD700', '#32CD32']
    
    bars = ax1.bar(difficulties, accuracies, color=colors_diff, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy by Question Difficulty', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 110)
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{acc}%', ha='center', va='bottom', fontweight='bold')
    
    # Add overall accuracy line
    ax1.axhline(y=85, color='red', linestyle='--', linewidth=2, label='Overall: 85%')
    ax1.legend()
    
    # Highlight the counter-intuitive result
    ax1.annotate('Counter-intuitive:\nHarder = Better!', 
                xy=(2, 100), xytext=(1.5, 70),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, fontweight='bold', color='red',
                ha='center')
    
    # Right: Scalability
    data_sizes = [10, 20, 50, 100]
    processing_times = [39, 41, 43, 45]
    
    ax2.plot(data_sizes, processing_times, 'b-o', linewidth=3, markersize=10)
    ax2.set_xlabel('Knowledge Base Size (items)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Processing Time (ms)', fontsize=12, fontweight='bold')
    ax2.set_title('Scalability: Sub-linear Growth', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add annotations
    for size, time in zip(data_sizes, processing_times):
        ax2.annotate(f'{time}ms', (size, time), textcoords="offset points",
                    xytext=(0,10), ha='center', fontsize=9)
    
    # Highlight sub-linear scaling
    ax2.text(50, 41, 'Only 15% increase\nfor 10x data!', 
            fontsize=10, fontweight='bold', color='green',
            ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('gedig_results_visualization.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("Results visualization saved as gedig_results_visualization.png")

if __name__ == "__main__":
    create_framework_diagram()
    create_results_visualization()