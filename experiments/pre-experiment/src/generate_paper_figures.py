#!/usr/bin/env python3
"""
Generate Publication-Ready Figures
=================================

Create high-quality figures for the paper from experimental results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, Circle
import matplotlib.patches as mpatches

# Set publication style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300


def create_figure_1_conceptual_overview():
    """Figure 1: Conceptual overview of question-aware message passing"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Left: Traditional approach
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1)
    
    # Question and items
    q_pos = (0, 0.8)
    items = [(-0.5, 0.2), (0, 0.2), (0.5, 0.2)]
    
    ax1.scatter(*q_pos, s=200, c='blue', marker='o', label='Question (Q)', zorder=5)
    for i, pos in enumerate(items):
        ax1.scatter(*pos, s=150, c='orange', marker='^', label='Item' if i==0 else '', zorder=4)
    
    # Average
    avg_pos = (0, 0.3)
    ax1.scatter(*avg_pos, s=200, c='gray', marker='*', label='Average', zorder=5)
    
    # Ideal answer (far)
    d_pos = (0, -0.7)
    ax1.scatter(*d_pos, s=200, c='red', marker='s', label='Ideal (D)', zorder=5)
    
    # Arrows
    ax1.annotate('', xy=avg_pos, xytext=q_pos, 
                arrowprops=dict(arrowstyle='->', lw=1.5, color='gray', alpha=0.5))
    ax1.annotate('?', xy=d_pos, xytext=avg_pos,
                arrowprops=dict(arrowstyle='->', lw=1.5, color='gray', alpha=0.3, linestyle='dashed'))
    
    ax1.set_title('(a) Traditional Averaging')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.axis('off')
    
    # Right: Message passing approach
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-1, 1)
    
    # Same positions
    ax2.scatter(*q_pos, s=200, c='blue', marker='o', zorder=5)
    for pos in items:
        ax2.scatter(*pos, s=150, c='orange', marker='^', zorder=4)
    
    # Message passing creates X
    x_pos = (0, -0.2)
    ax2.scatter(*x_pos, s=200, c='green', marker='*', label='MP Result (X)', zorder=5)
    ax2.scatter(*d_pos, s=200, c='red', marker='s', zorder=5)
    
    # Message passing arrows
    for pos in items:
        ax2.annotate('', xy=x_pos, xytext=pos,
                    arrowprops=dict(arrowstyle='->', lw=1, color='green', alpha=0.3))
    ax2.annotate('', xy=x_pos, xytext=q_pos,
                arrowprops=dict(arrowstyle='->', lw=2, color='blue', alpha=0.5))
    
    # Closer to D
    ax2.annotate('', xy=d_pos, xytext=x_pos,
                arrowprops=dict(arrowstyle='->', lw=2, color='green', alpha=0.7))
    
    ax2.set_title('(b) Question-Aware Message Passing')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('results/figures/figure1_conceptual_overview.pdf', bbox_inches='tight')
    plt.savefig('results/figures/figure1_conceptual_overview.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_figure_2_main_results():
    """Figure 2: Main experimental results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # Data from experiments
    # Plot 1: Scaling effect
    ax1 = axes[0, 0]
    items = [3, 7, 10]
    x_to_d = [0.779, 0.810, 0.813]
    x_to_q = [0.902, 0.834, 0.817]
    
    ax1.plot(items, x_to_d, 'ro-', linewidth=2, markersize=8, label='X↔D')
    ax1.plot(items, x_to_q, 'bo-', linewidth=2, markersize=8, label='X↔Q')
    
    ax1.set_xlabel('Number of Knowledge Items')
    ax1.set_ylabel('Cosine Similarity')
    ax1.set_title('(a) Scaling Effect')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(items)
    ax1.set_ylim(0.75, 0.95)
    
    # Plot 2: X trajectory in Q-D space
    ax2 = axes[0, 1]
    
    # Standard vs MP
    standard = (0.90, 0.77)
    mp_3 = (0.902, 0.779)
    mp_7 = (0.834, 0.810)
    mp_10 = (0.817, 0.813)
    
    ax2.scatter(1, 0, s=200, c='blue', marker='o', label='Q', edgecolors='black', linewidth=2)
    ax2.scatter(0, 1, s=200, c='red', marker='s', label='D', edgecolors='black', linewidth=2)
    
    ax2.scatter(*standard, s=150, c='gray', marker='*', label='Standard', alpha=0.7)
    ax2.scatter(*mp_3, s=150, c='green', marker='*', label='MP-3', alpha=0.8)
    ax2.scatter(*mp_7, s=150, c='green', marker='*', label='MP-7', alpha=0.9)
    ax2.scatter(*mp_10, s=150, c='green', marker='*', label='MP-10')
    
    # Trajectory
    trajectory = [standard, mp_3, mp_7, mp_10]
    x_coords = [p[0] for p in trajectory]
    y_coords = [p[1] for p in trajectory]
    ax2.plot(x_coords, y_coords, 'k--', alpha=0.3)
    
    ax2.plot([0, 1], [0, 1], 'gray', alpha=0.2, linestyle='--')
    ax2.set_xlabel('Similarity to Q')
    ax2.set_ylabel('Similarity to D')
    ax2.set_title('(b) X Position in Q-D Space')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0.5, 1.05)
    ax2.set_ylim(0.5, 1.05)
    
    # Plot 3: Convergence analysis
    ax3 = axes[1, 0]
    
    iterations = range(1, 6)
    conv_3 = [0.779, 0.779, 0.779, 0.779, 0.779]
    conv_7 = [0.810, 0.810, 0.810, 0.810, 0.810]
    conv_10 = [0.813, 0.813, 0.813, 0.813, 0.813]
    
    ax3.plot(iterations, conv_3, 'o-', label='3 items', linewidth=2)
    ax3.plot(iterations, conv_7, 's-', label='7 items', linewidth=2)
    ax3.plot(iterations, conv_10, '^-', label='10 items', linewidth=2)
    
    ax3.set_xlabel('Message Passing Iteration')
    ax3.set_ylabel('X↔D Similarity')
    ax3.set_title('(c) Convergence Speed')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0.77, 0.82)
    
    # Plot 4: Summary statistics
    ax4 = axes[1, 1]
    
    methods = ['Standard', 'MP-3', 'MP-7', 'MP-10']
    improvements = [0, 0.779-0.744, 0.810-0.744, 0.813-0.744]
    
    bars = ax4.bar(methods, improvements, alpha=0.8)
    bars[0].set_color('gray')
    bars[1].set_color('lightgreen')
    bars[2].set_color('green')
    bars[3].set_color('darkgreen')
    
    ax4.set_ylabel('Improvement in X↔D')
    ax4.set_title('(d) Performance Gains')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim(-0.01, 0.08)
    
    # Add value labels
    for bar, val in zip(bars, improvements):
        if val > 0:
            ax4.text(bar.get_x() + bar.get_width()/2, val + 0.002,
                    f'+{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('results/figures/figure2_main_results.pdf', bbox_inches='tight')
    plt.savefig('results/figures/figure2_main_results.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_figure_3_vector_space_analysis():
    """Figure 3: Vector space semantic validity"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Left: Heatmap of similarities
    test_cases = ['Newton/Apple', 'Photosynthesis', 'Gravity']
    components = ['Q', 'A', 'B', 'C', 'X', 'D']
    
    # Similarity matrix (example data)
    sim_matrix = np.array([
        [1.0, 0.7, 0.6, 0.6, 0.9, 0.4],  # Q
        [0.7, 1.0, 0.5, 0.5, 0.8, 0.5],  # A
        [0.6, 0.5, 1.0, 0.4, 0.7, 0.6],  # B
        [0.6, 0.5, 0.4, 1.0, 0.7, 0.6],  # C
        [0.9, 0.8, 0.7, 0.7, 1.0, 0.8],  # X
        [0.4, 0.5, 0.6, 0.6, 0.8, 1.0],  # D
    ])
    
    sns.heatmap(sim_matrix, 
                xticklabels=components,
                yticklabels=components,
                annot=True,
                fmt='.2f',
                cmap='YlOrRd',
                ax=ax1,
                vmin=0, vmax=1,
                cbar_kws={'label': 'Cosine Similarity'})
    
    ax1.set_title('(a) Similarity Matrix Example')
    
    # Right: Semantic interpolation
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    
    # Show interpolation
    points = np.linspace(0, 1, 11)
    semantics = ['Pure Question', '', '', '', '', 'Integrated\nKnowledge', '', '', '', '', 'Ideal Answer']
    
    for i, (x, label) in enumerate(zip(points, semantics)):
        color = plt.cm.RdBu_r(x)
        ax2.scatter(x, 0.5, s=100, c=[color], zorder=5)
        if label:
            ax2.text(x, 0.3, label, ha='center', va='top', fontsize=9)
    
    # Add X position
    x_pos = 0.7
    ax2.scatter(x_pos, 0.5, s=300, c='green', marker='*', zorder=6)
    ax2.annotate('X', (x_pos, 0.5), xytext=(x_pos, 0.7), 
                ha='center', fontsize=12, weight='bold',
                arrowprops=dict(arrowstyle='->', lw=2))
    
    ax2.set_xlabel('Semantic Space (Conceptual)')
    ax2.set_yticks([])
    ax2.set_title('(b) Semantic Interpolation')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('results/figures/figure3_vector_space.pdf', bbox_inches='tight')
    plt.savefig('results/figures/figure3_vector_space.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_table_1_summary():
    """Create summary statistics table"""
    
    import pandas as pd
    
    data = {
        'Configuration': ['Baseline', '3 Items', '7 Items', '10 Items'],
        'X↔Q': [0.900, 0.902, 0.834, 0.817],
        'X↔D': [0.744, 0.779, 0.810, 0.813],
        'X↔Items': [0.820, 0.840, 0.757, 0.697],
        'Improvement': ['—', '+3.5%', '+6.6%', '+6.9%']
    }
    
    df = pd.DataFrame(data)
    
    # Save as LaTeX
    latex = df.to_latex(index=False, float_format="%.3f", 
                       caption="Summary of experimental results",
                       label="tab:summary")
    
    with open('results/figures/table1_summary.tex', 'w') as f:
        f.write(latex)
    
    # Also save as CSV
    df.to_csv('results/data/table1_summary.csv', index=False)
    
    print("Table 1 saved as LaTeX and CSV")


def main():
    """Generate all figures"""
    
    print("Generating publication-ready figures...")
    
    create_figure_1_conceptual_overview()
    print("✓ Figure 1: Conceptual overview")
    
    create_figure_2_main_results()
    print("✓ Figure 2: Main results")
    
    create_figure_3_vector_space_analysis()
    print("✓ Figure 3: Vector space analysis")
    
    create_table_1_summary()
    print("✓ Table 1: Summary statistics")
    
    print("\nAll figures generated successfully!")
    print("Check results/figures/ directory")


if __name__ == "__main__":
    main()