#!/usr/bin/env python3
"""
Create detailed evolution diagrams showing what was added to concepts
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as patches


def create_detailed_evolution_diagram():
    """Create a detailed diagram showing concept evolution"""
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle('Mathematical Concept Evolution: What Was Added?', fontsize=16, fontweight='bold')
    
    # Define colors
    colors = {
        'basic': '#E8F4FD',      # Light blue
        'added': '#FFE5B4',      # Peach
        'advanced': '#E8BBE8',   # Light purple
        'arrow': '#4169E1'       # Royal blue
    }
    
    # --- Multiplication Evolution ---
    ax1 = axes[0]
    ax1.set_title('Multiplication Evolution', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 5)
    ax1.axis('off')
    
    # Basic understanding
    basic_mult = FancyBboxPatch((0.5, 2), 3, 2, 
                                boxstyle="round,pad=0.1",
                                facecolor=colors['basic'],
                                edgecolor='black',
                                linewidth=2)
    ax1.add_patch(basic_mult)
    ax1.text(2, 3.5, '基本理解', ha='center', fontweight='bold')
    ax1.text(2, 3, '同じ数を何回も足す', ha='center')
    ax1.text(2, 2.5, '3×4 = 3+3+3+3', ha='center', fontfamily='monospace')
    
    # What was added
    added_mult = FancyBboxPatch((4.5, 1.5), 3, 3,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['added'],
                                edgecolor='black',
                                linewidth=2,
                                linestyle='--')
    ax1.add_patch(added_mult)
    ax1.text(6, 4, '追加された視点', ha='center', fontweight='bold')
    ax1.text(6, 3.5, '• スケーリング概念', ha='center')
    ax1.text(6, 3, '• 連続性（小数倍）', ha='center')
    ax1.text(6, 2.5, '• 逆操作（×-1）', ha='center')
    ax1.text(6, 2, '• 幾何学的解釈', ha='center')
    
    # Advanced understanding
    adv_mult = FancyBboxPatch((8, 2), 3, 2,
                              boxstyle="round,pad=0.1",
                              facecolor=colors['advanced'],
                              edgecolor='black',
                              linewidth=2)
    ax1.add_patch(adv_mult)
    ax1.text(9.5, 3.5, '発展理解', ha='center', fontweight='bold')
    ax1.text(9.5, 3, 'スケーリング操作', ha='center')
    ax1.text(9.5, 2.5, '2倍、0.5倍、-1倍', ha='center', fontfamily='monospace')
    
    # Arrows
    arrow1 = FancyArrowPatch((3.5, 3), (4.5, 3),
                            connectionstyle="arc3,rad=0", 
                            arrowstyle='->', 
                            mutation_scale=20,
                            color=colors['arrow'],
                            linewidth=2)
    ax1.add_patch(arrow1)
    
    arrow2 = FancyArrowPatch((7.5, 3), (8, 3),
                            connectionstyle="arc3,rad=0",
                            arrowstyle='->',
                            mutation_scale=20,
                            color=colors['arrow'],
                            linewidth=2)
    ax1.add_patch(arrow2)
    
    # --- Function Evolution ---
    ax2 = axes[1]
    ax2.set_title('Function Evolution', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 5)
    ax2.axis('off')
    
    # Basic
    basic_func = FancyBboxPatch((0.5, 2), 3, 2,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['basic'],
                                edgecolor='black',
                                linewidth=2)
    ax2.add_patch(basic_func)
    ax2.text(2, 3.5, '基本理解', ha='center', fontweight='bold')
    ax2.text(2, 3, 'yはxによって決まる', ha='center')
    ax2.text(2, 2.5, 'y = 2x + 1', ha='center', fontfamily='monospace')
    
    # Added
    added_func = FancyBboxPatch((4.5, 1.5), 3, 3,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['added'],
                                edgecolor='black',
                                linewidth=2,
                                linestyle='--')
    ax2.add_patch(added_func)
    ax2.text(6, 4, '追加された視点', ha='center', fontweight='bold')
    ax2.text(6, 3.5, '• 集合論的視点', ha='center')
    ax2.text(6, 3, '• 対応関係の一般化', ha='center')
    ax2.text(6, 2.5, '• 多対一も可能', ha='center')
    ax2.text(6, 2, '• 全射・単射の概念', ha='center')
    
    # Advanced
    adv_func = FancyBboxPatch((8, 2), 3, 2,
                              boxstyle="round,pad=0.1",
                              facecolor=colors['advanced'],
                              edgecolor='black',
                              linewidth=2)
    ax2.add_patch(adv_func)
    ax2.text(9.5, 3.5, '発展理解', ha='center', fontweight='bold')
    ax2.text(9.5, 3, '集合間の対応関係', ha='center')
    ax2.text(9.5, 2.5, 'f: R → R', ha='center', fontfamily='monospace')
    
    # Arrows
    arrow3 = FancyArrowPatch((3.5, 3), (4.5, 3),
                            connectionstyle="arc3,rad=0",
                            arrowstyle='->',
                            mutation_scale=20,
                            color=colors['arrow'],
                            linewidth=2)
    ax2.add_patch(arrow3)
    
    arrow4 = FancyArrowPatch((7.5, 3), (8, 3),
                            connectionstyle="arc3,rad=0",
                            arrowstyle='->',
                            mutation_scale=20,
                            color=colors['arrow'],
                            linewidth=2)
    ax2.add_patch(arrow4)
    
    # --- Number Evolution ---
    ax3 = axes[2]
    ax3.set_title('Number Evolution', fontsize=14, fontweight='bold')
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 5)
    ax3.axis('off')
    
    # Basic
    basic_num = FancyBboxPatch((0.5, 2), 3, 2,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['basic'],
                               edgecolor='black',
                               linewidth=2)
    ax3.add_patch(basic_num)
    ax3.text(2, 3.5, '基本理解', ha='center', fontweight='bold')
    ax3.text(2, 3, '物を数えるもの', ha='center')
    ax3.text(2, 2.5, 'りんご3個', ha='center')
    
    # Added
    added_num = FancyBboxPatch((4.5, 1.5), 3, 3,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['added'],
                               edgecolor='black',
                               linewidth=2,
                               linestyle='--')
    ax3.add_patch(added_num)
    ax3.text(6, 4, '追加された視点', ha='center', fontweight='bold')
    ax3.text(6, 3.5, '• 数の拡張', ha='center')
    ax3.text(6, 3, '• 数体系の階層', ha='center')
    ax3.text(6, 2.5, '• 無限の概念', ha='center')
    ax3.text(6, 2, '• 抽象的定義', ha='center')
    
    # Advanced
    adv_num = FancyBboxPatch((8, 2), 3, 2,
                             boxstyle="round,pad=0.1",
                             facecolor=colors['advanced'],
                             edgecolor='black',
                             linewidth=2)
    ax3.add_patch(adv_num)
    ax3.text(9.5, 3.5, '発展理解', ha='center', fontweight='bold')
    ax3.text(9.5, 3, '抽象的な量の概念', ha='center')
    ax3.text(9.5, 2.5, 'ℕ⊂ℤ⊂ℚ⊂ℝ⊂ℂ', ha='center', fontfamily='monospace')
    
    # Arrows
    arrow5 = FancyArrowPatch((3.5, 3), (4.5, 3),
                            connectionstyle="arc3,rad=0",
                            arrowstyle='->',
                            mutation_scale=20,
                            color=colors['arrow'],
                            linewidth=2)
    ax3.add_patch(arrow5)
    
    arrow6 = FancyArrowPatch((7.5, 3), (8, 3),
                            connectionstyle="arc3,rad=0",
                            arrowstyle='->',
                            mutation_scale=20,
                            color=colors['arrow'],
                            linewidth=2)
    ax3.add_patch(arrow6)
    
    # Legend
    legend_elements = [
        mpatches.Patch(color=colors['basic'], label='基本理解（初期段階）'),
        mpatches.Patch(color=colors['added'], label='追加された視点'),
        mpatches.Patch(color=colors['advanced'], label='発展理解（統合後）')
    ]
    ax3.legend(handles=legend_elements, loc='lower center', ncol=3, 
               bbox_to_anchor=(0.5, -0.2), frameon=True)
    
    plt.tight_layout()
    
    # Save
    output_file = "concept_evolution_detailed.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n📊 Detailed evolution diagram saved to: {output_file}")
    
    # Create summary diagram
    create_evolution_summary()


def create_evolution_summary():
    """Create a summary diagram of evolution patterns"""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle('Concept Evolution Patterns', fontsize=16, fontweight='bold')
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, '概念進化の3つの軸', ha='center', fontsize=14, fontweight='bold')
    
    # Three axes
    axes_data = [
        {'y': 7, 'title': '1. 具体 → 抽象', 
         'examples': ['物を数える → 数学的構造', 'ピザを切る → 比率の概念']},
        {'y': 5, 'title': '2. 特殊 → 一般',
         'examples': ['整数の掛け算 → 任意の実数倍', 'y=2x+1 → f:A→B']},
        {'y': 3, 'title': '3. 孤立 → 体系',
         'examples': ['個別の数 → 数体系', '単一の式 → 関数空間']}
    ]
    
    for axis in axes_data:
        # Title box
        title_box = FancyBboxPatch((1, axis['y']-0.3), 2, 0.6,
                                   boxstyle="round,pad=0.05",
                                   facecolor='lightcoral',
                                   edgecolor='black',
                                   linewidth=2)
        ax.add_patch(title_box)
        ax.text(2, axis['y'], axis['title'], ha='center', fontweight='bold')
        
        # Arrow
        arrow = FancyArrowPatch((3.2, axis['y']), (6.8, axis['y']),
                               connectionstyle="arc3,rad=0",
                               arrowstyle='->',
                               mutation_scale=25,
                               color='darkblue',
                               linewidth=3)
        ax.add_patch(arrow)
        
        # Examples
        for i, example in enumerate(axis['examples']):
            ax.text(5, axis['y'] - 0.6 - i*0.3, example, 
                   ha='center', fontsize=9, style='italic')
    
    # Bottom note
    note_box = FancyBboxPatch((1, 0.5), 8, 1,
                              boxstyle="round,pad=0.1",
                              facecolor='lightyellow',
                              edgecolor='orange',
                              linewidth=2)
    ax.add_patch(note_box)
    ax.text(5, 1.2, '💡 洞察: 概念は更新されるのではなく、層状に蓄積される',
           ha='center', fontweight='bold')
    ax.text(5, 0.8, '初期理解は誤りではなく、発展理解の基礎となる',
           ha='center', fontsize=10)
    
    plt.tight_layout()
    
    # Save
    output_file = "evolution_patterns_summary.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Evolution patterns summary saved to: {output_file}")


if __name__ == "__main__":
    create_detailed_evolution_diagram()