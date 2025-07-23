#!/usr/bin/env python3
"""
Visualize Mathematical Concept Evolution
=======================================

Creates visualizations showing how concepts evolve and memory operations occur.
"""

import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np


def create_concept_evolution_diagram(json_file: Path):
    """Create a visual diagram of concept evolution"""
    
    # Load data
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('Mathematical Concept Evolution', fontsize=16)
    
    # --- Top plot: Concept Evolution Timeline ---
    ax1.set_title('Concept Evolution Across Learning Phases')
    ax1.set_xlabel('Learning Phase')
    ax1.set_ylabel('Concepts')
    
    # Track concepts and their evolution
    concept_tracks = {}
    y_positions = {}
    y_pos = 0
    
    for concept, evolution in data['concept_evolution'].items():
        concept_tracks[concept] = evolution
        y_positions[concept] = y_pos
        y_pos += 1
    
    # Plot concept evolution
    for concept, evolution in concept_tracks.items():
        y = y_positions[concept]
        
        for i, stage in enumerate(evolution):
            phase = stage['phase']
            name = stage['stage']
            
            # Determine color based on type
            if '_basic' in name:
                color = 'lightblue'
            elif '_advanced' in name:
                color = 'lightcoral'
            elif '_unified' in name:
                color = 'lightgreen'
            else:
                color = 'lightgray'
            
            # Draw concept box
            box = FancyBboxPatch(
                (phase - 0.4, y - 0.3), 0.8, 0.6,
                boxstyle="round,pad=0.1",
                facecolor=color,
                edgecolor='black',
                linewidth=1.5
            )
            ax1.add_patch(box)
            
            # Add text
            ax1.text(phase, y, name.replace('_', '\\n'), 
                    ha='center', va='center', fontsize=8)
    
    # Draw connections for splits
    for op in data['operations']:
        if op['operation'] == 'split':
            # Find y positions
            original = op['original']
            results = op['results']
            
            if original in y_positions:
                y_orig = y_positions[original]
                
                # Draw split arrows
                for result in results:
                    base_concept = result.split('_')[0]
                    if base_concept in y_positions:
                        y_result = y_positions[base_concept]
                        
                        # Find phases
                        phase_from = 1
                        phase_to = 2
                        
                        ax1.annotate('', xy=(phase_to - 0.4, y_result),
                                   xytext=(phase_from + 0.4, y_orig),
                                   arrowprops=dict(arrowstyle='->', 
                                                 color='red',
                                                 linestyle='--',
                                                 linewidth=2))
    
    ax1.set_xlim(0, 4)
    ax1.set_ylim(-1, len(concept_tracks))
    ax1.set_xticks([1, 2, 3])
    ax1.set_xticklabels(['Elementary', 'Middle School', 'High School'])
    ax1.set_yticks(list(y_positions.values()))
    ax1.set_yticklabels(list(y_positions.keys()))
    ax1.grid(True, alpha=0.3)
    
    # --- Bottom plot: Memory Operations Over Time ---
    ax2.set_title('Memory Operations During Learning')
    ax2.set_xlabel('Episode Number')
    ax2.set_ylabel('Total Episodes in Memory')
    
    # Simulate episode count over time
    episode_counts = [5]  # Start with 5 from phase 1
    operations_timeline = []
    
    # Phase 2
    for i in range(5):
        episode_counts.append(episode_counts[-1] + 1)
        operations_timeline.append('add')
    
    # Phase 3 with splits
    for op in data['operations']:
        if op['operation'] == 'split':
            # Split increases count by 1 (remove 1, add 2)
            episode_counts.append(episode_counts[-1] + 1)
            operations_timeline.append('split')
        else:
            episode_counts.append(episode_counts[-1] + 1)
            operations_timeline.append('add')
    
    # Plot episode count
    ax2.plot(range(len(episode_counts)), episode_counts, 'b-', linewidth=2)
    
    # Mark operations
    split_shown = False
    merge_shown = False
    for i, op in enumerate(operations_timeline):
        if i + 6 < len(episode_counts):
            if op == 'split':
                ax2.scatter(i + 6, episode_counts[i + 6], color='red', s=100, 
                           marker='v', label='Split' if not split_shown else '')
                split_shown = True
            elif op == 'merge':
                ax2.scatter(i + 6, episode_counts[i + 6], color='green', s=100,
                           marker='^', label='Merge' if not merge_shown else '')
                merge_shown = True
    
    # Add phase boundaries
    ax2.axvline(x=5, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=10, color='gray', linestyle='--', alpha=0.5)
    ax2.text(2.5, max(episode_counts) - 1, 'Phase 1', ha='center')
    ax2.text(7.5, max(episode_counts) - 1, 'Phase 2', ha='center')
    ax2.text(12, max(episode_counts) - 1, 'Phase 3', ha='center')
    
    ax2.set_xlim(0, len(episode_counts))
    ax2.set_ylim(4, max(episode_counts) + 1)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Save figure
    output_file = json_file.parent / f"concept_evolution_visualization.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n📊 Visualization saved to: {output_file}")
    
    # Also create a text-based visualization
    create_text_visualization(data, json_file.parent)


def create_text_visualization(data, output_dir):
    """Create a text-based visualization for terminal display"""
    
    output_file = output_dir / "concept_evolution_tree.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Mathematical Concept Evolution Tree\n")
        f.write("=" * 50 + "\n\n")
        
        for concept, evolution in data['concept_evolution'].items():
            f.write(f"{concept.upper()}\n")
            
            for i, stage in enumerate(evolution):
                phase = stage['phase']
                name = stage['stage']
                explanation = stage['explanation']
                
                # Indentation based on phase
                indent = "  " * (phase - 1)
                
                # Symbol based on type
                if '_basic' in name:
                    symbol = "📘"  # Basic understanding
                elif '_advanced' in name:
                    symbol = "📕"  # Advanced understanding
                elif '_unified' in name:
                    symbol = "📗"  # Unified understanding
                else:
                    symbol = "📙"  # Original
                
                f.write(f"{indent}{symbol} Phase {phase}: {name}\n")
                f.write(f"{indent}   → {explanation}\n")
                
                # Show connections
                if i < len(evolution) - 1:
                    f.write(f"{indent}   |\n")
            
            f.write("\n")
        
        # Operations summary
        f.write("\nMemory Operations Summary\n")
        f.write("-" * 30 + "\n")
        
        for op_type, count in data['summary']['operation_counts'].items():
            f.write(f"{op_type.capitalize()}: {count}\n")
        
        f.write("\nDetailed Operations:\n")
        for op in data['operations']:
            if op['operation'] == 'split':
                f.write(f"\n🔀 SPLIT: {op['original']}\n")
                f.write(f"   → {op['results'][0]} (basic)\n")
                f.write(f"   → {op['results'][1]} (advanced)\n")
                f.write(f"   Reason: {op['reason']}\n")
    
    print(f"\n📄 Text visualization saved to: {output_file}")


if __name__ == "__main__":
    # Find the latest JSON log
    results_dir = Path(__file__).parent.parent / "results"
    json_files = list(results_dir.glob("math_evolution_*.json"))
    
    if json_files:
        latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
        print(f"Visualizing: {latest_file}")
        create_concept_evolution_diagram(latest_file)
    else:
        print("No JSON log files found. Run the experiment first!")