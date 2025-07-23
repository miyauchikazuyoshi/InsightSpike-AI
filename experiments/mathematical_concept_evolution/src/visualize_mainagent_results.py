#!/usr/bin/env python3
"""
Visualize MainAgent experiment results
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime

def create_mainagent_visualization():
    """Create visualization of MainAgent experiment results"""
    
    # Load the results
    from pathlib import Path
    results_dir = Path(__file__).parent.parent / "results"
    results_file = results_dir / "math_evolution_fixed_20250723_161003.json"
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('MainAgent Mathematical Concept Evolution Experiment', fontsize=16, fontweight='bold')
    
    # --- Top plot: Episode count evolution ---
    ax1.set_title('Episode Count Evolution with Split Operations', fontsize=14)
    
    episode_history = data['episode_history']
    concepts = [entry['concept'] for entry in episode_history]
    before_counts = [entry['before'] for entry in episode_history]
    after_counts = [entry['after'] for entry in episode_history]
    operations = [entry['operation'] for entry in episode_history]
    
    # Plot the evolution
    x_pos = range(len(episode_history))
    ax1.plot(x_pos, after_counts, 'b-', linewidth=2, marker='o', markersize=8, label='Episode Count')
    
    # Highlight splits
    for i, op in enumerate(operations):
        if op == 'split_detected':
            ax1.axvline(x=i, color='red', linestyle='--', alpha=0.5)
            ax1.text(i, after_counts[i] + 0.5, 'SPLIT', rotation=90, 
                    verticalalignment='bottom', fontweight='bold', color='red')
    
    ax1.set_xlabel('Learning Steps')
    ax1.set_ylabel('Total Episodes in Memory')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(concepts, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # --- Bottom plot: Concept distribution ---
    ax2.set_title('Final Concept Distribution (Split vs Non-Split)', fontsize=14)
    
    final_state = data['final_state']
    concepts_data = final_state['concepts']
    
    # Categorize concepts
    split_concepts = []
    normal_concepts = []
    
    for concept, episodes in concepts_data.items():
        if len(episodes) > 1:
            split_concepts.append(concept)
        else:
            normal_concepts.append(concept)
    
    # Create bar chart
    y_pos = range(len(concepts_data))
    concept_names = list(concepts_data.keys())
    episode_counts = [len(episodes) for episodes in concepts_data.values()]
    colors = ['red' if name in split_concepts else 'blue' for name in concept_names]
    
    bars = ax2.barh(y_pos, episode_counts, color=colors, alpha=0.7)
    
    # Add value labels
    for i, (bar, count) in enumerate(zip(bars, episode_counts)):
        ax2.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2, 
                str(count), va='center')
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(concept_names)
    ax2.set_xlabel('Number of Episodes')
    ax2.set_ylabel('Concept')
    ax2.set_xlim(0, max(episode_counts) + 0.5)
    
    # Add legend
    split_patch = patches.Patch(color='red', alpha=0.7, label='Split Concepts')
    normal_patch = patches.Patch(color='blue', alpha=0.7, label='Normal Concepts')
    ax2.legend(handles=[split_patch, normal_patch], loc='lower right')
    
    plt.tight_layout()
    
    # Save
    output_file = results_dir / "mainagent_experiment_visualization.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n📊 MainAgent experiment visualization saved to: {output_file}")
    
    # Create summary report
    create_summary_report(data, results_dir)

def create_summary_report(data, results_dir=None):
    """Create a text summary of the experiment"""
    
    summary = f"""
=== MainAgent Mathematical Concept Evolution Experiment Summary ===

Experiment: {data['experiment']}
Timestamp: {data['timestamp']}

Key Results:
- Total episodes evolved from 0 to {data['summary']['final_episodes']}
- {data['summary']['unique_concepts']} unique concepts tracked
- {data['summary']['total_operations']} memory operations detected

Split Operations Detected:
"""
    
    for op in data['memory_operations']:
        summary += f"  • {op['concept']}: Episode count {op['episodes_before']} → {op['episodes_after']} at {op['timestamp']}\n"
    
    summary += "\nConcepts that Split:\n"
    for concept, episodes in data['final_state']['concepts'].items():
        if len(episodes) > 1:
            summary += f"  • {concept}: {len(episodes)} episodes\n"
            for i, ep in enumerate(episodes):
                summary += f"    - Episode {i+1}: {ep['text_preview'][:60]}...\n"
    
    summary += "\nConclusion:\n"
    summary += "The MainAgent successfully demonstrated episodic memory split operations when\n"
    summary += "conflicting mathematical concepts were introduced. Advanced concepts that\n"
    summary += "conflicted with elementary understanding triggered automatic splits, preserving\n"
    summary += "both basic and advanced interpretations of concepts like 'multiplication',\n"
    summary += "'function', and 'number'.\n"
    
    # Save report
    if results_dir:
        output_file = results_dir / "mainagent_experiment_summary.txt"
    else:
        from pathlib import Path
        output_file = Path(__file__).parent.parent / "results" / "mainagent_experiment_summary.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"📄 Summary report saved to: {output_file}")
    print(summary)

if __name__ == "__main__":
    create_mainagent_visualization()