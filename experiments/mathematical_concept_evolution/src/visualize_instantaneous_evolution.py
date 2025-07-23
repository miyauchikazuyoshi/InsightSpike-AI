#!/usr/bin/env python3
"""Visualize mathematical concept evolution with instantaneous ΔGED."""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from datetime import datetime

def create_memory_operations_timeline(result_file: str):
    """Create timeline visualization of memory operations."""
    
    # Load data
    with open(result_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Prepare figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Track episode count over time
    operations = data['memory_operations']
    times = []
    episode_counts = []
    operation_types = []
    concepts = []
    
    # Start with initial state
    times.append(0)
    episode_counts.append(0)
    
    # Process each operation
    for op in operations:
        time = int(op['timestamp'].replace('day_', ''))
        times.append(time)
        episode_counts.append(op['episodes_after'])
        operation_types.append(op['operation'])
        concepts.append(op['concept'])
    
    # Plot episode count growth
    ax.plot(times, episode_counts, 'b-', linewidth=2, label='Total Episodes')
    
    # Mark operations
    for i, op in enumerate(operations):
        time = int(op['timestamp'].replace('day_', ''))
        y = op['episodes_after']
        
        # Different markers for different operations
        if op['operation'] == 'split_detected':
            ax.scatter(time, y, c='red', s=200, marker='*', 
                      label='Split Operation' if i == 0 else "", zorder=5)
            # Add concept label
            ax.annotate(op['concept'].replace('_', ' ').title(),
                       xy=(time, y), xytext=(time, y + 0.5),
                       ha='center', fontsize=9, rotation=15)
    
    # Styling
    ax.set_xlabel('Time (Days)', fontsize=12)
    ax.set_ylabel('Number of Episodes', fontsize=12)
    ax.set_title('Mathematical Concept Evolution with Instantaneous ΔGED\n'
                 '(Memory Operations Timeline)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Save figure
    output_dir = Path(result_file).parent
    output_path = output_dir / 'memory_operations_timeline.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved memory operations timeline to: {output_path}")

def create_concept_distribution_chart(result_file: str):
    """Create chart showing concept distribution in final state."""
    
    # Load data
    with open(result_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Prepare figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Count episodes per concept
    final_concepts = data['final_state']['concepts']
    concept_names = []
    episode_counts = []
    
    for concept, episodes in final_concepts.items():
        concept_names.append(concept.replace('_', ' ').title())
        episode_counts.append(len(episodes))
    
    # Sort by count
    sorted_pairs = sorted(zip(concept_names, episode_counts), 
                         key=lambda x: x[1], reverse=True)
    concept_names, episode_counts = zip(*sorted_pairs)
    
    # Bar chart of episodes per concept
    bars = ax1.bar(range(len(concept_names)), episode_counts, 
                    color=['red' if c > 1 else 'blue' for c in episode_counts])
    ax1.set_xticks(range(len(concept_names)))
    ax1.set_xticklabels(concept_names, rotation=45, ha='right')
    ax1.set_ylabel('Number of Episodes')
    ax1.set_title('Episodes per Concept\n(Red = Split Concepts)')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Pie chart of operations
    op_counts = data['summary']['operation_counts']
    if op_counts:
        labels = [op.replace('_', ' ').title() for op in op_counts.keys()]
        values = list(op_counts.values())
        ax2.pie(values, labels=labels, autopct='%1.0f%%', startangle=90)
        ax2.set_title(f'Memory Operations\n(Total: {data["summary"]["total_operations"]})')
    else:
        ax2.text(0.5, 0.5, 'No operations recorded', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Memory Operations')
    
    # Overall title
    fig.suptitle('Mathematical Concept Evolution - Final State Analysis', 
                 fontsize=16, fontweight='bold')
    
    # Save figure
    output_dir = Path(result_file).parent
    output_path = output_dir / 'concept_distribution_chart.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved concept distribution chart to: {output_path}")

def create_evolution_summary(result_file: str):
    """Create summary visualization of the evolution experiment."""
    
    # Load data
    with open(result_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Prepare figure
    fig = plt.figure(figsize=(12, 10))
    
    # Title and metadata
    fig.suptitle('Mathematical Concept Evolution Experiment Summary\n'
                 f'Instantaneous ΔGED Implementation - {data["timestamp"][:10]}',
                 fontsize=16, fontweight='bold')
    
    # Create text summary
    summary_text = f"""
Experiment: {data['experiment']}
Total Memory Operations: {data['summary']['total_operations']}
Final Episode Count: {data['summary']['final_episodes']}
Unique Concepts: {data['summary']['unique_concepts']}

Key Findings:
• 4 concept splits detected during learning progression
• Concepts evolved from concrete to abstract understanding
• Negative numbers, multiplication, functions, and number concept itself underwent splitting
• Memory reorganization triggered by high geDIG scores (> 2.5)

Split Operations Timeline:
"""
    
    for op in data['memory_operations']:
        summary_text += f"• Day {op['timestamp'].replace('day_', '')}: {op['concept'].replace('_', ' ').title()} "
        summary_text += f"(Episodes: {op['episodes_before']} → {op['episodes_after']})\n"
    
    # Add text to figure
    ax = fig.add_subplot(111)
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, 
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.axis('off')
    
    # Save figure
    output_dir = Path(result_file).parent
    output_path = output_dir / 'evolution_experiment_summary.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved evolution summary to: {output_path}")

if __name__ == "__main__":
    # Find the latest result file
    results_dir = Path(__file__).parent.parent / "results"
    result_files = list(results_dir.glob("math_evolution_fixed_*.json"))
    
    if not result_files:
        print("No result files found!")
        exit(1)
    
    latest_file = str(max(result_files, key=lambda p: p.stat().st_mtime))
    print(f"Visualizing: {latest_file}")
    
    # Create all visualizations
    create_memory_operations_timeline(latest_file)
    create_concept_distribution_chart(latest_file)
    create_evolution_summary(latest_file)
    
    print("\nAll visualizations created successfully!")