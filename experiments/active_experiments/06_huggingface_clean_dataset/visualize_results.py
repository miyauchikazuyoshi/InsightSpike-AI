#!/usr/bin/env python3
"""
Visualize learning progress and generate insight episode summary
===============================================================
"""

import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from datetime import datetime

# Set style
plt.style.use('default')
sns.set_palette("husl")


def load_latest_results():
    """Load the most recent experiment results."""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    result_files = list(script_dir.glob('results_*.json'))
    if not result_files:
        raise FileNotFoundError("No result files found")
    
    latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
    print(f"Loading results from: {latest_file}")
    
    with open(latest_file, 'r') as f:
        return json.load(f)


def create_progress_visualization(results):
    """Create visualization of episodes, nodes, and edges over time."""
    metrics = results['metrics']
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Documents processed (x-axis)
    documents = metrics['timestamps']
    
    # Plot 1: Episodes
    ax1.plot(documents, metrics['episodes'], 'b-', linewidth=2, marker='o', markersize=4)
    ax1.set_ylabel('Number of Episodes', fontsize=12)
    ax1.set_title('Learning Progress: Episodes, Nodes, and Edges', fontsize=16, pad=20)
    ax1.grid(True, alpha=0.3)
    ax1.fill_between(documents, metrics['episodes'], alpha=0.3)
    
    # Add integration points (where episodes < documents)
    for i, (docs, eps) in enumerate(zip(documents, metrics['episodes'])):
        if i > 0 and eps < docs:
            integration_rate = 1 - (eps / docs)
            if integration_rate > 0.1:  # Significant integration
                ax1.annotate(f'Integration\n{integration_rate:.1%}', 
                           xy=(docs, eps), 
                           xytext=(docs, eps + 50),
                           arrowprops=dict(arrowstyle='->', color='red', alpha=0.5),
                           fontsize=9, ha='center')
    
    # Plot 2: Graph Nodes
    ax2.plot(documents, metrics['graph_nodes'], 'g-', linewidth=2, marker='s', markersize=4)
    ax2.set_ylabel('Number of Nodes', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.fill_between(documents, metrics['graph_nodes'], alpha=0.3)
    
    # Plot 3: Graph Edges
    ax3.plot(documents, metrics['graph_edges'], 'r-', linewidth=2, marker='^', markersize=4)
    ax3.set_ylabel('Number of Edges', fontsize=12)
    ax3.set_xlabel('Documents Processed', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.fill_between(documents, metrics['graph_edges'], alpha=0.3)
    
    # Add edge density annotation
    for i in range(0, len(documents), 3):  # Every 3rd point
        if metrics['graph_nodes'][i] > 1:
            density = metrics['graph_edges'][i] / (metrics['graph_nodes'][i] * (metrics['graph_nodes'][i] - 1))
            ax3.text(documents[i], metrics['graph_edges'][i], 
                    f'{density:.2%}', 
                    fontsize=8, ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save the figure
    script_dir = Path(__file__).parent
    output_file = script_dir / 'learning_progress_viz.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to: {output_file}")
    plt.close()


def create_growth_rate_visualization(results):
    """Visualize growth rates and efficiency metrics."""
    metrics = results['metrics']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Calculate growth rates
    documents = metrics['timestamps']
    edge_growth_rate = []
    node_growth_rate = []
    
    for i in range(1, len(documents)):
        doc_diff = documents[i] - documents[i-1]
        if doc_diff > 0:
            edge_growth = (metrics['graph_edges'][i] - metrics['graph_edges'][i-1]) / doc_diff
            node_growth = (metrics['graph_nodes'][i] - metrics['graph_nodes'][i-1]) / doc_diff
            edge_growth_rate.append(edge_growth)
            node_growth_rate.append(node_growth)
        else:
            edge_growth_rate.append(0)
            node_growth_rate.append(0)
    
    # Plot growth rates
    ax1.plot(documents[1:], edge_growth_rate, 'r-', label='Edge Growth Rate', linewidth=2)
    ax1.plot(documents[1:], node_growth_rate, 'g-', label='Node Growth Rate', linewidth=2)
    ax1.set_ylabel('Growth Rate (per document)', fontsize=12)
    ax1.set_title('Growth Rate Analysis', fontsize=16, pad=20)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot efficiency: edges per node
    edges_per_node = [e/max(n, 1) for e, n in zip(metrics['graph_edges'], metrics['graph_nodes'])]
    ax2.plot(documents, edges_per_node, 'purple', linewidth=2, marker='o', markersize=4)
    ax2.set_ylabel('Average Edges per Node', fontsize=12)
    ax2.set_xlabel('Documents Processed', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Max neighbors (top-k=50)')
    ax2.legend()
    
    plt.tight_layout()
    
    # Save the figure
    script_dir = Path(__file__).parent
    output_file = script_dir / 'growth_rate_viz.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved growth rate visualization to: {output_file}")
    plt.close()


def analyze_insight_episodes(results):
    """Analyze episodes with high insight rewards and create CSV summary."""
    # Since we don't have actual insight rewards in the current data,
    # we'll simulate based on graph metrics changes
    
    metrics = results['metrics']
    documents = metrics['timestamps']
    
    insight_episodes = []
    
    # Analyze significant changes in graph structure
    for i in range(1, len(documents)):
        # Calculate metrics changes
        doc_range = f"{documents[i-1]+1}-{documents[i]}"
        
        edge_change = metrics['graph_edges'][i] - metrics['graph_edges'][i-1]
        node_change = metrics['graph_nodes'][i] - metrics['graph_nodes'][i-1]
        episode_change = metrics['episodes'][i] - metrics['episodes'][i-1]
        
        # Integration rate
        docs_added = documents[i] - documents[i-1]
        integration_rate = 1 - (episode_change / docs_added) if docs_added > 0 else 0
        
        # Calculate pseudo-insight score based on graph changes
        if node_change > 0:
            avg_new_connections = edge_change / node_change
            
            # High insight if: many new connections OR high integration
            insight_score = (avg_new_connections / 10) + (integration_rate * 2)
            
            if insight_score > 0.5 or integration_rate > 0.2:
                insight_episodes.append({
                    'document_range': doc_range,
                    'documents_processed': documents[i],
                    'new_episodes': episode_change,
                    'new_nodes': node_change,
                    'new_edges': edge_change,
                    'avg_connections_per_node': round(avg_new_connections, 2),
                    'integration_rate': round(integration_rate, 3),
                    'insight_score': round(insight_score, 3),
                    'episode_type': 'Integration' if integration_rate > 0.2 else 'High Connectivity'
                })
    
    # Create DataFrame
    df = pd.DataFrame(insight_episodes)
    
    if not df.empty:
        # Sort by insight score
        df = df.sort_values('insight_score', ascending=False)
        
        # Save to CSV
        script_dir = Path(__file__).parent
        csv_file = script_dir / 'insight_episodes_summary.csv'
        df.to_csv(csv_file, index=False)
        print(f"\nSaved insight episodes summary to: {csv_file}")
        
        # Display top insights
        print("\nTop 10 Insight Episodes:")
        print(df.head(10).to_string(index=False))
        
        # Create visualization of insight episodes
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Bar plot of insight scores
        top_insights = df.head(20)
        x = range(len(top_insights))
        colors = ['red' if t == 'Integration' else 'blue' for t in top_insights['episode_type']]
        
        bars = ax.bar(x, top_insights['insight_score'], color=colors, alpha=0.7)
        ax.set_xlabel('Episode Rank', fontsize=12)
        ax.set_ylabel('Insight Score', fontsize=12)
        ax.set_title('Top 20 Insight Episodes by Score', fontsize=16, pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(top_insights['document_range'], rotation=45, ha='right')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.7, label='Integration'),
            Patch(facecolor='blue', alpha=0.7, label='High Connectivity')
        ]
        ax.legend(handles=legend_elements)
        
        plt.tight_layout()
        
        # Save the figure
        script_dir = Path(__file__).parent
        output_file = script_dir / 'insight_episodes_viz.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved insight episodes visualization to: {output_file}")
        plt.close()
    
    return df


def create_summary_statistics(results):
    """Create summary statistics and save to file."""
    metrics = results['metrics']
    final_state = results['final_state']
    
    summary = {
        'Experiment Summary': {
            'Total Documents': results['total_documents'],
            'Final Episodes': final_state['episodes'],
            'Final Graph Nodes': final_state['graph_nodes'],
            'Final Graph Edges': final_state['graph_edges'],
            'Integration Rate': f"{(1 - final_state['episodes']/results['total_documents'])*100:.1f}%",
            'Graph Density': f"{final_state['graph_edges']/(final_state['graph_nodes']*(final_state['graph_nodes']-1))*100:.2f}%",
            'Avg Edges per Node': f"{final_state['graph_edges']/final_state['graph_nodes']:.1f}"
        },
        'Performance Metrics': {
            'Avg Processing Time': f"{np.mean(metrics['build_times']):.3f}s",
            'Min Processing Time': f"{np.min(metrics['build_times']):.3f}s",
            'Max Processing Time': f"{np.max(metrics['build_times']):.3f}s",
            'Total Experiment Time': f"{np.sum(metrics['build_times']):.1f}s"
        }
    }
    
    # Save summary
    script_dir = Path(__file__).parent
    summary_file = script_dir / 'experiment_summary_stats.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary statistics to: {summary_file}")
    
    # Display summary
    print("\nExperiment Summary Statistics:")
    for category, stats in summary.items():
        print(f"\n{category}:")
        for key, value in stats.items():
            print(f"  {key}: {value}")


def main():
    """Run all visualizations and analysis."""
    print("=== Learning Progress Visualization ===")
    print(f"Start time: {datetime.now()}\n")
    
    try:
        # Load results
        results = load_latest_results()
        
        # Create visualizations
        create_progress_visualization(results)
        create_growth_rate_visualization(results)
        
        # Analyze insight episodes
        insight_df = analyze_insight_episodes(results)
        
        # Create summary statistics
        create_summary_statistics(results)
        
        print(f"\n✅ All visualizations completed successfully!")
        print(f"End time: {datetime.now()}")
        
    except Exception as e:
        print(f"\n❌ Visualization failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()