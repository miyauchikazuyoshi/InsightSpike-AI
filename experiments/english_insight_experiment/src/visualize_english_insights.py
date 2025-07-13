#!/usr/bin/env python3
"""
Visualize English Insight Generation Process
===========================================

Create before/after visualizations for English experiment
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Set
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch


def load_results(json_file: str) -> List[Dict]:
    """Load experiment results"""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_insight_info(results: List[Dict]) -> List[Dict]:
    """Extract insight information"""
    insights = []
    
    for result in results:
        if result.get('insightspike', {}).get('spike_detected', False):
            insight = {
                'query': result['query'],
                'response': result['insightspike']['response'],
                'confidence': result['insightspike']['confidence'],
                'context': result['insightspike']['context'],
                'new_concepts': set()
            }
            
            # Extract new concepts (simplified for English)
            response_words = set(insight['response'].lower().split())
            direct_words = set(result['direct_llm']['response'].lower().split())
            
            # Key concepts that appear in InsightSpike but not in Direct
            key_concepts = ['energy', 'information', 'entropy', 'quantum', 'consciousness', 
                           'integration', 'correlation', 'principle', 'theory']
            
            for concept in key_concepts:
                if concept in response_words and concept not in direct_words:
                    insight['new_concepts'].add(concept)
                    
            insights.append(insight)
    
    return insights


def create_before_after_graphs(insight: Dict) -> Tuple[nx.DiGraph, nx.DiGraph, Set, Set]:
    """Create before and after graphs"""
    
    # Before: existing knowledge nodes only
    G_before = nx.DiGraph()
    
    # Add source nodes
    source_nodes = []
    phase_groups = {}
    
    for i, context in enumerate(insight['context']):
        if ']' in context:
            phase = context.split(']')[0][1:]
            text = context.split(']')[1].strip()[:60] + '...'
            node_id = f"source_{i}"
            
            G_before.add_node(
                node_id,
                type='source',
                phase=phase,
                text=text
            )
            source_nodes.append(node_id)
            
            # Group by phase
            if phase not in phase_groups:
                phase_groups[phase] = []
            phase_groups[phase].append(node_id)
    
    # Add existing relationships
    phase_order = ['Basic Concepts', 'Relationships', 'Deep Integration', 
                   'Emergent Insights', 'Integration and Circulation']
    
    # Connect within phases
    for phase, nodes in phase_groups.items():
        for i in range(len(nodes) - 1):
            G_before.add_edge(nodes[i], nodes[i+1], type='same_phase')
    
    # Connect between adjacent phases
    for i in range(len(phase_order) - 1):
        curr_phase = phase_order[i]
        next_phase = phase_order[i + 1]
        
        if curr_phase in phase_groups and next_phase in phase_groups:
            if phase_groups[curr_phase] and phase_groups[next_phase]:
                G_before.add_edge(
                    phase_groups[curr_phase][-1], 
                    phase_groups[next_phase][0], 
                    type='phase_transition'
                )
    
    # After: add insight node and new edges
    G_after = G_before.copy()
    
    # Add insight node
    insight_node_id = "insight_0"
    G_after.add_node(
        insight_node_id,
        type='insight',
        text=insight['query'][:40] + '...',
        confidence=insight['confidence']
    )
    
    # Add new concept nodes
    new_concept_nodes = set()
    for concept in insight['new_concepts']:
        concept_id = f"concept_{concept}"
        G_after.add_node(
            concept_id,
            type='new_concept',
            text=concept.capitalize()
        )
        new_concept_nodes.add(concept_id)
        # Edge from insight to new concept
        G_after.add_edge(insight_node_id, concept_id, type='generates')
    
    # Edges from sources to insight (new)
    new_edges = set()
    for source_id in source_nodes:
        G_after.add_edge(source_id, insight_node_id, type='integrates_into')
        new_edges.add((source_id, insight_node_id))
    
    # Set of new nodes
    new_nodes = {insight_node_id} | new_concept_nodes
    
    return G_before, G_after, new_nodes, new_edges


def calculate_graph_metrics(G: nx.DiGraph) -> Dict:
    """Calculate graph structural metrics"""
    metrics = {}
    
    # Basic metrics
    metrics['nodes'] = G.number_of_nodes()
    metrics['edges'] = G.number_of_edges()
    metrics['density'] = nx.density(G) if G.number_of_nodes() > 1 else 0
    
    # Connectivity
    if G.number_of_nodes() > 0:
        metrics['components'] = nx.number_weakly_connected_components(G)
        G_undirected = G.to_undirected()
        metrics['clustering'] = nx.average_clustering(G_undirected)
        metrics['avg_degree'] = sum(dict(G.degree()).values()) / G.number_of_nodes()
    else:
        metrics['components'] = 0
        metrics['clustering'] = 0
        metrics['avg_degree'] = 0
    
    # Structural complexity score
    metrics['complexity_score'] = (
        0.3 * metrics['density'] + 
        0.2 * metrics['clustering'] + 
        0.3 * (metrics['avg_degree'] / max(metrics['nodes'], 1)) +
        0.2 * (1 - metrics['components'] / max(metrics['nodes'], 1))
    )
    
    return metrics


def visualize_before_after(insight: Dict, output_file: str):
    """Visualize before and after insight generation"""
    
    G_before, G_after, new_nodes, new_edges = create_before_after_graphs(insight)
    
    # Calculate metrics
    metrics_before = calculate_graph_metrics(G_before)
    metrics_after = calculate_graph_metrics(G_after)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Layout (use same positions for both)
    pos = nx.spring_layout(G_after, k=2, iterations=50, seed=42)
    
    # Color map for phases
    phase_colors = {
        'Basic Concepts': '#1f77b4',
        'Relationships': '#2ca02c',
        'Deep Integration': '#d62728',
        'Emergent Insights': '#9467bd',
        'Integration and Circulation': '#8c564b'
    }
    
    # Before graph
    ax1.set_title('Before Insight Generation', fontsize=16, pad=20)
    
    # Metrics display (Before)
    metrics_text_before = (
        f"Nodes: {metrics_before['nodes']}\n"
        f"Edges: {metrics_before['edges']}\n"
        f"Density: {metrics_before['density']:.3f}\n"
        f"Clustering: {metrics_before['clustering']:.3f}\n"
        f"Complexity: {metrics_before['complexity_score']:.3f}"
    )
    ax1.text(0.02, 0.98, metrics_text_before, 
             transform=ax1.transAxes,
             fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Draw nodes (Before)
    for node, data in G_before.nodes(data=True):
        color = phase_colors.get(data.get('phase', ''), '#7f7f7f')
        nx.draw_networkx_nodes(
            G_before, pos, [node], 
            node_color=color, 
            node_size=2000, 
            alpha=0.8, 
            ax=ax1
        )
    
    # Draw edges (Before)
    nx.draw_networkx_edges(
        G_before, pos,
        edge_color='gray',
        arrows=True,
        arrowsize=20,
        alpha=0.3,
        ax=ax1
    )
    
    # Labels (Before)
    labels = {n: d['text'] for n, d in G_before.nodes(data=True)}
    nx.draw_networkx_labels(G_before, pos, labels, font_size=8, ax=ax1)
    
    # After graph
    ax2.set_title('After Insight Generation', fontsize=16, pad=20)
    
    # Metrics display (After)
    metrics_text_after = (
        f"Nodes: {metrics_after['nodes']} (+{metrics_after['nodes'] - metrics_before['nodes']})\n"
        f"Edges: {metrics_after['edges']} (+{metrics_after['edges'] - metrics_before['edges']})\n"
        f"Density: {metrics_after['density']:.3f} ({metrics_after['density'] - metrics_before['density']:+.3f})\n"
        f"Clustering: {metrics_after['clustering']:.3f} ({metrics_after['clustering'] - metrics_before['clustering']:+.3f})\n"
        f"Complexity: {metrics_after['complexity_score']:.3f} ({metrics_after['complexity_score'] - metrics_before['complexity_score']:+.3f})"
    )
    
    # Complexity increase rate
    complexity_increase = ((metrics_after['complexity_score'] - metrics_before['complexity_score']) / 
                          max(metrics_before['complexity_score'], 0.001)) * 100
    
    ax2.text(0.02, 0.98, metrics_text_after, 
             transform=ax2.transAxes,
             fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Highlight complexity increase
    ax2.text(0.98, 0.98, f"Complexity Increase: {complexity_increase:.1f}%", 
             transform=ax2.transAxes,
             fontsize=12,
             fontweight='bold',
             color='red',
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    # Draw nodes (After)
    for node, data in G_after.nodes(data=True):
        if node in new_nodes:
            if data['type'] == 'insight':
                color = '#ff7f0e'  # Orange for insight
                size = 3000
            else:  # new_concept
                color = '#ff1493'  # Pink for new concepts
                size = 1500
            # New nodes with thick border
            nx.draw_networkx_nodes(
                G_after, pos, [node],
                node_color=color,
                node_size=size,
                alpha=0.9,
                edgecolors='black',
                linewidths=3,
                ax=ax2
            )
        else:
            color = phase_colors.get(data.get('phase', ''), '#7f7f7f')
            nx.draw_networkx_nodes(
                G_after, pos, [node],
                node_color=color,
                node_size=2000,
                alpha=0.8,
                ax=ax2
            )
    
    # Draw edges (After)
    # Old edges
    old_edges = [(u, v) for u, v in G_after.edges() if (u, v) not in new_edges]
    nx.draw_networkx_edges(
        G_after, pos, old_edges,
        edge_color='gray',
        arrows=True,
        arrowsize=20,
        alpha=0.3,
        ax=ax2
    )
    
    # New edges (thick, red)
    nx.draw_networkx_edges(
        G_after, pos, list(new_edges),
        edge_color='red',
        arrows=True,
        arrowsize=25,
        width=3,
        alpha=0.8,
        ax=ax2
    )
    
    # Labels (After)
    labels = {n: d['text'] for n, d in G_after.nodes(data=True)}
    nx.draw_networkx_labels(G_after, pos, labels, font_size=8, ax=ax2)
    
    # Legend
    legend_elements = [
        patches.Patch(color='#1f77b4', label='Basic Concepts', alpha=0.8),
        patches.Patch(color='#2ca02c', label='Relationships', alpha=0.8),
        patches.Patch(color='#d62728', label='Deep Integration', alpha=0.8),
        patches.Patch(color='#9467bd', label='Emergent Insights', alpha=0.8),
        patches.Patch(color='#8c564b', label='Integration', alpha=0.8),
        patches.Patch(color='#ff7f0e', label='Generated Insight', alpha=0.9),
        patches.Patch(color='#ff1493', label='New Concepts', alpha=0.9),
        patches.Patch(color='red', label='New Edges', alpha=0.8),
        patches.Patch(color='gray', label='Existing Edges', alpha=0.3)
    ]
    
    # Place legend at bottom center
    fig.legend(
        handles=legend_elements,
        loc='lower center',
        ncol=5,
        bbox_to_anchor=(0.5, -0.05),
        fontsize=10,
        frameon=True,
        fancybox=True,
        shadow=True
    )
    
    # Axis settings
    for ax in [ax1, ax2]:
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()
    
    print(f"✅ Saved visualization: {output_file}")


def generate_insight_report(insights: List[Dict], output_file: str = 'english_insight_report.md'):
    """Generate insight generation report"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Insight Generation Process Analysis Report\n\n")
        
        for i, insight in enumerate(insights):
            f.write(f"## Insight {i+1}: {insight['query']}\n\n")
            f.write(f"**Confidence**: {insight['confidence']:.2%}\n\n")
            
            f.write("### Integrated Knowledge Nodes\n\n")
            
            # Group by phase
            phase_groups = {}
            for context in insight['context']:
                if ']' in context:
                    phase = context.split(']')[0][1:]
                    text = context.split(']')[1].strip()
                    if phase not in phase_groups:
                        phase_groups[phase] = []
                    phase_groups[phase].append(text)
            
            for phase, texts in phase_groups.items():
                f.write(f"#### {phase} ({len(texts)} items)\n")
                for text in texts:
                    f.write(f"- {text}\n")
                f.write("\n")
            
            f.write("### Generated Insight\n\n")
            f.write(f"> {insight['response']}\n\n")
            
            if insight['new_concepts']:
                f.write("### New Concepts Emerged\n\n")
                f.write(f"{', '.join(sorted(insight['new_concepts']))}\n\n")
            
            f.write("---\n\n")
        
        # Summary statistics
        f.write("## Summary Statistics\n\n")
        f.write(f"- Total insights: {len(insights)}\n")
        f.write(f"- Average confidence: {sum(i['confidence'] for i in insights) / len(insights):.2%}\n")
        
        # Phase usage frequency
        phase_counts = {}
        for insight in insights:
            for context in insight['context']:
                if ']' in context:
                    phase = context.split(']')[0][1:]
                    phase_counts[phase] = phase_counts.get(phase, 0) + 1
        
        f.write("\n### Phase Usage Frequency\n\n")
        for phase, count in sorted(phase_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"- {phase}: {count} times\n")
    
    print(f"✅ Report saved: {output_file}")


def main():
    """Main process"""
    
    # Load experiment results
    result_file = "english_experiment_results.json"
    
    if not Path(result_file).exists():
        print(f"❌ File not found: {result_file}")
        return
    
    results = load_results(result_file)
    insights = extract_insight_info(results)
    
    if not insights:
        print("⚠️ No insights detected")
        return
    
    print(f"\n📊 Found {len(insights)} insights to visualize")
    
    # Visualize each insight
    for i, insight in enumerate(insights):
        print(f"\nVisualizing insight {i+1}: {insight['query']}")
        print(f"  Confidence: {insight['confidence']:.2%}")
        print(f"  New concepts: {', '.join(insight['new_concepts']) if insight['new_concepts'] else 'None'}")
        
        output_file = f"english_insight_before_after_{i+1}.png"
        visualize_before_after(insight, output_file)
    
    # Generate report
    generate_insight_report(insights)
    
    print("\n✅ Analysis complete!")
    print("  - Visualizations: english_insight_before_after_*.png")
    print("  - Report: english_insight_report.md")


if __name__ == "__main__":
    main()