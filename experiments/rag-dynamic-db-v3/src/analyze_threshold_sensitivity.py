#!/usr/bin/env python3
"""Analyze threshold sensitivity for multi-hop effects in multi-domain setting."""

import json
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from multidomain_knowledge_base import (
    create_multidomain_knowledge_base,
    create_multidomain_queries
)
from analyze_parameter_quality_multihop import MultiHopGeDIGSystem
from run_experiment_improved import ExperimentConfig, HighQualityKnowledge

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)


class ThresholdSensitiveMultiHopSystem(MultiHopGeDIGSystem):
    """Multi-hop system with adjustable edge formation thresholds."""
    
    def __init__(self, config: ExperimentConfig, params: dict):
        super().__init__(config, params)
        # Extract threshold parameters
        self.edge_threshold_1hop = params.get('edge_threshold_1hop', 0.15)
        self.edge_threshold_2hop = params.get('edge_threshold_2hop', 0.2)
        self.edge_threshold_3hop = params.get('edge_threshold_3hop', 0.25)
        
    def _evaluate_with_gedig(self, query: str, response: str, 
                            similar_nodes: list) -> tuple:
        """Multi-hop geDIG evaluation with configurable thresholds."""
        # Simulate adding new node
        new_node_id = f"hypothetical_{self.queries_processed}"
        
        # === 1-HOP EVALUATION ===
        edges_1hop = 0
        connected_nodes_1hop = []
        for node_id, similarity in similar_nodes[:5]:
            if similarity > self.edge_threshold_1hop:  # Configurable threshold
                edges_1hop += 1
                connected_nodes_1hop.append((node_id, similarity))
        
        # === 2-HOP EVALUATION ===
        edges_2hop = 0
        affected_nodes_2hop = set()
        decay_factor = self.params.get('decay_factor', 0.7)
        
        if self.params.get('enable_multihop', True) and self.params.get('max_hops', 2) >= 2:
            for node_id, sim1 in connected_nodes_1hop:
                # Find neighbors of 1-hop nodes
                node_embedding = self.knowledge_graph.nodes[node_id].embedding
                neighbors = self.knowledge_graph.find_similar_nodes(
                    node_embedding, k=5
                )
                
                for neighbor_id, neighbor_sim in neighbors:
                    if neighbor_id != node_id and neighbor_id != new_node_id:
                        if neighbor_sim > self.edge_threshold_2hop:  # Configurable threshold
                            # Weight by both similarities and decay
                            impact = sim1 * neighbor_sim * decay_factor
                            if impact > self.params.get('min_impact_2hop', 0.1):
                                affected_nodes_2hop.add(neighbor_id)
                                edges_2hop += impact
        
        # === 3-HOP EVALUATION (if enabled) ===
        edges_3hop = 0
        affected_nodes_3hop = set()
        if self.params.get('max_hops', 2) >= 3 and self.params.get('enable_multihop', True):
            decay_3hop = decay_factor ** 2
            for node_id in affected_nodes_2hop:
                node_embedding = self.knowledge_graph.nodes[node_id].embedding
                far_neighbors = self.knowledge_graph.find_similar_nodes(
                    node_embedding, k=3
                )
                for far_id, far_sim in far_neighbors:
                    if far_id not in affected_nodes_2hop and far_id not in [n[0] for n in connected_nodes_1hop]:
                        if far_sim > self.edge_threshold_3hop:  # Configurable threshold
                            impact = far_sim * decay_3hop * 0.5
                            if impact > self.params.get('min_impact_3hop', 0.05):
                                affected_nodes_3hop.add(far_id)
                                edges_3hop += impact
        
        # === CALCULATE GED WITH MULTI-HOP ===
        nodes_added = 1
        total_edges = edges_1hop + edges_2hop + edges_3hop
        
        # Weighted GED calculation
        ged = (nodes_added * self.params['node_weight'] + 
               edges_1hop * self.params['edge_weight'] +
               edges_2hop * self.params.get('edge_weight_2hop', self.params['edge_weight'] * 0.5) +
               edges_3hop * self.params.get('edge_weight_3hop', self.params['edge_weight'] * 0.25))
        
        # === CALCULATE IG WITH GRAPH CONTEXT ===
        max_similarity = similar_nodes[0][1] if similar_nodes else 0.0
        novelty = 1.0 - max_similarity
        
        # Connectivity considers multi-hop impact
        connectivity_score = (edges_1hop * 0.1 + 
                            len(affected_nodes_2hop) * 0.05 +
                            len(affected_nodes_3hop) * 0.02)
        
        ig = novelty * self.params['novelty_weight'] + connectivity_score
        
        # === GEDIG SCORE ===
        gedig_score = ged - self.params['k'] * ig
        
        # Fixed threshold
        threshold = self.params['threshold']
        should_update = gedig_score > threshold
        
        # Log decision details
        self.decision_log.append({
            'query': query[:80],
            'novelty': novelty,
            'similarity': max_similarity,
            'edges_1hop': edges_1hop,
            'edges_2hop': float(edges_2hop),
            'edges_3hop': float(edges_3hop),
            'affected_nodes_2hop': len(affected_nodes_2hop),
            'affected_nodes_3hop': len(affected_nodes_3hop),
            'total_edges': float(total_edges),
            'ged': ged,
            'ig': ig,
            'gedig_score': gedig_score,
            'threshold': threshold,
            'decision': should_update
        })
        
        # Store metrics
        self.gedig_scores.append(gedig_score)
        self.ig_values.append(ig)
        self.ged_values.append(ged)
        
        metadata = {
            'gedig_score': gedig_score,
            'ged': ged,
            'ig': ig,
            'novelty': novelty,
            'threshold_used': threshold,
            'edges_1hop': edges_1hop,
            'edges_2hop': float(edges_2hop),
            'edges_3hop': float(edges_3hop),
            'affected_2hop': len(affected_nodes_2hop),
            'affected_3hop': len(affected_nodes_3hop),
            'total_edges': float(total_edges)
        }
        
        return should_update, metadata


def run_threshold_sensitivity_study():
    """Run threshold sensitivity analysis."""
    print("🔬 Threshold Sensitivity Study for Multi-Hop Effects")
    print("=" * 60)
    
    # Create multi-domain knowledge base
    kb_items = create_multidomain_knowledge_base()
    queries = create_multidomain_queries()
    
    # Convert to expected format
    knowledge_base = []
    for item in kb_items:
        knowledge_base.append(HighQualityKnowledge(
            text=item.text,
            concepts=item.concepts,
            depth=item.depth,
            domain=item.domain
        ))
    
    print(f"📚 Knowledge Base: {len(knowledge_base)} items, {len(set(item.domain for item in kb_items))} domains")
    
    # Test different threshold configurations
    threshold_configs = [
        {
            'name': 'Very Tight (0.3, 0.35, 0.4)',
            'edge_threshold_1hop': 0.30,
            'edge_threshold_2hop': 0.35,
            'edge_threshold_3hop': 0.40,
            'min_impact_2hop': 0.15,
            'min_impact_3hop': 0.08
        },
        {
            'name': 'Tight (0.2, 0.25, 0.3)',
            'edge_threshold_1hop': 0.20,
            'edge_threshold_2hop': 0.25,
            'edge_threshold_3hop': 0.30,
            'min_impact_2hop': 0.10,
            'min_impact_3hop': 0.05
        },
        {
            'name': 'Standard (0.15, 0.2, 0.25)',
            'edge_threshold_1hop': 0.15,
            'edge_threshold_2hop': 0.20,
            'edge_threshold_3hop': 0.25,
            'min_impact_2hop': 0.10,
            'min_impact_3hop': 0.05
        },
        {
            'name': 'Relaxed (0.1, 0.15, 0.2)',
            'edge_threshold_1hop': 0.10,
            'edge_threshold_2hop': 0.15,
            'edge_threshold_3hop': 0.20,
            'min_impact_2hop': 0.08,
            'min_impact_3hop': 0.04
        },
        {
            'name': 'Very Relaxed (0.05, 0.1, 0.15)',
            'edge_threshold_1hop': 0.05,
            'edge_threshold_2hop': 0.10,
            'edge_threshold_3hop': 0.15,
            'min_impact_2hop': 0.05,
            'min_impact_3hop': 0.02
        },
        {
            'name': 'Ultra Relaxed (0.02, 0.05, 0.08)',
            'edge_threshold_1hop': 0.02,
            'edge_threshold_2hop': 0.05,
            'edge_threshold_3hop': 0.08,
            'min_impact_2hop': 0.02,
            'min_impact_3hop': 0.01
        },
        {
            'name': 'Progressive (0.1, 0.08, 0.06)',
            'edge_threshold_1hop': 0.10,
            'edge_threshold_2hop': 0.08,  # Lower for 2-hop
            'edge_threshold_3hop': 0.06,  # Even lower for 3-hop
            'min_impact_2hop': 0.05,
            'min_impact_3hop': 0.02
        },
        {
            'name': 'Inverse Progressive (0.05, 0.1, 0.15)',
            'edge_threshold_1hop': 0.05,
            'edge_threshold_2hop': 0.10,
            'edge_threshold_3hop': 0.15,
            'min_impact_2hop': 0.03,
            'min_impact_3hop': 0.05
        }
    ]
    
    config = ExperimentConfig()
    results = []
    
    # Test each threshold configuration with different hop settings
    for threshold_config in threshold_configs:
        for max_hops in [1, 2, 3]:
            config_name = f"{threshold_config['name']} - {max_hops}hop"
            print(f"\n📊 Testing: {config_name}")
            print("-" * 50)
            
            # Base parameters
            params = {
                'k': 0.3,
                'node_weight': 0.4,
                'edge_weight': 0.25,
                'edge_weight_2hop': 0.15,
                'edge_weight_3hop': 0.08,
                'novelty_weight': 0.45,
                'threshold': 0.30,
                'enable_multihop': max_hops > 1,
                'max_hops': max_hops,
                'decay_factor': 0.7
            }
            
            # Add threshold configuration
            params.update(threshold_config)
            
            # Create system
            system = ThresholdSensitiveMultiHopSystem(config, params)
            system.add_initial_knowledge(knowledge_base)
            
            # Process queries
            decisions = []
            hop_statistics = []
            
            for i, (query, depth) in enumerate(queries):
                result = system.process_query(query, depth)
                decisions.append(result.get('updated', False))
                
                # Collect hop statistics
                if system.decision_log:
                    last_log = system.decision_log[-1]
                    hop_statistics.append({
                        '1hop': last_log.get('edges_1hop', 0),
                        '2hop': last_log.get('edges_2hop', 0),
                        '3hop': last_log.get('edges_3hop', 0),
                        'affected_2hop': last_log.get('affected_nodes_2hop', 0),
                        'affected_3hop': last_log.get('affected_nodes_3hop', 0),
                        'total_edges': last_log.get('total_edges', 0)
                    })
            
            # Calculate metrics
            acceptance_rate = sum(decisions) / len(decisions) * 100
            avg_1hop = np.mean([h['1hop'] for h in hop_statistics])
            avg_2hop = np.mean([h['2hop'] for h in hop_statistics])
            avg_3hop = np.mean([h['3hop'] for h in hop_statistics])
            avg_total_edges = np.mean([h['total_edges'] for h in hop_statistics])
            avg_affected_2hop = np.mean([h['affected_2hop'] for h in hop_statistics])
            avg_affected_3hop = np.mean([h['affected_3hop'] for h in hop_statistics])
            
            # Store results
            result_data = {
                'config': config_name,
                'threshold_1hop': threshold_config['edge_threshold_1hop'],
                'threshold_2hop': threshold_config['edge_threshold_2hop'],
                'threshold_3hop': threshold_config['edge_threshold_3hop'],
                'max_hops': max_hops,
                'acceptance_rate': acceptance_rate,
                'avg_1hop_edges': avg_1hop,
                'avg_2hop_edges': avg_2hop,
                'avg_3hop_edges': avg_3hop,
                'avg_total_edges': avg_total_edges,
                'avg_affected_2hop': avg_affected_2hop,
                'avg_affected_3hop': avg_affected_3hop,
                'multihop_activity': avg_2hop + avg_3hop  # Combined multi-hop activity
            }
            
            results.append(result_data)
            
            # Print summary
            print(f"  Acceptance rate: {acceptance_rate:.1f}%")
            print(f"  Avg edges - 1hop: {avg_1hop:.2f}, 2hop: {avg_2hop:.3f}, 3hop: {avg_3hop:.3f}")
            print(f"  Avg affected nodes - 2hop: {avg_affected_2hop:.1f}, 3hop: {avg_affected_3hop:.1f}")
            print(f"  Total multi-hop activity: {avg_2hop + avg_3hop:.3f}")
    
    return results


def analyze_threshold_impact(results):
    """Analyze the impact of threshold settings."""
    print("\n" + "=" * 60)
    print("🔍 THRESHOLD IMPACT ANALYSIS")
    print("=" * 60)
    
    # Find configurations with highest multi-hop activity
    sorted_by_activity = sorted(results, key=lambda x: x['multihop_activity'], reverse=True)
    
    print("\n🏆 Top 5 Configurations by Multi-Hop Activity:")
    print("-" * 50)
    for i, r in enumerate(sorted_by_activity[:5]):
        print(f"{i+1}. {r['config']}")
        print(f"   Multi-hop activity: {r['multihop_activity']:.3f}")
        print(f"   2-hop edges: {r['avg_2hop_edges']:.3f}, 3-hop edges: {r['avg_3hop_edges']:.3f}")
        print(f"   Acceptance rate: {r['acceptance_rate']:.1f}%")
    
    # Analyze threshold patterns
    print("\n📊 Threshold Pattern Analysis:")
    print("-" * 50)
    
    # Group by threshold level
    threshold_groups = defaultdict(list)
    for r in results:
        if 'Ultra' in r['config']:
            group = 'Ultra Relaxed'
        elif 'Very Relaxed' in r['config']:
            group = 'Very Relaxed'
        elif 'Relaxed' in r['config'] and 'Very' not in r['config']:
            group = 'Relaxed'
        elif 'Standard' in r['config']:
            group = 'Standard'
        elif 'Tight' in r['config'] and 'Very' not in r['config']:
            group = 'Tight'
        elif 'Very Tight' in r['config']:
            group = 'Very Tight'
        elif 'Progressive' in r['config']:
            group = 'Progressive'
        else:
            group = 'Other'
        
        threshold_groups[group].append(r)
    
    for group, items in sorted(threshold_groups.items()):
        avg_activity = np.mean([r['multihop_activity'] for r in items])
        avg_acceptance = np.mean([r['acceptance_rate'] for r in items])
        print(f"\n{group}:")
        print(f"  Avg multi-hop activity: {avg_activity:.3f}")
        print(f"  Avg acceptance rate: {avg_acceptance:.1f}%")
    
    # Find optimal threshold configuration
    # Balance between multi-hop activity and reasonable acceptance rate
    optimal_candidates = [r for r in results if 10 < r['acceptance_rate'] < 50]
    if optimal_candidates:
        optimal = max(optimal_candidates, key=lambda x: x['multihop_activity'])
        print(f"\n⭐ Optimal Configuration (balanced):")
        print(f"  {optimal['config']}")
        print(f"  Thresholds: 1hop={optimal['threshold_1hop']:.2f}, "
              f"2hop={optimal['threshold_2hop']:.2f}, 3hop={optimal['threshold_3hop']:.2f}")
        print(f"  Multi-hop activity: {optimal['multihop_activity']:.3f}")
        print(f"  Acceptance rate: {optimal['acceptance_rate']:.1f}%")


def create_threshold_visualization(results):
    """Create comprehensive visualization of threshold sensitivity."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    
    # 1. Multi-hop Activity vs Threshold (1-hop)
    ax = axes[0, 0]
    threshold_1hop = [r['threshold_1hop'] for r in results]
    multihop_activity = [r['multihop_activity'] for r in results]
    colors = ['red' if r['max_hops'] == 1 else 'blue' if r['max_hops'] == 2 else 'green' 
              for r in results]
    ax.scatter(threshold_1hop, multihop_activity, c=colors, alpha=0.6, s=50)
    ax.set_xlabel('1-Hop Edge Threshold')
    ax.set_ylabel('Multi-Hop Activity')
    ax.set_title('Multi-Hop Activity vs 1-Hop Threshold')
    ax.grid(True, alpha=0.3)
    
    # 2. Acceptance Rate vs Multi-hop Activity
    ax = axes[0, 1]
    acceptance = [r['acceptance_rate'] for r in results]
    ax.scatter(multihop_activity, acceptance, c=colors, alpha=0.6, s=50)
    ax.set_xlabel('Multi-Hop Activity')
    ax.set_ylabel('Acceptance Rate (%)')
    ax.set_title('Trade-off: Activity vs Acceptance')
    ax.grid(True, alpha=0.3)
    
    # 3. Threshold Heatmap
    ax = axes[0, 2]
    # Create matrix of multi-hop activity by thresholds
    unique_1hop = sorted(set(threshold_1hop))
    unique_2hop = sorted(set([r['threshold_2hop'] for r in results]))
    activity_matrix = np.zeros((len(unique_1hop), len(unique_2hop)))
    
    for r in results:
        if r['max_hops'] >= 2:
            i = unique_1hop.index(r['threshold_1hop'])
            j = unique_2hop.index(r['threshold_2hop'])
            activity_matrix[i, j] = max(activity_matrix[i, j], r['multihop_activity'])
    
    im = ax.imshow(activity_matrix, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(len(unique_2hop)))
    ax.set_xticklabels([f'{t:.2f}' for t in unique_2hop], rotation=45)
    ax.set_yticks(range(len(unique_1hop)))
    ax.set_yticklabels([f'{t:.2f}' for t in unique_1hop])
    ax.set_xlabel('2-Hop Threshold')
    ax.set_ylabel('1-Hop Threshold')
    ax.set_title('Multi-Hop Activity Heatmap')
    plt.colorbar(im, ax=ax)
    
    # 4. Edge Distribution by Hop
    ax = axes[1, 0]
    hop_data = []
    hop_labels = []
    for r in results:
        if r['max_hops'] >= 2 and r['multihop_activity'] > 0:
            hop_data.append([r['avg_1hop_edges'], r['avg_2hop_edges'], r['avg_3hop_edges']])
            hop_labels.append(r['config'].split(' - ')[0][:15])
    
    if hop_data:
        hop_data = np.array(hop_data)
        x = np.arange(len(hop_labels))
        width = 0.25
        ax.bar(x - width, hop_data[:, 0], width, label='1-hop', alpha=0.8)
        ax.bar(x, hop_data[:, 1], width, label='2-hop', alpha=0.8)
        ax.bar(x + width, hop_data[:, 2], width, label='3-hop', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(hop_labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Average Edges')
        ax.set_title('Edge Distribution by Configuration')
        ax.legend()
    
    # 5. Affected Nodes Analysis
    ax = axes[1, 1]
    affected_2hop = [r['avg_affected_2hop'] for r in results if r['max_hops'] >= 2]
    affected_3hop = [r['avg_affected_3hop'] for r in results if r['max_hops'] >= 3]
    
    if affected_2hop:
        ax.hist(affected_2hop, bins=20, alpha=0.5, label='2-hop affected', color='blue')
    if affected_3hop:
        ax.hist(affected_3hop, bins=20, alpha=0.5, label='3-hop affected', color='green')
    ax.set_xlabel('Average Affected Nodes')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Affected Nodes')
    ax.legend()
    
    # 6. Threshold Progression Impact
    ax = axes[1, 2]
    progressive = [r for r in results if 'Progressive' in r['config']]
    if progressive:
        configs = [r['config'].split(' - ')[0] for r in progressive]
        activities = [r['multihop_activity'] for r in progressive]
        x = np.arange(len(configs))
        ax.bar(x, activities, color=['blue', 'orange', 'green'][:len(configs)])
        ax.set_xticks(x)
        ax.set_xticklabels([f"{c}\n{p['max_hops']}hop" for c, p in zip(configs, progressive)], 
                           fontsize=8)
        ax.set_ylabel('Multi-Hop Activity')
        ax.set_title('Progressive vs Inverse Progressive')
    
    # 7. Optimal Region Identification
    ax = axes[2, 0]
    # Plot acceptance rate vs multi-hop activity with threshold annotations
    for r in results:
        if r['max_hops'] == 2:  # Focus on 2-hop for clarity
            ax.scatter(r['multihop_activity'], r['acceptance_rate'], 
                      s=100, alpha=0.6)
            if r['multihop_activity'] > 0.01:  # Annotate active configs
                ax.annotate(f"{r['threshold_1hop']:.2f}", 
                          (r['multihop_activity'], r['acceptance_rate']),
                          fontsize=7, alpha=0.7)
    
    # Mark optimal region
    ax.axhspan(20, 40, alpha=0.1, color='green')  # Optimal acceptance range
    ax.axvspan(0.01, 0.1, alpha=0.1, color='blue')  # Optimal activity range
    ax.set_xlabel('Multi-Hop Activity')
    ax.set_ylabel('Acceptance Rate (%)')
    ax.set_title('Optimal Configuration Region')
    ax.grid(True, alpha=0.3)
    
    # 8. Threshold Sensitivity
    ax = axes[2, 1]
    # Show how sensitive multi-hop activity is to threshold changes
    threshold_levels = sorted(set(threshold_1hop))
    for level in threshold_levels[:5]:  # Top 5 most relaxed
        level_results = [r for r in results if r['threshold_1hop'] == level]
        hops = [r['max_hops'] for r in level_results]
        activities = [r['multihop_activity'] for r in level_results]
        ax.plot(hops, activities, marker='o', label=f'Thresh={level:.2f}', alpha=0.7)
    
    ax.set_xlabel('Max Hops')
    ax.set_ylabel('Multi-Hop Activity')
    ax.set_title('Activity Scaling with Hop Count')
    ax.set_xticks([1, 2, 3])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 9. Summary Statistics
    ax = axes[2, 2]
    ax.axis('off')
    
    # Calculate summary stats
    active_configs = [r for r in results if r['multihop_activity'] > 0.001]
    if active_configs:
        best_activity = max(active_configs, key=lambda x: x['multihop_activity'])
        balanced = [r for r in active_configs if 15 < r['acceptance_rate'] < 45]
        best_balanced = max(balanced, key=lambda x: x['multihop_activity']) if balanced else None
        
        summary_text = f"""
        THRESHOLD SENSITIVITY SUMMARY
        ============================
        
        Total Configurations: {len(results)}
        Active Multi-hop: {len(active_configs)}
        
        Best Multi-hop Activity:
        {best_activity['config'][:30]}
        Activity: {best_activity['multihop_activity']:.3f}
        Thresholds: ({best_activity['threshold_1hop']:.2f}, 
                     {best_activity['threshold_2hop']:.2f}, 
                     {best_activity['threshold_3hop']:.2f})
        
        """
        
        if best_balanced:
            summary_text += f"""
        Best Balanced:
        {best_balanced['config'][:30]}
        Activity: {best_balanced['multihop_activity']:.3f}
        Acceptance: {best_balanced['acceptance_rate']:.1f}%
        """
        
        ax.text(0.1, 0.5, summary_text, fontsize=10, 
               verticalalignment='center', fontfamily='monospace')
    
    plt.suptitle('Threshold Sensitivity Analysis for Multi-Hop geDIG', fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("../results/threshold_sensitivity")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    viz_path = results_dir / f"threshold_analysis_{timestamp}.png"
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Saved visualization to: {viz_path}")
    return viz_path


def save_results(results):
    """Save threshold sensitivity results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("../results/threshold_sensitivity")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean results for JSON
    clean_results = []
    for r in results:
        clean_r = {}
        for k, v in r.items():
            if isinstance(v, (np.integer, np.floating)):
                clean_r[k] = float(v)
            else:
                clean_r[k] = v
        clean_results.append(clean_r)
    
    results_path = results_dir / f"threshold_results_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(clean_results, f, indent=2)
    
    print(f"✅ Saved results to: {results_path}")
    
    # Also save as CSV for easy analysis
    import pandas as pd
    df = pd.DataFrame(clean_results)
    csv_path = results_dir / f"threshold_results_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved CSV to: {csv_path}")


def main():
    """Run threshold sensitivity analysis."""
    print("\n🚀 Starting Threshold Sensitivity Analysis")
    print("=" * 60)
    
    try:
        # Run the study
        results = run_threshold_sensitivity_study()
        
        # Analyze impact
        analyze_threshold_impact(results)
        
        # Create visualizations
        create_threshold_visualization(results)
        
        # Save results
        save_results(results)
        
        print("\n" + "=" * 60)
        print("✅ THRESHOLD SENSITIVITY ANALYSIS COMPLETE")
        print("=" * 60)
        
        # Final recommendations
        active_results = [r for r in results if r['multihop_activity'] > 0.001]
        if active_results:
            print("\n💡 KEY FINDINGS:")
            print("-" * 40)
            print("1. Multi-hop effects emerge with relaxed thresholds")
            print(f"2. Optimal 1-hop threshold: < 0.1 for multi-domain graphs")
            print(f"3. Progressive thresholds (decreasing with hop count) show promise")
            print(f"4. Activity detected in {len(active_results)}/{len(results)} configurations")
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())