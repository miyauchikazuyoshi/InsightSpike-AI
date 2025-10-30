#!/usr/bin/env python3
"""Analyze qualitative trends with multi-hop geDIG evaluation."""

import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from run_experiment_improved import (
    create_high_quality_knowledge_base,
    create_meaningful_queries,
    ExperimentConfig,
    EnhancedRAGSystem
)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


class MultiHopGeDIGSystem(EnhancedRAGSystem):
    """geDIG system with multi-hop evaluation."""
    
    def __init__(self, config: ExperimentConfig, params: dict):
        super().__init__("gedig", config)
        self.params = params
        self.decision_log = []
        # Force gedig_core to be truthy so parent calls our _evaluate_with_gedig
        self.gedig_core = True
        
    def _evaluate_with_gedig(self, query: str, response: str, 
                            similar_nodes: list) -> tuple:
        """Multi-hop geDIG evaluation."""
        # Simulate adding new node
        new_node_id = f"hypothetical_{self.queries_processed}"
        
        # === 1-HOP EVALUATION ===
        edges_1hop = 0
        connected_nodes_1hop = []
        for node_id, similarity in similar_nodes[:5]:
            if similarity > 0.15:  # Edge threshold
                edges_1hop += 1
                connected_nodes_1hop.append((node_id, similarity))
        
        # === 2-HOP EVALUATION ===
        edges_2hop = 0
        affected_nodes_2hop = set()
        decay_factor = self.params.get('decay_factor', 0.7)
        
        if self.params.get('enable_multihop', True):
            for node_id, sim1 in connected_nodes_1hop:
                # Find neighbors of 1-hop nodes
                node_embedding = self.knowledge_graph.nodes[node_id].embedding
                neighbors = self.knowledge_graph.find_similar_nodes(
                    node_embedding, k=5
                )
                
                for neighbor_id, neighbor_sim in neighbors:
                    if neighbor_id != node_id and neighbor_id != new_node_id:
                        if neighbor_sim > 0.2:  # 2-hop threshold
                            # Weight by both similarities and decay
                            impact = sim1 * neighbor_sim * decay_factor
                            if impact > 0.1:
                                affected_nodes_2hop.add(neighbor_id)
                                edges_2hop += impact
        
        # === 3-HOP EVALUATION (if enabled) ===
        edges_3hop = 0
        if self.params.get('max_hops', 2) >= 3 and self.params.get('enable_multihop', True):
            decay_3hop = decay_factor ** 2
            for node_id in affected_nodes_2hop:
                node_embedding = self.knowledge_graph.nodes[node_id].embedding
                far_neighbors = self.knowledge_graph.find_similar_nodes(
                    node_embedding, k=3
                )
                for far_id, far_sim in far_neighbors:
                    if far_id not in affected_nodes_2hop and far_id not in [n[0] for n in connected_nodes_1hop]:
                        impact = far_sim * decay_3hop * 0.5  # Further reduced impact
                        if impact > 0.05:
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
                            edges_3hop * 0.02)
        
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
            'ged': ged,
            'ig': ig,
            'gedig_score': gedig_score,
            'threshold': threshold,
            'decision': should_update
        })
        
        # Store metrics like parent class expects
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
            'affected_2hop': len(affected_nodes_2hop)
        }
        
        return should_update, metadata


def categorize_queries():
    """Categorize queries by their actual value."""
    test_queries = create_meaningful_queries()
    
    query_categories = []
    
    # High value queries (should definitely accept)
    high_value_indices = [
        5,   # Advanced regularization beyond L1/L2
        7,   # Transfer learning
        8,   # Vanishing gradient problem
        9,   # GANs
        11,  # LSTM vs GRU vs Transformer
        12,  # Automatic differentiation in PyTorch
        13,  # Batch normalization mathematics
        14,  # ML production deployment
        15,  # Imbalanced datasets
        18,  # State-of-the-art beyond BERT
    ]
    
    # Low value queries (should reject - already answerable)
    low_value_indices = [
        0,   # Python GIL (already in KB)
        1,   # Overfitting causes (already in KB)
        2,   # Transformers vs RNNs (already in KB)
        3,   # Gradient descent and backprop (already in KB)
    ]
    
    # Medium value queries (either is acceptable)
    medium_value_indices = [
        4,   # Overcome GIL limitations
        6,   # Attention mechanism details
        10,  # CNN vs transformer for vision
        16,  # Python memory internals
        17,  # Advanced optimization techniques
    ]
    
    for i, (query, depth) in enumerate(test_queries):
        if i in high_value_indices:
            value = 'HIGH'
            reason = 'New important concept/technique'
        elif i in low_value_indices:
            value = 'LOW'
            reason = 'Already covered in knowledge base'
        else:
            value = 'MEDIUM'
            reason = 'Extends existing knowledge'
            
        query_categories.append({
            'index': i,
            'query': query,
            'value': value,
            'reason': reason
        })
    
    return query_categories


def run_multihop_study():
    """Run study comparing 1-hop vs multi-hop evaluation."""
    print("🔬 Multi-Hop geDIG Quality Study")
    print("=" * 60)
    
    # Configurations to test with multi-hop
    configurations = [
        {
            'name': 'Conservative 1-hop only',
            'params': {
                'k': 0.5,
                'node_weight': 0.35,
                'edge_weight': 0.15,
                'novelty_weight': 0.5,
                'threshold': 0.38,
                'enable_multihop': False,
                'max_hops': 1
            }
        },
        {
            'name': 'Conservative 2-hop',
            'params': {
                'k': 0.5,
                'node_weight': 0.35,
                'edge_weight': 0.15,
                'edge_weight_2hop': 0.08,
                'novelty_weight': 0.5,
                'threshold': 0.42,  # Slightly higher for multi-hop
                'enable_multihop': True,
                'max_hops': 2,
                'decay_factor': 0.7
            }
        },
        {
            'name': 'Conservative 3-hop',
            'params': {
                'k': 0.5,
                'node_weight': 0.35,
                'edge_weight': 0.15,
                'edge_weight_2hop': 0.08,
                'edge_weight_3hop': 0.04,
                'novelty_weight': 0.5,
                'threshold': 0.45,  # Even higher for 3-hop
                'enable_multihop': True,
                'max_hops': 3,
                'decay_factor': 0.7
            }
        },
        {
            'name': 'Balanced 1-hop only',
            'params': {
                'k': 0.3,
                'node_weight': 0.4,
                'edge_weight': 0.2,
                'novelty_weight': 0.5,
                'threshold': 0.28,
                'enable_multihop': False,
                'max_hops': 1
            }
        },
        {
            'name': 'Balanced 2-hop',
            'params': {
                'k': 0.3,
                'node_weight': 0.4,
                'edge_weight': 0.2,
                'edge_weight_2hop': 0.1,
                'novelty_weight': 0.5,
                'threshold': 0.32,
                'enable_multihop': True,
                'max_hops': 2,
                'decay_factor': 0.7
            }
        },
        {
            'name': 'Liberal 1-hop only',
            'params': {
                'k': 0.15,
                'node_weight': 0.45,
                'edge_weight': 0.25,
                'novelty_weight': 0.4,
                'threshold': 0.15,
                'enable_multihop': False,
                'max_hops': 1
            }
        },
        {
            'name': 'Liberal 2-hop',
            'params': {
                'k': 0.15,
                'node_weight': 0.45,
                'edge_weight': 0.25,
                'edge_weight_2hop': 0.12,
                'novelty_weight': 0.4,
                'threshold': 0.18,
                'enable_multihop': True,
                'max_hops': 2,
                'decay_factor': 0.7
            }
        }
    ]
    
    # Get query categories
    query_categories = categorize_queries()
    high_value_indices = [q['index'] for q in query_categories if q['value'] == 'HIGH']
    low_value_indices = [q['index'] for q in query_categories if q['value'] == 'LOW']
    
    # Setup
    config = ExperimentConfig()
    knowledge_base = create_high_quality_knowledge_base()
    test_queries = create_meaningful_queries()
    
    results = []
    
    for config_data in configurations:
        print(f"\n📊 Testing: {config_data['name']}")
        print("-" * 50)
        
        # Create system with multi-hop parameters
        system = MultiHopGeDIGSystem(config, config_data['params'])
        system.add_initial_knowledge(knowledge_base)
        
        # Process all queries
        decisions = []
        for i, (query, depth) in enumerate(test_queries):
            result = system.process_query(query, depth)
            updated = result.get('updated', False)
            decisions.append(updated)
            
            # Debug print for first query of 2-hop configs
            if i == 0 and '2-hop' in config_data['name'] and 'Conservative' in config_data['name']:
                print(f"\n🔍 Debug multi-hop evaluation:")
                if system.decision_log:
                    last_log = system.decision_log[-1]
                    print(f"  1-hop edges: {last_log['edges_1hop']}")
                    print(f"  2-hop edges: {last_log['edges_2hop']:.2f}")
                    print(f"  2-hop affected nodes: {last_log['affected_nodes_2hop']}")
                    print(f"  Total GED: {last_log['ged']:.3f}")
                    print(f"  geDIG score: {last_log['gedig_score']:.3f}")
                    print(f"  Decision: {'Accept' if last_log['decision'] else 'Reject'}")
        
        # Analyze decisions by value category
        high_value_accepted = 0
        low_value_rejected = 0
        
        for i, decision in enumerate(decisions):
            cat = query_categories[i]
            if cat['value'] == 'HIGH' and decision:
                high_value_accepted += 1
            elif cat['value'] == 'LOW' and not decision:
                low_value_rejected += 1
        
        high_value_total = len(high_value_indices)
        low_value_total = len(low_value_indices)
        
        high_acceptance_rate = (high_value_accepted / high_value_total) * 100
        low_rejection_rate = (low_value_rejected / low_value_total) * 100
        
        # Overall quality score
        quality_score = (high_acceptance_rate + low_rejection_rate) / 2
        
        # Store results
        result_data = {
            'config': config_data['name'],
            'k': float(config_data['params']['k']),
            'threshold': float(config_data['params']['threshold']),
            'max_hops': config_data['params']['max_hops'],
            'multihop': config_data['params']['enable_multihop'],
            'high_value_accepted': int(high_value_accepted),
            'high_value_total': int(high_value_total),
            'high_acceptance_rate': float(high_acceptance_rate),
            'low_value_rejected': int(low_value_rejected),
            'low_value_total': int(low_value_total),
            'low_rejection_rate': float(low_rejection_rate),
            'total_accepted': int(sum(decisions)),
            'total_queries': len(decisions),
            'overall_rate': float(sum(decisions) / len(decisions) * 100),
            'quality_score': float(quality_score)
        }
        
        results.append(result_data)
        
        # Print summary
        print(f"  High-value acceptance: {high_value_accepted}/{high_value_total} ({high_acceptance_rate:.0f}%)")
        print(f"  Low-value rejection: {low_value_rejected}/{low_value_total} ({low_rejection_rate:.0f}%)")
        print(f"  Overall acceptance: {sum(decisions)}/{len(decisions)} ({result_data['overall_rate']:.0f}%)")
        print(f"  📊 Quality Score: {quality_score:.0f}%")
    
    return results


def analyze_multihop_impact(results):
    """Analyze the impact of multi-hop evaluation."""
    print("\n" + "=" * 60)
    print("📈 MULTI-HOP IMPACT ANALYSIS")
    print("=" * 60)
    
    # Group by base configuration
    conservative_results = [r for r in results if 'Conservative' in r['config']]
    balanced_results = [r for r in results if 'Balanced' in r['config']]
    liberal_results = [r for r in results if 'Liberal' in r['config']]
    
    for group_name, group_results in [
        ('Conservative', conservative_results),
        ('Balanced', balanced_results),
        ('Liberal', liberal_results)
    ]:
        if not group_results:
            continue
            
        print(f"\n{group_name} Settings:")
        print("-" * 30)
        
        for result in sorted(group_results, key=lambda x: x['max_hops']):
            hops = result['max_hops']
            quality = result['quality_score']
            acceptance = result['overall_rate']
            high_acc = result['high_acceptance_rate']
            low_rej = result['low_rejection_rate']
            
            print(f"  {hops}-hop: Quality={quality:.0f}%, "
                  f"Acceptance={acceptance:.0f}%, "
                  f"High-value={high_acc:.0f}%, "
                  f"Low-reject={low_rej:.0f}%")
    
    # Find best configuration
    best_config = max(results, key=lambda x: x['quality_score'])
    print(f"\n🏆 Best Configuration: {best_config['config']}")
    print(f"   Quality Score: {best_config['quality_score']:.0f}%")
    print(f"   Uses multi-hop: {best_config['multihop']}")


def save_results(results):
    """Save results with proper type conversion."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create results directory
    results_dir = Path("../results/multihop_quality")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON results
    results_path = results_dir / f"multihop_results_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Saved results to: {results_path}")
    
    # Create comparison visualization
    create_multihop_visualization(results, results_dir, timestamp)


def create_multihop_visualization(results, results_dir, timestamp):
    """Create visualization comparing 1-hop vs multi-hop."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Prepare data
    df = pd.DataFrame(results)
    
    # 1. Quality Score by Hop Count
    ax = axes[0, 0]
    hop_groups = df.groupby('max_hops')['quality_score'].mean()
    ax.bar(hop_groups.index, hop_groups.values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_xlabel('Max Hops')
    ax.set_ylabel('Average Quality Score (%)')
    ax.set_title('Quality Score by Hop Count')
    ax.set_xticks([1, 2, 3])
    
    # 2. High-Value Acceptance vs Low-Value Rejection
    ax = axes[0, 1]
    for _, row in df.iterrows():
        color = '#1f77b4' if row['max_hops'] == 1 else ('#ff7f0e' if row['max_hops'] == 2 else '#2ca02c')
        marker = 'o' if not row['multihop'] else 's'
        ax.scatter(row['high_acceptance_rate'], row['low_rejection_rate'], 
                  s=100, color=color, marker=marker, alpha=0.7,
                  label=f"{row['config'][:20]}...")
    ax.set_xlabel('High-Value Acceptance Rate (%)')
    ax.set_ylabel('Low-Value Rejection Rate (%)')
    ax.set_title('Trade-off: Value Acceptance vs Redundancy Rejection')
    ax.grid(True, alpha=0.3)
    
    # 3. Acceptance Rate Comparison
    ax = axes[1, 0]
    configs = ['Conservative', 'Balanced', 'Liberal']
    x = np.arange(len(configs))
    width = 0.25
    
    for i, hops in enumerate([1, 2, 3]):
        hop_data = []
        for config in configs:
            matching = [r['overall_rate'] for r in results 
                       if config in r['config'] and r['max_hops'] == hops]
            hop_data.append(matching[0] if matching else 0)
        
        if any(hop_data):
            ax.bar(x + i * width, hop_data, width, 
                  label=f'{hops}-hop', alpha=0.8)
    
    ax.set_xlabel('Configuration Type')
    ax.set_ylabel('Overall Acceptance Rate (%)')
    ax.set_title('Acceptance Rate by Configuration and Hop Count')
    ax.set_xticks(x + width)
    ax.set_xticklabels(configs)
    ax.legend()
    
    # 4. Quality Score Heatmap
    ax = axes[1, 1]
    pivot_data = df.pivot_table(values='quality_score', 
                                index='k', 
                                columns='max_hops',
                                aggfunc='mean')
    
    if not pivot_data.empty:
        sns.heatmap(pivot_data, annot=True, fmt='.0f', 
                   cmap='YlGn', ax=ax, cbar_kws={'label': 'Quality Score (%)'})
        ax.set_title('Quality Score Heatmap: k vs Hop Count')
        ax.set_xlabel('Max Hops')
        ax.set_ylabel('k value')
    
    plt.suptitle('Multi-Hop geDIG Evaluation Analysis', fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save
    viz_path = results_dir / f"multihop_analysis_{timestamp}.png"
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved visualization to: {viz_path}")


def main():
    """Run the multi-hop quality study."""
    print("\n🚀 Starting Multi-Hop geDIG Quality Study")
    print("=" * 60)
    
    try:
        # Run the study
        results = run_multihop_study()
        
        # Analyze impact
        analyze_multihop_impact(results)
        
        # Save results
        save_results(results)
        
        print("\n" + "=" * 60)
        print("✅ MULTI-HOP STUDY COMPLETE")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Study failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())