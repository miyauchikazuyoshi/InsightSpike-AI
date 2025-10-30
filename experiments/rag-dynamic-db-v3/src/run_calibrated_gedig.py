#!/usr/bin/env python3
"""Calibrated geDIG implementation accounting for low baseline similarity."""

import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd

from run_improved_gedig import ImprovedGeDIGSystem
from run_experiment_improved import (
    EnhancedRAGSystem,
    create_high_quality_knowledge_base,
    create_meaningful_queries,
    ExperimentConfig
)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)


class CalibratedGeDIGSystem(ImprovedGeDIGSystem):
    """Calibrated geDIG system for low-similarity environments."""
    
    def _evaluate_with_gedig(self, query: str, response: str, 
                            similar_nodes: List) -> Tuple[bool, Dict]:
        """Calibrated geDIG evaluation for low baseline similarity."""
        # Get base calculation from parent
        g_before = self.nx_graph.copy()
        g_after = self.nx_graph.copy()
        
        new_node_id = f"hypothetical_{self.queries_processed}"
        g_after.add_node(new_node_id, text=f"Q: {query} A: {response}")
        
        # Calibrated edge formation (lower threshold for low-similarity environment)
        edges_added = 0
        edge_threshold = 0.08  # Much lower threshold since similarities are all <0.2
        
        for node_id, similarity in similar_nodes[:5]:
            if similarity > edge_threshold:
                g_after.add_edge(new_node_id, node_id, weight=similarity)
                edges_added += 1
        
        # Get adaptive k
        adaptive_k = self._calculate_adaptive_k()
        
        # Calculate GED
        nodes_added = 1  # Always adding one node
        edges_change = edges_added
        
        params = self.gedig_params
        ged = nodes_added * params['node_weight'] + edges_change * params['edge_weight']
        
        # Calibrated novelty calculation for low-similarity environment
        max_similarity = similar_nodes[0][1] if similar_nodes else 0.0
        
        # Since all similarities are <0.2, we need to recalibrate novelty
        # Map [0, 0.2] similarity to [0.5, 1.0] novelty
        if max_similarity < 0.05:
            novelty = 1.0  # Extremely novel
        elif max_similarity < 0.1:
            novelty = 0.9  # Very novel
        elif max_similarity < 0.15:
            novelty = 0.75  # Novel
        elif max_similarity < 0.2:
            novelty = 0.6  # Moderately novel
        else:
            novelty = 0.5  # Still somewhat novel
        
        connectivity_score = edges_added * params['connectivity_weight']
        ig = novelty * params['novelty_weight'] + connectivity_score
        
        # geDIG score
        gedig_score = ged - adaptive_k * ig
        
        # Calibrated thresholds for high-novelty environment
        # Since everything is novel, we need MUCH higher thresholds to be selective
        # geDIG scores are typically 0.3-0.6, so thresholds need to be in that range
        if self.queries_processed <= 3:
            # Accept first 3 queries to build initial knowledge
            threshold = 0.2
        elif novelty >= 1.0:
            # Extremely novel - use configured threshold
            threshold = params.get('threshold_extreme_novelty', 0.45)
        elif novelty >= 0.9:
            # Very novel - slightly higher
            threshold = params.get('threshold_very_high_novelty', 0.48)
        elif novelty >= 0.75:
            # Novel - even higher
            threshold = params.get('threshold_high_novelty', 0.5)
        else:
            # Moderately novel - highest threshold
            threshold = params.get('threshold_moderate_novelty', 0.52)
        
        should_update = gedig_score > threshold
        
        # Store metrics
        self.gedig_scores.append(gedig_score)
        self.ig_values.append(ig)
        self.ged_values.append(ged)
        
        metadata = {
            'gedig_score': gedig_score,
            'ged': ged,
            'ig': ig,
            'novelty': novelty,
            'max_similarity': max_similarity,
            'threshold_used': threshold,
            'edges_added': edges_added,
            'adaptive_k': adaptive_k,
            'query_number': self.queries_processed
        }
        
        self.last_gedig_metadata = metadata
        
        return should_update, metadata


def run_calibrated_experiment():
    """Run calibrated geDIG experiment."""
    print("🎯 Calibrated geDIG Experiment")
    print("=" * 60)
    print("Calibrated for low-similarity environment (all queries <0.2 similarity)")
    print()
    
    config = ExperimentConfig()
    knowledge_base = create_high_quality_knowledge_base()
    test_queries = create_meaningful_queries()
    
    # Calibrated configurations with realistic thresholds
    configurations = [
        {
            'name': 'Target 35%',
            'params': {
                'k': 0.15,
                'node_weight': 0.5,
                'edge_weight': 0.2,
                'novelty_weight': 0.4,
                'connectivity_weight': 0.1,
                'threshold_extreme_novelty': 0.42,  # Most queries get geDIG ~0.4-0.5
                'threshold_very_high_novelty': 0.44,
                'threshold_high_novelty': 0.46,
                'threshold_moderate_novelty': 0.48,
                'adaptive_k': True,
                'density_k_factor': 0.3
            }
        },
        {
            'name': 'Target 40%',
            'params': {
                'k': 0.12,
                'node_weight': 0.45,
                'edge_weight': 0.18,
                'novelty_weight': 0.35,
                'connectivity_weight': 0.08,
                'threshold_extreme_novelty': 0.40,  # Slightly lower thresholds
                'threshold_very_high_novelty': 0.42,
                'threshold_high_novelty': 0.44,
                'threshold_moderate_novelty': 0.46,
                'adaptive_k': True,
                'density_k_factor': 0.25
            }
        },
        {
            'name': 'Target 30%',
            'params': {
                'k': 0.18,
                'node_weight': 0.55,
                'edge_weight': 0.22,
                'novelty_weight': 0.45,
                'connectivity_weight': 0.12,
                'threshold_extreme_novelty': 0.44,  # Higher thresholds
                'threshold_very_high_novelty': 0.46,
                'threshold_high_novelty': 0.48,
                'threshold_moderate_novelty': 0.50,
                'adaptive_k': True,
                'density_k_factor': 0.35
            }
        },
        {
            'name': 'Target 25%',
            'params': {
                'k': 0.2,
                'node_weight': 0.6,
                'edge_weight': 0.25,
                'novelty_weight': 0.5,
                'connectivity_weight': 0.15,
                'threshold_extreme_novelty': 0.46,  # Even higher
                'threshold_very_high_novelty': 0.48,
                'threshold_high_novelty': 0.50,
                'threshold_moderate_novelty': 0.52,
                'adaptive_k': True,
                'density_k_factor': 0.4
            }
        }
    ]
    
    results = []
    
    for config_data in configurations:
        print(f"\n📊 Testing: {config_data['name']}")
        print("-" * 40)
        
        system = CalibratedGeDIGSystem("gedig", config, config_data['params'])
        n_added = system.add_initial_knowledge(knowledge_base)
        print(f"  Initial: {n_added} items")
        
        query_results = []
        decisions = []
        
        for i, (query, depth) in enumerate(test_queries):
            result = system.process_query(query, depth)
            
            metadata = system.last_gedig_metadata
            updated = result.get('updated', False)
            decisions.append(updated)
            
            query_results.append({
                'query': query[:50],
                'updated': updated,
                'gedig_score': metadata.get('gedig_score', 0),
                'threshold': metadata.get('threshold_used', 0),
                'similarity': metadata.get('max_similarity', 0),
                'novelty': metadata.get('novelty', 0)
            })
            
            if (i + 1) % 5 == 0:
                rate = sum(decisions) / len(decisions) * 100
                print(f"    After {i+1} queries: {sum(decisions)} updates ({rate:.0f}%)")
        
        stats = system.get_statistics()
        
        # Categorize updates
        early_updates = sum(decisions[:5])
        mid_updates = sum(decisions[5:12])
        late_updates = sum(decisions[12:])
        
        result = {
            'config_name': config_data['name'],
            'params': config_data['params'],
            'update_rate': stats['update_rate'],
            'updates_applied': stats['updates_applied'],
            'early_updates': early_updates,
            'mid_updates': mid_updates,
            'late_updates': late_updates,
            'final_nodes': stats['graph_nodes'],
            'final_edges': stats['graph_edges'],
            'avg_gedig': np.mean(system.gedig_scores) if system.gedig_scores else 0,
            'query_results': query_results
        }
        
        results.append(result)
        
        print(f"\n  Final Results:")
        print(f"    Update rate: {stats['update_rate']:.1%}")
        print(f"    Distribution: Early={early_updates}/5, Mid={mid_updates}/7, Late={late_updates}/7")
        print(f"    Final graph: {stats['graph_nodes']} nodes, {stats['graph_edges']} edges")
        
        if 0.3 <= stats['update_rate'] <= 0.4:
            print(f"    ✅ IN TARGET RANGE (30-40%)!")
        elif 0.25 <= stats['update_rate'] <= 0.45:
            print(f"    ⚠️ Close to target")
    
    # Baselines
    print("\n📊 Running Baselines")
    print("-" * 40)
    
    for method in ["static", "frequency", "cosine"]:
        system = EnhancedRAGSystem(method, config)
        system.add_initial_knowledge(knowledge_base)
        
        for query, depth in test_queries:
            system.process_query(query, depth)
        
        stats = system.get_statistics()
        results.append({
            'config_name': f"{method.upper()}",
            'update_rate': stats['update_rate'],
            'updates_applied': stats['updates_applied'],
            'final_nodes': stats['graph_nodes'],
            'final_edges': stats['graph_edges']
        })
        print(f"  {method.upper()}: {stats['update_rate']:.1%}")
    
    return results


def visualize_calibrated_results(results: List[Dict]):
    """Visualize calibrated results."""
    print("\n📈 Generating Calibrated Visualizations...")
    
    output_dir = Path("../results/calibrated_gedig")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame([{k: v for k, v in r.items() if k != 'query_results'} 
                      for r in results])
    
    fig = plt.figure(figsize=(16, 8))
    
    # 1. Update Rate with Target
    ax1 = plt.subplot(2, 3, 1)
    
    colors = []
    for rate, name in zip(df['update_rate'], df['config_name']):
        if any(x in name for x in ['STATIC', 'FREQUENCY', 'COSINE']):
            colors.append('#888888')
        elif 0.3 <= rate <= 0.4:
            colors.append('#2E7D32')
        elif 0.25 <= rate <= 0.45:
            colors.append('#FFA726')
        else:
            colors.append('#D32F2F')
    
    bars = ax1.barh(df['config_name'], df['update_rate'], color=colors)
    ax1.axvspan(0.3, 0.4, alpha=0.2, color='green', label='Target')
    ax1.set_xlabel('Update Rate')
    ax1.set_title('Update Rate Achievement')
    ax1.legend()
    
    for bar, rate in zip(bars, df['update_rate']):
        ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{rate:.1%}', ha='left', va='center')
    
    # 2. Update Distribution
    ax2 = plt.subplot(2, 3, 2)
    
    gedig_results = [r for r in results if 'early_updates' in r]
    if gedig_results:
        categories = ['Early\n(1-5)', 'Mid\n(6-12)', 'Late\n(13-19)']
        x = np.arange(len(categories))
        width = 0.15
        
        for i, result in enumerate(gedig_results):
            values = [
                result.get('early_updates', 0),
                result.get('mid_updates', 0),
                result.get('late_updates', 0)
            ]
            ax2.bar(x + i * width, values, width, label=result['config_name'])
        
        ax2.set_xlabel('Query Phase')
        ax2.set_ylabel('Updates')
        ax2.set_title('Update Distribution by Phase')
        ax2.set_xticks(x + width * 1.5)
        ax2.set_xticklabels(categories)
        ax2.legend(fontsize=8)
    
    # 3. Graph Growth
    ax3 = plt.subplot(2, 3, 3)
    
    growth = df['final_nodes'] - 13
    x = np.arange(len(df))
    
    bars = ax3.bar(x, growth, color=['green' if g > 5 else 'orange' for g in growth])
    ax3.set_xlabel('Configuration')
    ax3.set_ylabel('Nodes Added')
    ax3.set_title('Knowledge Growth')
    ax3.set_xticks(x)
    ax3.set_xticklabels([n.split()[0] for n in df['config_name']], rotation=45, ha='right')
    ax3.axhline(y=7, color='red', linestyle='--', alpha=0.5, label='Target (7 nodes)')
    ax3.legend()
    
    # 4. Decision Pattern (first config with query results)
    ax4 = plt.subplot(2, 3, 4)
    
    for result in results[:1]:  # First config only
        if 'query_results' in result:
            decisions = [1 if qr['updated'] else 0 for qr in result['query_results']]
            ax4.plot(decisions, 'o-', label=result['config_name'])
            ax4.set_xlabel('Query Number')
            ax4.set_ylabel('Update Decision')
            ax4.set_title('Update Decision Pattern')
            ax4.set_ylim(-0.1, 1.1)
            ax4.legend()
    
    # 5. geDIG Score vs Threshold
    ax5 = plt.subplot(2, 3, 5)
    
    for result in results[:1]:  # First config
        if 'query_results' in result:
            scores = [qr['gedig_score'] for qr in result['query_results']]
            thresholds = [qr['threshold'] for qr in result['query_results']]
            ax5.plot(scores, label='geDIG Score', marker='o')
            ax5.plot(thresholds, label='Threshold', marker='s', linestyle='--')
            ax5.set_xlabel('Query Number')
            ax5.set_ylabel('Value')
            ax5.set_title('geDIG Score vs Threshold')
            ax5.legend()
            ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # 6. Summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    gedig_only = df[df['config_name'].str.contains('%')]
    if not gedig_only.empty:
        in_target = sum(1 for _, r in gedig_only.iterrows() if 0.3 <= r['update_rate'] <= 0.4)
        
        best_distance = float('inf')
        best_config = None
        for _, r in gedig_only.iterrows():
            distance = abs(r['update_rate'] - 0.35)
            if distance < best_distance:
                best_distance = distance
                best_config = r
        
        if best_config is not None:
            config_name = best_config['config_name']
            update_rate = best_config['update_rate']
        else:
            config_name = 'N/A'
            update_rate = 0
            
        summary = f"""
Calibrated Results Summary
{'='*35}

Best Config: {config_name}
Update Rate: {update_rate:.1%}

Target Achievement:
• In target (30-40%): {in_target}/{len(gedig_only)}
• Best distance from 35%: {best_distance*100:.1f}%

Calibration Success:
{'✅ TARGET ACHIEVED' if in_target > 0 else '⚠️ Close to target'}
        """
        
        ax6.text(0.1, 0.5, summary, fontsize=11, family='monospace',
                verticalalignment='center')
    
    plt.suptitle('Calibrated geDIG Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"calibrated_{timestamp}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved to: {output_path}")
    
    plt.show()
    
    return output_dir


def main():
    """Main execution."""
    try:
        results = run_calibrated_experiment()
        output_dir = visualize_calibrated_results(results)
        
        print("\n" + "=" * 60)
        print("🎯 CALIBRATED EXPERIMENT COMPLETE")
        print("=" * 60)
        
        # Find successes
        successes = [r for r in results if 'params' in r and 0.3 <= r['update_rate'] <= 0.4]
        
        if successes:
            print("\n✅ Configurations achieving target (30-40%):")
            for r in successes:
                print(f"  • {r['config_name']}: {r['update_rate']:.1%}")
        
        print(f"\n📁 Results saved in: {output_dir}")
        
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)