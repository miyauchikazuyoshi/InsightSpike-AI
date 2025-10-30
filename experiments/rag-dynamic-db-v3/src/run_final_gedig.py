#!/usr/bin/env python3
"""Final optimized geDIG implementation targeting 30-40% update rate."""

import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd

# Import from improved experiment
from run_improved_gedig import ImprovedGeDIGSystem
from run_experiment_improved import (
    EnhancedRAGSystem,
    create_high_quality_knowledge_base,
    create_meaningful_queries,
    ExperimentConfig
)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)


def run_final_experiment():
    """Run final optimized geDIG experiment."""
    print("🎯 Final Optimized geDIG Experiment")
    print("=" * 60)
    print("Target: 30-40% update rate with meaningful selections")
    print()
    
    # Setup
    config = ExperimentConfig()
    config.cosine_similarity_threshold = 0.6
    
    knowledge_base = create_high_quality_knowledge_base()
    test_queries = create_meaningful_queries()
    
    # Optimized configurations based on learnings
    configurations = [
        {
            'name': 'Target Balanced',
            'params': {
                'k': 0.12,  # Moderate IG penalty
                'node_weight': 0.3,  # Moderate node weight
                'edge_weight': 0.2,  # Moderate edge weight
                'novelty_weight': 0.4,  # Balanced novelty
                'connectivity_weight': 0.1,
                'threshold_base': 0.05,  # Slightly positive threshold
                'threshold_novelty_high': -0.15,  # Accept novel content
                'threshold_novelty_low': 0.15,  # Reject redundant
                'edge_similarity_threshold': 0.35,  # Moderate edge threshold
                'adaptive_k': True,
                'density_k_factor': 0.3
            }
        },
        {
            'name': 'Conservative Smart',
            'params': {
                'k': 0.18,  # Higher IG penalty
                'node_weight': 0.25,
                'edge_weight': 0.15,
                'novelty_weight': 0.5,  # Emphasize novelty
                'connectivity_weight': 0.08,
                'threshold_base': 0.1,  # Positive threshold
                'threshold_novelty_high': -0.1,  # Still accept very novel
                'threshold_novelty_low': 0.2,  # Stricter for redundant
                'edge_similarity_threshold': 0.4,  # Higher edge threshold
                'adaptive_k': True,
                'density_k_factor': 0.4
            }
        },
        {
            'name': 'Selective Growth',
            'params': {
                'k': 0.15,
                'node_weight': 0.28,
                'edge_weight': 0.18,
                'novelty_weight': 0.45,
                'connectivity_weight': 0.09,
                'threshold_base': 0.08,
                'threshold_novelty_high': -0.12,
                'threshold_novelty_low': 0.18,
                'edge_similarity_threshold': 0.38,
                'adaptive_k': True,
                'density_k_factor': 0.35
            }
        },
        {
            'name': 'Quality Focus',
            'params': {
                'k': 0.2,  # High IG penalty for quality
                'node_weight': 0.22,
                'edge_weight': 0.12,
                'novelty_weight': 0.6,  # High novelty requirement
                'connectivity_weight': 0.05,
                'threshold_base': 0.12,
                'threshold_novelty_high': -0.08,
                'threshold_novelty_low': 0.25,
                'edge_similarity_threshold': 0.45,  # High similarity for edges
                'adaptive_k': True,
                'density_k_factor': 0.5
            }
        }
    ]
    
    results = []
    
    for config_data in configurations:
        print(f"\n📊 Testing: {config_data['name']}")
        print("-" * 40)
        
        # Create system
        system = ImprovedGeDIGSystem("gedig", config, config_data['params'])
        
        # Add initial knowledge
        n_added = system.add_initial_knowledge(knowledge_base)
        print(f"  Initial: {n_added} items, density: {system.initial_density:.3f}")
        
        # Process queries
        query_results = []
        update_count = 0
        
        for i, (query, depth) in enumerate(test_queries):
            result = system.process_query(query, depth)
            
            # Get metadata
            if hasattr(system, 'last_gedig_metadata'):
                metadata = system.last_gedig_metadata
            else:
                metadata = {}
            
            updated = result.get('updated', False)
            if updated:
                update_count += 1
            
            query_results.append({
                'query': query,
                'depth': depth,
                'updated': updated,
                'gedig_score': metadata.get('gedig_score', 0),
                'novelty': metadata.get('novelty', 0),
                'threshold': metadata.get('threshold_used', 0)
            })
            
            # Progress indicator
            if (i + 1) % 5 == 0:
                rate = update_count / (i + 1) * 100
                print(f"    After {i+1} queries: {update_count} updates ({rate:.0f}%)")
        
        # Collect final statistics
        stats = system.get_statistics()
        
        # Calculate metrics
        novel_updates = sum(1 for qr in query_results 
                           if qr['updated'] and qr['novelty'] > 0.7)
        moderate_updates = sum(1 for qr in query_results 
                              if qr['updated'] and 0.4 <= qr['novelty'] <= 0.7)
        low_novelty_updates = sum(1 for qr in query_results 
                                 if qr['updated'] and qr['novelty'] < 0.4)
        
        result = {
            'config_name': config_data['name'],
            'params': config_data['params'],
            'update_rate': stats['update_rate'],
            'updates_applied': stats['updates_applied'],
            'novel_updates': novel_updates,
            'moderate_updates': moderate_updates,
            'low_novelty_updates': low_novelty_updates,
            'final_nodes': stats['graph_nodes'],
            'final_edges': stats['graph_edges'],
            'avg_gedig': np.mean(system.gedig_scores) if system.gedig_scores else 0,
            'positive_rate': sum(1 for s in system.gedig_scores if s > 0) / max(1, len(system.gedig_scores)),
            'avg_similarity': stats['avg_similarity'],
            'query_results': query_results
        }
        
        results.append(result)
        
        print(f"\n  Final Results:")
        print(f"    Update rate: {stats['update_rate']:.1%}")
        print(f"    Updates by novelty: High={novel_updates}, Med={moderate_updates}, Low={low_novelty_updates}")
        print(f"    Final graph: {stats['graph_nodes']} nodes, {stats['graph_edges']} edges")
        
        # Check if in target range
        if 0.3 <= stats['update_rate'] <= 0.4:
            print(f"    ✅ IN TARGET RANGE!")
        elif 0.25 <= stats['update_rate'] <= 0.45:
            print(f"    ⚠️ Close to target range")
        else:
            print(f"    ❌ Outside target range")
    
    # Also run baselines for comparison
    print("\n📊 Running Baselines")
    print("-" * 40)
    
    for method in ["static", "frequency", "cosine"]:
        system = EnhancedRAGSystem(method, config)
        system.add_initial_knowledge(knowledge_base)
        
        for query, depth in test_queries:
            system.process_query(query, depth)
        
        stats = system.get_statistics()
        
        result = {
            'config_name': f"{method.upper()} (Baseline)",
            'params': None,
            'update_rate': stats['update_rate'],
            'updates_applied': stats['updates_applied'],
            'final_nodes': stats['graph_nodes'],
            'final_edges': stats['graph_edges'],
            'avg_similarity': stats['avg_similarity']
        }
        
        results.append(result)
        print(f"  {method.upper()}: {stats['update_rate']:.1%} update rate")
    
    return results, test_queries


def visualize_final_results(results: List[Dict], queries: List):
    """Create final visualization focused on achieving target."""
    print("\n📈 Generating Final Visualizations...")
    
    # Create output directory
    output_dir = Path("../results/final_gedig")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data
    df = pd.DataFrame([{k: v for k, v in r.items() if k != 'query_results'} 
                      for r in results])
    
    # Create figure
    fig = plt.figure(figsize=(18, 10))
    
    # 1. Update Rate with Target Zone
    ax1 = plt.subplot(2, 3, 1)
    
    colors = []
    for rate, name in zip(df['update_rate'], df['config_name']):
        if 'Baseline' in name:
            colors.append('#888888')
        elif 0.3 <= rate <= 0.4:
            colors.append('#2E7D32')  # Green for target
        elif 0.25 <= rate <= 0.45:
            colors.append('#FFA726')  # Orange for close
        else:
            colors.append('#D32F2F')  # Red for outside
    
    bars = ax1.barh(df['config_name'], df['update_rate'], color=colors)
    
    # Add target zone
    ax1.axvspan(0.3, 0.4, alpha=0.2, color='green', label='Target Zone (30-40%)')
    ax1.axvspan(0.25, 0.45, alpha=0.1, color='orange', label='Acceptable (25-45%)')
    
    ax1.set_xlabel('Update Rate')
    ax1.set_title('Update Rate vs Target')
    ax1.set_xlim(0, 1.0)
    ax1.legend(loc='upper right')
    
    # Add value labels
    for bar, rate in zip(bars, df['update_rate']):
        width = bar.get_width()
        ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'{rate:.1%}', ha='left', va='center')
    
    # 2. Update Distribution by Novelty
    ax2 = plt.subplot(2, 3, 2)
    
    gedig_results = [r for r in results if 'novel_updates' in r]
    
    if gedig_results:
        categories = ['High Nov\n(>0.7)', 'Med Nov\n(0.4-0.7)', 'Low Nov\n(<0.4)']
        x = np.arange(len(categories))
        width = 0.2
        
        for i, result in enumerate(gedig_results):
            values = [
                result.get('novel_updates', 0),
                result.get('moderate_updates', 0),
                result.get('low_novelty_updates', 0)
            ]
            ax2.bar(x + i * width, values, width, label=result['config_name'])
        
        ax2.set_xlabel('Novelty Category')
        ax2.set_ylabel('Number of Updates')
        ax2.set_title('Update Distribution by Novelty')
        ax2.set_xticks(x + width * 1.5)
        ax2.set_xticklabels(categories)
        ax2.legend(fontsize=8)
    
    # 3. Graph Growth
    ax3 = plt.subplot(2, 3, 3)
    
    initial_nodes = 13
    initial_edges = 29
    
    growth_nodes = df['final_nodes'] - initial_nodes
    growth_edges = df['final_edges'] - initial_edges
    
    x = np.arange(len(df))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, growth_nodes, width, label='Node Growth', color='skyblue')
    bars2 = ax3.bar(x + width/2, growth_edges, width, label='Edge Growth', color='lightcoral')
    
    ax3.set_xlabel('Configuration')
    ax3.set_ylabel('Growth from Initial')
    ax3.set_title('Knowledge Graph Growth')
    ax3.set_xticks(x)
    ax3.set_xticklabels([name.split()[0] for name in df['config_name']], 
                        rotation=45, ha='right')
    ax3.legend()
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # 4. geDIG Score Distribution
    ax4 = plt.subplot(2, 3, 4)
    
    for result in gedig_results:
        if 'query_results' in result:
            scores = [qr['gedig_score'] for qr in result['query_results']]
            if scores:
                ax4.hist(scores, alpha=0.5, label=result['config_name'], bins=10)
    
    ax4.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Decision Boundary')
    ax4.set_xlabel('geDIG Score')
    ax4.set_ylabel('Frequency')
    ax4.set_title('geDIG Score Distribution')
    ax4.legend(fontsize=8)
    
    # 5. Performance Metrics
    ax5 = plt.subplot(2, 3, 5)
    
    # Calculate performance score (targeting 30-40% update rate)
    target_center = 0.35
    df['distance_from_target'] = abs(df['update_rate'] - target_center)
    df['performance_score'] = 1 - df['distance_from_target'] / 0.5  # Normalize
    
    # Scatter plot
    scatter = ax5.scatter(df['update_rate'], df['final_nodes'], 
                         s=100, c=df['performance_score'],
                         cmap='RdYlGn', vmin=0, vmax=1, alpha=0.7)
    
    # Add target zone
    ax5.axvspan(0.3, 0.4, alpha=0.2, color='green')
    
    # Add labels
    for idx, row in df.iterrows():
        name_short = row['config_name'].split()[0]
        ax5.annotate(name_short, 
                    (row['update_rate'], row['final_nodes']),
                    fontsize=8, alpha=0.7)
    
    ax5.set_xlabel('Update Rate')
    ax5.set_ylabel('Final Nodes')
    ax5.set_title('Configuration Performance')
    plt.colorbar(scatter, ax=ax5, label='Performance Score')
    
    # 6. Summary Statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Find best configuration (closest to target)
    gedig_only = df[df['params'].notna()]
    if not gedig_only.empty:
        best_idx = gedig_only['distance_from_target'].idxmin()
        best = gedig_only.loc[best_idx]
        
        # Count configs in ranges
        in_target = sum(1 for _, r in gedig_only.iterrows() if 0.3 <= r['update_rate'] <= 0.4)
        in_acceptable = sum(1 for _, r in gedig_only.iterrows() if 0.25 <= r['update_rate'] <= 0.45)
        
        summary_text = f"""
Final Optimization Results
{'='*40}

Best Configuration: {best['config_name']}
  • Update Rate: {best['update_rate']:.1%}
  • Distance from 35%: {best['distance_from_target']*100:.1f}%
  • Final Size: {int(best['final_nodes'])} nodes, {int(best['final_edges'])} edges
  
Target Achievement:
  • In target (30-40%): {in_target}/{len(gedig_only)} configs
  • In acceptable (25-45%): {in_acceptable}/{len(gedig_only)} configs
  
Key Success Factors:
  • k coefficient: {best['params']['k'] if best['params'] else 'N/A'}
  • Novelty weight: {best['params']['novelty_weight'] if best['params'] else 'N/A'}
  • Base threshold: {best['params']['threshold_base'] if best['params'] else 'N/A'}
  
{'✅ TARGET ACHIEVED!' if 0.3 <= best['update_rate'] <= 0.4 else 
 '⚠️ Close to target' if 0.25 <= best['update_rate'] <= 0.45 else 
 '❌ Target not achieved'}
        """
        
        ax6.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                verticalalignment='center')
    
    ax6.set_title('Optimization Summary')
    
    plt.suptitle('Final geDIG Optimization Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"final_gedig_{timestamp}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved visualization to: {output_path}")
    
    plt.show()
    
    return output_dir, df


def save_final_results(results: List[Dict], df: pd.DataFrame, output_dir: Path):
    """Save final experiment results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save summary
    summary_path = output_dir / f"final_summary_{timestamp}.json"
    
    summary = {
        'timestamp': timestamp,
        'target': '30-40% update rate',
        'configurations': []
    }
    
    for _, row in df.iterrows():
        if row.get('params') is not None:
            config_summary = {
                'name': row['config_name'],
                'update_rate': float(row['update_rate']),
                'in_target': 0.3 <= row['update_rate'] <= 0.4,
                'final_nodes': int(row['final_nodes']),
                'final_edges': int(row['final_edges']),
                'key_params': {
                    'k': row['params']['k'],
                    'threshold_base': row['params']['threshold_base']
                }
            }
            summary['configurations'].append(config_summary)
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save CSV
    csv_path = output_dir / f"final_results_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"  Saved summary to: {summary_path}")
    print(f"  Saved results to: {csv_path}")


def main():
    """Main execution for final geDIG optimization."""
    try:
        # Run final experiment
        results, queries = run_final_experiment()
        
        # Visualize results
        output_dir, df = visualize_final_results(results, queries)
        
        # Save results
        save_final_results(results, df, output_dir)
        
        # Print final summary
        print("\n" + "=" * 60)
        print("🎯 FINAL OPTIMIZATION COMPLETE")
        print("=" * 60)
        
        gedig_only = df[df['params'].notna()]
        if not gedig_only.empty:
            # Find configs in target
            in_target = gedig_only[(gedig_only['update_rate'] >= 0.3) & 
                                   (gedig_only['update_rate'] <= 0.4)]
            
            if not in_target.empty:
                print("\n✅ Configurations achieving target (30-40%):")
                for _, row in in_target.iterrows():
                    print(f"  • {row['config_name']}: {row['update_rate']:.1%}")
            else:
                print("\n⚠️ No configurations achieved exact target.")
                closest = gedig_only.loc[gedig_only['distance_from_target'].idxmin()]
                print(f"  Closest: {closest['config_name']} at {closest['update_rate']:.1%}")
        
        print(f"\n📁 Results saved in: {output_dir}")
        
    except Exception as e:
        print(f"\n❌ Final experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)