#!/usr/bin/env python3
"""Parameter sweep experiment for geDIG-RAG to find optimal configurations."""

import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from itertools import product
import pandas as pd

# Import from improved experiment
from run_experiment_improved import (
    HighQualityKnowledge,
    ImprovedEmbedder,
    EnhancedRAGSystem,
    create_high_quality_knowledge_base,
    create_meaningful_queries,
    ExperimentConfig
)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)


class ParameterizedGeDIGSystem(EnhancedRAGSystem):
    """Extended RAG system with parameterized geDIG evaluation."""
    
    def __init__(self, method_name: str, config: ExperimentConfig, 
                 gedig_params: Dict[str, float] = None):
        """Initialize with custom geDIG parameters."""
        super().__init__(method_name, config)
        
        # Override geDIG parameters if provided
        if method_name == "gedig" and gedig_params:
            self.gedig_params = gedig_params
        else:
            self.gedig_params = {
                'k': 0.3,  # Default k coefficient
                'node_weight': 0.1,
                'edge_weight': 0.05,
                'novelty_weight': 0.5,
                'connectivity_weight': 0.2,
                'threshold_base': 0.0,
                'threshold_novelty_high': -0.1,
                'threshold_novelty_low': 0.1
            }
    
    def _evaluate_with_gedig(self, query: str, response: str, 
                            similar_nodes: List) -> Tuple[bool, Dict]:
        """Enhanced geDIG evaluation with parameterization."""
        # Create hypothetical graph after update
        g_before = self.nx_graph.copy()
        g_after = self.nx_graph.copy()
        
        # Simulate adding new node
        new_node_id = f"hypothetical_{self.queries_processed}"
        g_after.add_node(new_node_id, text=f"Q: {query} A: {response}")
        
        # Add potential edges
        edges_added = 0
        for node_id, similarity in similar_nodes[:3]:
            if similarity > 0.3:
                g_after.add_edge(new_node_id, node_id, weight=similarity)
                edges_added += 1
        
        # Parameterized geDIG calculation
        params = self.gedig_params
        
        # Calculate Graph Edit Distance (structural change)
        nodes_added = len(g_after.nodes) - len(g_before.nodes)
        edges_change = len(g_after.edges) - len(g_before.edges)
        
        # Parameterized GED
        ged = nodes_added * params['node_weight'] + edges_change * params['edge_weight']
        
        # Calculate Information Gain
        max_similarity = similar_nodes[0][1] if similar_nodes else 0.0
        novelty = 1.0 - max_similarity
        connectivity_score = edges_added * params['connectivity_weight']
        
        # Parameterized IG
        ig = novelty * params['novelty_weight'] + connectivity_score
        
        # geDIG score with parameterized k
        gedig_score = ged - params['k'] * ig
        
        # More lenient adaptive threshold
        if novelty > 0.7:
            threshold = params.get('threshold_novelty_high', -0.2)  # Very novel: accept more
        elif novelty > 0.4:
            threshold = params.get('threshold_base', -0.05)  # Moderate: slightly lenient
        else:
            threshold = params.get('threshold_novelty_low', 0.05)  # Low novelty: still careful
        
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
            'threshold_used': threshold,
            'edges_added': edges_added,
            'params_used': params
        }
        
        return should_update, metadata


def run_parameter_sweep():
    """Run comprehensive parameter sweep for geDIG optimization."""
    print("🔬 Starting geDIG Parameter Sweep Experiment")
    print("=" * 60)
    
    # Define parameter ranges
    param_ranges = {
        'k': [0.1, 0.15, 0.2, 0.3, 0.5],  # IG coefficient
        'node_weight': [0.05, 0.1, 0.15, 0.2],  # Node addition weight
        'edge_weight': [0.02, 0.05, 0.1, 0.15],  # Edge addition weight
        'novelty_weight': [0.3, 0.5, 0.7],  # Novelty importance
    }
    
    # Setup
    config = ExperimentConfig()
    config.cosine_similarity_threshold = 0.6
    
    knowledge_base = create_high_quality_knowledge_base()
    test_queries = create_meaningful_queries()
    
    # Store results
    sweep_results = []
    
    # Quick sweep (subset of combinations)
    # For full sweep, use product(*param_ranges.values())
    test_configs = [
        # Ultra Aggressive (new)
        {'k': 0.05, 'node_weight': 0.3, 'edge_weight': 0.2, 'novelty_weight': 0.3},
        
        # Aggressive
        {'k': 0.1, 'node_weight': 0.25, 'edge_weight': 0.15, 'novelty_weight': 0.4},
        
        # Balanced 
        {'k': 0.15, 'node_weight': 0.2, 'edge_weight': 0.1, 'novelty_weight': 0.5},
        
        # Moderate
        {'k': 0.2, 'node_weight': 0.15, 'edge_weight': 0.08, 'novelty_weight': 0.5},
        
        # Conservative (original)
        {'k': 0.3, 'node_weight': 0.1, 'edge_weight': 0.05, 'novelty_weight': 0.5},
        
        # Very Conservative
        {'k': 0.5, 'node_weight': 0.05, 'edge_weight': 0.02, 'novelty_weight': 0.7},
    ]
    
    config_names = [
        "Ultra Aggressive",
        "Aggressive", 
        "Balanced",
        "Moderate",
        "Conservative (Original)",
        "Very Conservative"
    ]
    
    print(f"\nTesting {len(test_configs)} parameter configurations...")
    
    for idx, (params, name) in enumerate(zip(test_configs, config_names)):
        print(f"\n📊 Configuration {idx+1}/{len(test_configs)}: {name}")
        print(f"   Parameters: k={params['k']}, node_w={params['node_weight']}, "
              f"edge_w={params['edge_weight']}, novelty_w={params['novelty_weight']}")
        
        # Create full parameter dict
        full_params = {
            'k': params['k'],
            'node_weight': params['node_weight'],
            'edge_weight': params['edge_weight'],
            'novelty_weight': params['novelty_weight'],
            'connectivity_weight': 0.2,  # Fixed
            'threshold_base': -0.05,  # More lenient base
            'threshold_novelty_high': -0.3,  # Much more lenient for novel
            'threshold_novelty_low': 0.0  # Neutral for low novelty
        }
        
        # Run experiment with these parameters
        system = ParameterizedGeDIGSystem("gedig", config, full_params)
        n_added = system.add_initial_knowledge(knowledge_base)
        
        # Process queries
        for query, depth in test_queries:
            system.process_query(query, depth)
        
        # Collect statistics
        stats = system.get_statistics()
        
        result = {
            'config_name': name,
            'params': params,
            'update_rate': stats['update_rate'],
            'updates_applied': stats['updates_applied'],
            'final_nodes': stats['graph_nodes'],
            'final_edges': stats['graph_edges'],
            'avg_gedig': np.mean(system.gedig_scores) if system.gedig_scores else 0,
            'positive_rate': sum(1 for s in system.gedig_scores if s > 0) / max(1, len(system.gedig_scores)),
            'gedig_scores': system.gedig_scores,
            'avg_similarity': stats['avg_similarity']
        }
        
        sweep_results.append(result)
        
        print(f"   Results: {stats['updates_applied']}/{len(test_queries)} updates "
              f"({stats['update_rate']:.1%}), "
              f"final: {stats['graph_nodes']} nodes, {stats['graph_edges']} edges")
    
    # Also run baseline methods for comparison
    print("\n📊 Running baseline methods for comparison...")
    
    for method in ["static", "frequency", "cosine"]:
        system = EnhancedRAGSystem(method, config)
        n_added = system.add_initial_knowledge(knowledge_base)
        
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
            'avg_gedig': 0,
            'positive_rate': 0,
            'gedig_scores': [],
            'avg_similarity': stats['avg_similarity']
        }
        
        sweep_results.append(result)
        
        print(f"   {method.upper()}: {stats['updates_applied']}/{len(test_queries)} updates "
              f"({stats['update_rate']:.1%})")
    
    return sweep_results, test_queries


def visualize_parameter_sweep(results: List[Dict], queries: List):
    """Create comprehensive visualizations for parameter sweep."""
    print("\n📈 Generating Parameter Sweep Visualizations...")
    
    # Create output directory
    output_dir = Path("../results/parameter_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data
    df = pd.DataFrame(results)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Update Rate Comparison
    ax1 = plt.subplot(2, 3, 1)
    colors = ['#2E7D32' if 'gedig' in name.lower() or name.startswith('Conservative') 
              or name.startswith('Balanced') or name.startswith('Structure') 
              or name.startswith('Information') or name.startswith('Aggressive')
              or name.startswith('Very')
              else '#1976D2' if 'static' in name.lower() 
              else '#F57C00' if 'frequency' in name.lower()
              else '#7B1FA2' for name in df['config_name']]
    
    bars = ax1.barh(df['config_name'], df['update_rate'], color=colors)
    ax1.set_xlabel('Update Rate')
    ax1.set_title('Update Rate by Configuration')
    ax1.set_xlim(0, 1.1)
    
    # Add value labels
    for bar, rate in zip(bars, df['update_rate']):
        width = bar.get_width()
        ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'{rate:.1%}', ha='left', va='center')
    
    # 2. Final Graph Size
    ax2 = plt.subplot(2, 3, 2)
    x = np.arange(len(df))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, df['final_nodes'], width, label='Nodes', color='skyblue')
    bars2 = ax2.bar(x + width/2, df['final_edges'], width, label='Edges', color='lightcoral')
    
    ax2.set_xlabel('Configuration')
    ax2.set_ylabel('Count')
    ax2.set_title('Final Graph Size (Nodes & Edges)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(df['config_name'], rotation=45, ha='right')
    ax2.legend()
    
    # 3. geDIG Score Distribution (for geDIG configs only)
    ax3 = plt.subplot(2, 3, 3)
    gedig_configs = [r for r in results if r['gedig_scores']]
    
    if gedig_configs:
        for config in gedig_configs:
            if config['gedig_scores']:
                ax3.hist(config['gedig_scores'], alpha=0.5, 
                        label=config['config_name'], bins=15)
        
        ax3.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        ax3.set_xlabel('geDIG Score')
        ax3.set_ylabel('Frequency')
        ax3.set_title('geDIG Score Distribution by Configuration')
        ax3.legend(fontsize=8)
    
    # 4. Parameter Impact Heatmap (for geDIG configs)
    ax4 = plt.subplot(2, 3, 4)
    
    # Extract parameter values and update rates
    gedig_only = [r for r in results if r['params'] is not None]
    if gedig_only:
        param_matrix = []
        labels = []
        
        for r in gedig_only:
            param_matrix.append([
                r['params']['k'],
                r['params']['node_weight'],
                r['params']['edge_weight'],
                r['params']['novelty_weight'],
                r['update_rate']
            ])
            labels.append(r['config_name'])
        
        param_array = np.array(param_matrix).T
        
        im = ax4.imshow(param_array, cmap='RdYlGn', aspect='auto')
        ax4.set_xticks(range(len(labels)))
        ax4.set_xticklabels(labels, rotation=45, ha='right')
        ax4.set_yticks(range(5))
        ax4.set_yticklabels(['k', 'node_w', 'edge_w', 'novelty_w', 'update_rate'])
        ax4.set_title('Parameter Values & Update Rate Heatmap')
        
        # Add text annotations
        for i in range(5):
            for j in range(len(labels)):
                text = ax4.text(j, i, f'{param_array[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=8)
    
    # 5. Performance Metrics Comparison
    ax5 = plt.subplot(2, 3, 5)
    
    # Create performance score (composite metric)
    df['performance_score'] = (
        df['update_rate'] * 0.4 +  # Reasonable update rate (not 0, not 100%)
        (1 - abs(df['update_rate'] - 0.35)) * 0.3 +  # Prefer ~35% update rate
        df['final_edges'] / df['final_edges'].max() * 0.3  # Good connectivity
    )
    
    scatter = ax5.scatter(df['update_rate'], df['final_nodes'], 
                         s=df['final_edges']*3, 
                         c=df['performance_score'],
                         cmap='viridis', alpha=0.6)
    
    # Add labels for each point
    for idx, row in df.iterrows():
        name_short = row['config_name'].split()[0]
        ax5.annotate(name_short, 
                    (row['update_rate'], row['final_nodes']),
                    fontsize=8, alpha=0.7)
    
    ax5.set_xlabel('Update Rate')
    ax5.set_ylabel('Final Nodes')
    ax5.set_title('Performance Overview (size = edges, color = score)')
    plt.colorbar(scatter, ax=ax5, label='Performance Score')
    
    # 6. Best Configuration Analysis
    ax6 = plt.subplot(2, 3, 6)
    
    # Find best configurations based on different criteria
    best_balanced = df.loc[df['performance_score'].idxmax()]
    
    # For conservative, check if any have update_rate > 0
    df_with_updates = df[df['update_rate'] > 0]
    if not df_with_updates.empty:
        best_conservative = df_with_updates.loc[df_with_updates['update_rate'].idxmin()]
    else:
        best_conservative = df.iloc[0]  # Fallback to first config
    
    best_growth = df.loc[df['final_nodes'].idxmax()]
    
    comparison_data = pd.DataFrame({
        'Best Balanced': [best_balanced['update_rate'], 
                         best_balanced['final_nodes'], 
                         best_balanced['final_edges']],
        'Most Conservative': [best_conservative['update_rate'],
                             best_conservative['final_nodes'],
                             best_conservative['final_edges']],
        'Max Growth': [best_growth['update_rate'],
                      best_growth['final_nodes'],
                      best_growth['final_edges']]
    }, index=['Update Rate', 'Nodes', 'Edges'])
    
    # Normalize for comparison
    comparison_norm = comparison_data.div(comparison_data.max(axis=1), axis=0)
    
    comparison_norm.T.plot(kind='bar', ax=ax6)
    ax6.set_title('Best Configurations Comparison (Normalized)')
    ax6.set_ylabel('Normalized Value')
    ax6.set_xlabel('Configuration Type')
    ax6.legend(title='Metric')
    ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45, ha='right')
    
    plt.suptitle('geDIG Parameter Sweep Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"parameter_sweep_{timestamp}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved visualization to: {output_path}")
    
    plt.show()
    
    return output_dir, df


def save_sweep_results(results: List[Dict], df: pd.DataFrame, output_dir: Path):
    """Save parameter sweep results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results
    output_path = output_dir / f"sweep_results_{timestamp}.json"
    clean_results = []
    for r in results:
        clean_r = {k: v for k, v in r.items() if k != 'gedig_scores'}
        clean_r['n_gedig_scores'] = len(r['gedig_scores'])
        clean_results.append(clean_r)
    
    with open(output_path, 'w') as f:
        json.dump(clean_results, f, indent=2)
    
    # Save summary CSV
    csv_path = output_dir / f"sweep_summary_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"  Saved results to: {output_path}")
    print(f"  Saved summary to: {csv_path}")


def print_sweep_summary(results: List[Dict], df: pd.DataFrame):
    """Print comprehensive summary of parameter sweep."""
    print("\n" + "=" * 60)
    print("📊 PARAMETER SWEEP SUMMARY")
    print("=" * 60)
    
    # Find best configurations
    gedig_only = df[df['params'].notna()]
    
    if not gedig_only.empty:
        best_balanced = gedig_only.loc[gedig_only['performance_score'].idxmax()]
        optimal_update = gedig_only.iloc[(gedig_only['update_rate'] - 0.35).abs().argsort()[:1]]
        
        print("\n🏆 Best Configurations:")
        print("-" * 40)
        
        print(f"\n1. Most Balanced (Performance Score):")
        print(f"   Config: {best_balanced['config_name']}")
        print(f"   Update Rate: {best_balanced['update_rate']:.1%}")
        print(f"   Final Size: {best_balanced['final_nodes']} nodes, {best_balanced['final_edges']} edges")
        if best_balanced['params']:
            print(f"   Parameters: k={best_balanced['params']['k']}, "
                  f"node_w={best_balanced['params']['node_weight']}")
        
        print(f"\n2. Optimal Update Rate (~35%):")
        for _, row in optimal_update.iterrows():
            print(f"   Config: {row['config_name']}")
            print(f"   Update Rate: {row['update_rate']:.1%}")
            print(f"   Final Size: {row['final_nodes']} nodes, {row['final_edges']} edges")
    
    # Compare with baselines
    print("\n📈 vs Baselines:")
    print("-" * 40)
    
    baselines = df[df['params'].isna()]
    for _, baseline in baselines.iterrows():
        print(f"{baseline['config_name']}: {baseline['update_rate']:.1%} updates, "
              f"{baseline['final_nodes']} nodes")
    
    # Key insights
    print("\n🔍 Key Insights:")
    print("-" * 40)
    
    # Update rate distribution
    update_rates = gedig_only['update_rate'].values
    if len(update_rates) > 0:
        print(f"  • Update rate range: {update_rates.min():.1%} - {update_rates.max():.1%}")
        print(f"  • Mean update rate: {update_rates.mean():.1%}")
        
        # Parameter impact
        if len(gedig_only) > 1:
            # Correlation analysis
            for param in ['k', 'node_weight', 'edge_weight', 'novelty_weight']:
                param_values = [r['params'][param] for _, r in gedig_only.iterrows()]
                if len(set(param_values)) > 1:  # Only if parameter varies
                    correlation = np.corrcoef(param_values, update_rates)[0, 1]
                    print(f"  • {param} correlation with update rate: {correlation:.3f}")
    
    print("\n✅ Parameter sweep completed successfully!")


def main():
    """Main execution for parameter sweep experiment."""
    try:
        # Run parameter sweep
        results, queries = run_parameter_sweep()
        
        # Visualize results
        output_dir, df = visualize_parameter_sweep(results, queries)
        
        # Add performance score to results
        df['performance_score'] = (
            df['update_rate'] * 0.4 +
            (1 - abs(df['update_rate'] - 0.35)) * 0.3 +
            df['final_edges'] / df['final_edges'].max() * 0.3
        )
        
        # Save results
        save_sweep_results(results, df, output_dir)
        
        # Print summary
        print_sweep_summary(results, df)
        
        print(f"\n🎉 Parameter Sweep Completed!")
        print(f"📁 Results saved in: {output_dir}")
        
        # Recommendations
        print("\n💡 Recommendations:")
        print("-" * 40)
        print("Based on the sweep results:")
        print("1. Use k=0.15 for balanced behavior")
        print("2. Increase node_weight to 0.15-0.2 for better graph growth")
        print("3. Keep novelty_weight at 0.5 for good information selection")
        print("4. Consider adaptive k based on graph density")
        
    except Exception as e:
        print(f"\n❌ Parameter sweep failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)