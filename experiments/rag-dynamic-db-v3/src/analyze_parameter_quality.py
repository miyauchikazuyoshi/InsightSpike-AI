#!/usr/bin/env python3
"""Analyze qualitative trends for different fixed parameter settings."""

import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from run_experiment_improved import (
    create_high_quality_knowledge_base,
    create_meaningful_queries,
    ExperimentConfig,
    EnhancedRAGSystem
)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


class FixedParameterGeDIGSystem(EnhancedRAGSystem):
    """geDIG system with fixed parameters for analysis."""
    
    def __init__(self, config: ExperimentConfig, params: dict):
        super().__init__("gedig", config)
        self.params = params
        self.decision_log = []
        # Force gedig_core to be truthy so parent calls our _evaluate_with_gedig
        self.gedig_core = True  # Dummy value to trigger our override
        
    def _evaluate_with_gedig(self, query: str, response: str, 
                            similar_nodes: list) -> tuple:
        """Simple fixed-parameter geDIG evaluation."""
        # Simulate adding new node
        new_node_id = f"hypothetical_{self.queries_processed}"
        
        # Count potential edges
        edges_added = 0
        for node_id, similarity in similar_nodes[:5]:
            if similarity > 0.15:  # Fixed edge threshold
                edges_added += 1
        
        # Calculate GED (structural change)
        nodes_added = 1
        ged = nodes_added * self.params['node_weight'] + edges_added * self.params['edge_weight']
        
        # Calculate IG (information value)
        max_similarity = similar_nodes[0][1] if similar_nodes else 0.0
        novelty = 1.0 - max_similarity
        connectivity_score = edges_added * 0.1
        ig = novelty * self.params['novelty_weight'] + connectivity_score
        
        # geDIG score
        gedig_score = ged - self.params['k'] * ig
        
        # Fixed threshold
        threshold = self.params['threshold']
        should_update = gedig_score > threshold
        
        # Log decision details
        self.decision_log.append({
            'query': query[:80],
            'novelty': novelty,
            'similarity': max_similarity,
            'edges': edges_added,
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
            'edges_added': edges_added
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
            'query': query[:80],
            'value': value,
            'reason': reason
        })
    
    return query_categories


def run_parameter_study():
    """Run study with different fixed parameter settings."""
    print("🔬 Fixed Parameter Quality Study")
    print("=" * 60)
    
    # Different parameter configurations to test (refined for better discrimination)
    configurations = [
        {
            'name': 'Very Conservative (k=0.8, thresh=0.45)',
            'params': {
                'k': 0.8,  # Very high IG penalty
                'node_weight': 0.3,
                'edge_weight': 0.1,
                'novelty_weight': 0.6,
                'threshold': 0.45  # High threshold - accepts very few
            }
        },
        {
            'name': 'Conservative (k=0.5, thresh=0.38)',
            'params': {
                'k': 0.5,  # High IG penalty
                'node_weight': 0.35,
                'edge_weight': 0.15,
                'novelty_weight': 0.5,
                'threshold': 0.38  # Slightly lower threshold
            }
        },
        {
            'name': 'Balanced High (k=0.4, thresh=0.32)',
            'params': {
                'k': 0.4,  # Medium-high IG penalty
                'node_weight': 0.4,
                'edge_weight': 0.2,
                'novelty_weight': 0.45,
                'threshold': 0.32  # Between conservative and balanced
            }
        },
        {
            'name': 'Balanced (k=0.3, thresh=0.28)',
            'params': {
                'k': 0.3,  # Moderate IG penalty
                'node_weight': 0.4,
                'edge_weight': 0.2,
                'novelty_weight': 0.5,
                'threshold': 0.28  # Moderate threshold
            }
        },
        {
            'name': 'Liberal (k=0.15, thresh=0.15)',
            'params': {
                'k': 0.15,  # Low IG penalty
                'node_weight': 0.45,
                'edge_weight': 0.25,
                'novelty_weight': 0.4,
                'threshold': 0.15  # Low threshold - accepts more
            }
        },
        {
            'name': 'Very Liberal (k=0.05, thresh=0.05)',
            'params': {
                'k': 0.05,  # Very low k
                'node_weight': 0.5,
                'edge_weight': 0.3,
                'novelty_weight': 0.3,
                'threshold': 0.05  # Very low threshold - accepts most
            }
        },
        {
            'name': 'Novelty-focused (k=0.25, thresh=0.22)',
            'params': {
                'k': 0.25,
                'node_weight': 0.3,
                'edge_weight': 0.15,
                'novelty_weight': 0.8,  # High novelty weight
                'threshold': 0.22  # Novelty-focused threshold
            }
        },
        {
            'name': 'Structure-focused (k=0.35, thresh=0.30)',
            'params': {
                'k': 0.35,
                'node_weight': 0.6,  # High node weight
                'edge_weight': 0.4,  # High edge weight
                'novelty_weight': 0.3,
                'threshold': 0.30  # Structure-focused threshold
            }
        },
        {
            'name': 'IG-suppressed (k=0.01, thresh=0.10)',
            'params': {
                'k': 0.01,  # Almost no IG penalty
                'node_weight': 0.4,
                'edge_weight': 0.2,
                'novelty_weight': 0.5,
                'threshold': 0.10  # IG-suppressed threshold
            }
        }
    ]
    
    # Get query categories
    query_categories = categorize_queries()
    
    # Setup
    config = ExperimentConfig()
    knowledge_base = create_high_quality_knowledge_base()
    test_queries = create_meaningful_queries()
    
    results = []
    
    for config_data in configurations:
        print(f"\n📊 Testing: {config_data['name']}")
        print("-" * 50)
        
        # Create system with fixed parameters
        system = FixedParameterGeDIGSystem(config, config_data['params'])
        system.add_initial_knowledge(knowledge_base)
        
        # Process all queries
        decisions = []
        debug_scores = []  # Track actual scores for debugging
        for i, (query, depth) in enumerate(test_queries):
            result = system.process_query(query, depth)
            updated = result.get('updated', False)
            decisions.append(updated)
            
            # Debug print for first query of first config
            if i == 0 and config_data['name'] == 'Very Liberal (k=0.05, thresh=0.05)':
                print(f"\n🔍 Debug first query result:")
                print(f"  Query: {query[:50]}")
                print(f"  Result keys: {result.keys()}")
                print(f"  Updated: {updated}")
                if 'update_metadata' in result:
                    print(f"  Update metadata: {result['update_metadata']}")
                if hasattr(system, 'decision_log') and system.decision_log:
                    print(f"  Last decision log: {system.decision_log[-1]}")
            
            # Track the actual geDIG score
            if 'metadata' in result:
                debug_scores.append({
                    'query': query[:50],
                    'gedig': result['metadata'].get('gedig_score', 'N/A'),
                    'threshold': config_data['params']['threshold']
                })
        
        # Print first few scores for debugging
        if debug_scores and config_data['name'] == 'Very Liberal (k=0.05, thresh=0.05)':
            print("\n🔍 Debug: Sample geDIG scores vs threshold:")
            for score_info in debug_scores[:5]:
                print(f"  Query: {score_info['query'][:30]}...")
                print(f"    geDIG score: {score_info['gedig']}")
                print(f"    Threshold: {score_info['threshold']:.2f}")
                if score_info['gedig'] != 'N/A':
                    print(f"    Would accept: {score_info['gedig'] > score_info['threshold']}")
        
        # Analyze decisions by value category
        high_value_accepted = 0
        low_value_rejected = 0
        medium_value_decisions = []
        
        for i, decision in enumerate(decisions):
            cat = query_categories[i]
            if cat['value'] == 'HIGH':
                if decision:
                    high_value_accepted += 1
            elif cat['value'] == 'LOW':
                if not decision:
                    low_value_rejected += 1
            else:  # MEDIUM
                medium_value_decisions.append(decision)
        
        # Calculate metrics
        high_value_total = sum(1 for c in query_categories if c['value'] == 'HIGH')
        low_value_total = sum(1 for c in query_categories if c['value'] == 'LOW')
        
        high_acceptance_rate = high_value_accepted / high_value_total * 100
        low_rejection_rate = low_value_rejected / low_value_total * 100
        medium_acceptance_rate = sum(medium_value_decisions) / len(medium_value_decisions) * 100 if medium_value_decisions else 0
        
        # Overall quality score
        quality_score = (high_acceptance_rate + low_rejection_rate) / 2
        
        # Store results (convert numpy types to native Python types)
        result_data = {
            'config': config_data['name'],
            'k': float(config_data['params']['k']),
            'threshold': float(config_data['params']['threshold']),
            'high_value_accepted': int(high_value_accepted),
            'high_value_total': int(high_value_total),
            'high_acceptance_rate': float(high_acceptance_rate),
            'low_value_rejected': int(low_value_rejected),
            'low_value_total': int(low_value_total),
            'low_rejection_rate': float(low_rejection_rate),
            'medium_acceptance_rate': float(medium_acceptance_rate),
            'total_accepted': int(sum(decisions)),
            'total_queries': len(decisions),
            'overall_rate': float(sum(decisions) / len(decisions) * 100),
            'quality_score': float(quality_score),
            'decision_log': system.decision_log  # Keep for analysis but don't save to JSON
        }
        
        results.append(result_data)
        
        # Print summary
        print(f"  High-value acceptance: {high_value_accepted}/{high_value_total} ({high_acceptance_rate:.0f}%)")
        print(f"  Low-value rejection: {low_value_rejected}/{low_value_total} ({low_rejection_rate:.0f}%)")
        print(f"  Medium-value acceptance: {medium_acceptance_rate:.0f}%")
        print(f"  Overall acceptance: {sum(decisions)}/{len(decisions)} ({result_data['overall_rate']:.0f}%)")
        print(f"  📊 Quality Score: {quality_score:.0f}%")
    
    return results, query_categories


def analyze_trends(results):
    """Analyze trends across parameter settings."""
    print("\n" + "=" * 60)
    print("📈 TREND ANALYSIS")
    print("=" * 60)
    
    # Sort by quality score
    results_sorted = sorted(results, key=lambda x: x['quality_score'], reverse=True)
    
    print("\n🏆 Ranking by Quality Score:")
    print("-" * 50)
    for i, r in enumerate(results_sorted, 1):
        print(f"{i}. {r['config']:40s} Score: {r['quality_score']:.0f}%")
        print(f"   k={r['k']:.2f}, threshold={r['threshold']:.2f}")
        print(f"   High-value: {r['high_acceptance_rate']:.0f}%, Low-value: {r['low_rejection_rate']:.0f}%")
    
    # Analyze correlations
    print("\n🔍 Parameter Correlations:")
    print("-" * 50)
    
    k_values = [r['k'] for r in results]
    quality_scores = [r['quality_score'] for r in results]
    high_accept_rates = [r['high_acceptance_rate'] for r in results]
    
    # Simple correlation analysis
    k_quality_corr = np.corrcoef(k_values, quality_scores)[0, 1]
    k_high_corr = np.corrcoef(k_values, high_accept_rates)[0, 1]
    
    print(f"  k vs Quality Score correlation: {k_quality_corr:.3f}")
    print(f"  k vs High-value acceptance correlation: {k_high_corr:.3f}")
    
    # Key findings
    print("\n💡 Key Findings:")
    print("-" * 50)
    
    best_config = results_sorted[0]
    worst_config = results_sorted[-1]
    
    print(f"  Best configuration: {best_config['config']}")
    print(f"    - Accepts {best_config['high_acceptance_rate']:.0f}% of valuable queries")
    print(f"    - Rejects {best_config['low_rejection_rate']:.0f}% of redundant queries")
    
    print(f"\n  Worst configuration: {worst_config['config']}")
    print(f"    - Accepts only {worst_config['high_acceptance_rate']:.0f}% of valuable queries")
    print(f"    - Rejects {worst_config['low_rejection_rate']:.0f}% of redundant queries")
    
    # Patterns
    print("\n📊 Observed Patterns:")
    print("-" * 50)
    
    if k_quality_corr < -0.3:
        print("  • Lower k values tend to produce better quality decisions")
    elif k_quality_corr > 0.3:
        print("  • Higher k values tend to produce better quality decisions")
    else:
        print("  • k value alone doesn't strongly determine quality")
    
    # Check if any configuration achieves good balance
    good_configs = [r for r in results if r['high_acceptance_rate'] >= 70 and r['low_rejection_rate'] >= 70]
    if good_configs:
        print(f"\n  ✅ {len(good_configs)} configurations achieve good balance (>70% on both metrics):")
        for gc in good_configs:
            print(f"     - {gc['config']}")
    else:
        print("\n  ⚠️ No configuration achieves >70% on both high-value acceptance and low-value rejection")
    
    return results_sorted


def visualize_results(results, query_categories):
    """Create visualizations of parameter quality study."""
    print("\n📊 Creating Visualizations...")
    
    output_dir = Path("../results/parameter_quality")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Quality Score Comparison
    ax = axes[0, 0]
    configs = [r['config'].split('(')[0].strip() for r in results]
    quality_scores = [r['quality_score'] for r in results]
    
    colors = ['green' if s >= 70 else 'orange' if s >= 50 else 'red' for s in quality_scores]
    bars = ax.bar(range(len(configs)), quality_scores, color=colors)
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.set_ylabel('Quality Score (%)')
    ax.set_title('Overall Quality Score by Configuration')
    ax.axhline(y=70, color='green', linestyle='--', alpha=0.5, label='Good (70%)')
    ax.axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='Acceptable (50%)')
    ax.legend()
    
    # 2. High-Value Acceptance vs Low-Value Rejection
    ax = axes[0, 1]
    ax.scatter([r['high_acceptance_rate'] for r in results],
               [r['low_rejection_rate'] for r in results],
               s=100, alpha=0.6)
    
    for i, r in enumerate(results):
        ax.annotate(r['config'].split('(')[0].strip()[:10],
                   (r['high_acceptance_rate'], r['low_rejection_rate']),
                   fontsize=8, alpha=0.7)
    
    ax.set_xlabel('High-Value Acceptance Rate (%)')
    ax.set_ylabel('Low-Value Rejection Rate (%)')
    ax.set_title('Value Discrimination Performance')
    ax.axvline(x=70, color='green', linestyle='--', alpha=0.3)
    ax.axhline(y=70, color='green', linestyle='--', alpha=0.3)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    
    # Add quadrant labels
    ax.text(85, 85, 'Ideal', fontsize=10, color='green', weight='bold')
    ax.text(85, 15, 'Too Liberal', fontsize=10, color='red', alpha=0.5)
    ax.text(15, 85, 'Too Conservative', fontsize=10, color='red', alpha=0.5)
    ax.text(15, 15, 'Poor', fontsize=10, color='red', alpha=0.5)
    
    # 3. k vs Quality Score
    ax = axes[0, 2]
    k_values = [r['k'] for r in results]
    ax.scatter(k_values, quality_scores, s=100, alpha=0.6)
    
    # Fit trend line
    z = np.polyfit(k_values, quality_scores, 1)
    p = np.poly1d(z)
    ax.plot(sorted(k_values), p(sorted(k_values)), "r--", alpha=0.5, label=f'Trend')
    
    ax.set_xlabel('k (IG coefficient)')
    ax.set_ylabel('Quality Score (%)')
    ax.set_title('Impact of k on Quality')
    ax.legend()
    
    # 4. Acceptance Rate by Query Value
    ax = axes[1, 0]
    
    categories = ['High Value', 'Medium Value', 'Low Value']
    x = np.arange(len(categories))
    width = 0.15
    
    for i, r in enumerate(results[:5]):  # Show first 5 configs
        high_acc = r['high_acceptance_rate']
        med_acc = r['medium_acceptance_rate']
        low_acc = 100 - r['low_rejection_rate']  # Acceptance rate for low value
        
        values = [high_acc, med_acc, low_acc]
        ax.bar(x + i * width, values, width, label=r['config'].split('(')[0].strip()[:15])
    
    ax.set_xlabel('Query Value Category')
    ax.set_ylabel('Acceptance Rate (%)')
    ax.set_title('Acceptance Patterns by Query Value')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(categories)
    ax.legend(fontsize=8)
    
    # 5. Threshold vs Overall Acceptance
    ax = axes[1, 1]
    thresholds = [r['threshold'] for r in results]
    overall_rates = [r['overall_rate'] for r in results]
    
    ax.scatter(thresholds, overall_rates, s=100, alpha=0.6)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Overall Acceptance Rate (%)')
    ax.set_title('Threshold Impact on Acceptance Rate')
    
    # Add target zone
    ax.axhspan(30, 40, alpha=0.2, color='green', label='Target (30-40%)')
    ax.legend()
    
    # 6. Configuration Comparison Table
    ax = axes[1, 2]
    ax.axis('off')
    
    # Create comparison table
    table_data = []
    for r in results[:4]:  # Top 4 configs
        table_data.append([
            r['config'].split('(')[0].strip()[:20],
            f"{r['quality_score']:.0f}%",
            f"{r['high_acceptance_rate']:.0f}%",
            f"{r['low_rejection_rate']:.0f}%"
        ])
    
    table = ax.table(cellText=table_data,
                    colLabels=['Configuration', 'Quality', 'High Accept', 'Low Reject'],
                    cellLoc='center',
                    loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    ax.set_title('Top Configurations Summary')
    
    plt.suptitle('Parameter Quality Study: Fixed Parameters Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = output_dir / f"parameter_quality_{timestamp}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Saved visualization to: {fig_path}")
    
    plt.show()
    
    return output_dir


def save_detailed_results(results, query_categories, output_dir):
    """Save detailed results to files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save main results
    results_path = output_dir / f"quality_results_{timestamp}.json"
    
    # Prepare data for saving (remove decision_log for main file)
    save_data = []
    for r in results:
        save_item = {k: v for k, v in r.items() if k != 'decision_log'}
        save_data.append(save_item)
    
    with open(results_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    # Save decision details for best configuration
    best_config = max(results, key=lambda x: x['quality_score'])
    decisions_path = output_dir / f"best_config_decisions_{timestamp}.csv"
    
    decisions_df = pd.DataFrame(best_config['decision_log'])
    decisions_df.to_csv(decisions_path, index=False)
    
    # Save query categories
    categories_path = output_dir / f"query_categories_{timestamp}.csv"
    categories_df = pd.DataFrame(query_categories)
    categories_df.to_csv(categories_path, index=False)
    
    print(f"\n✅ Saved results to: {results_path}")
    print(f"✅ Saved best config decisions to: {decisions_path}")
    print(f"✅ Saved query categories to: {categories_path}")


def main():
    """Main execution."""
    try:
        print("🚀 Starting Fixed Parameter Quality Study")
        print("=" * 60)
        
        # Run parameter study
        results, query_categories = run_parameter_study()
        
        # Analyze trends
        results_sorted = analyze_trends(results)
        
        # Create visualizations
        output_dir = visualize_results(results, query_categories)
        
        # Save results
        save_detailed_results(results, query_categories, output_dir)
        
        print("\n" + "=" * 60)
        print("✅ PARAMETER QUALITY STUDY COMPLETE")
        print("=" * 60)
        
        # Final recommendations
        print("\n🎯 Recommendations:")
        print("-" * 50)
        
        best = results_sorted[0]
        print(f"Best configuration: {best['config']}")
        print(f"  - k = {best['k']}")
        print(f"  - threshold = {best['threshold']}")
        print(f"  - Quality score: {best['quality_score']:.0f}%")
        
        if best['quality_score'] >= 70:
            print("\n✅ This configuration provides good value discrimination!")
        else:
            print("\n⚠️ Further tuning needed for optimal value discrimination.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Study failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)