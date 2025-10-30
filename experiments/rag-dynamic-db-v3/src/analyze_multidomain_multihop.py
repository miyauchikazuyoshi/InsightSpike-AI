#!/usr/bin/env python3
"""Analyze multi-hop geDIG with multi-domain knowledge base."""

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
from run_experiment_improved import ExperimentConfig

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


def analyze_domain_hops(system, query_result):
    """Analyze which domains are involved in multi-hop reasoning."""
    domain_hops = defaultdict(int)
    
    if hasattr(system, 'decision_log') and system.decision_log:
        last_log = system.decision_log[-1]
        
        # Track hop statistics
        hop_stats = {
            '1-hop': last_log.get('edges_1hop', 0),
            '2-hop': last_log.get('edges_2hop', 0),
            '3-hop': last_log.get('edges_3hop', 0),
            'affected_2hop': last_log.get('affected_nodes_2hop', 0)
        }
        
        return hop_stats
    
    return None


def run_multidomain_study():
    """Run multi-hop study with multi-domain knowledge base."""
    print("🌐 Multi-Domain Multi-Hop geDIG Study")
    print("=" * 60)
    
    # Create multi-domain knowledge base
    kb_items = create_multidomain_knowledge_base()
    queries = create_multidomain_queries()
    
    # Convert to format expected by system
    from run_experiment_improved import HighQualityKnowledge
    knowledge_base = []
    for item in kb_items:
        knowledge_base.append(HighQualityKnowledge(
            text=item.text,
            concepts=item.concepts,
            depth=item.depth,
            domain=item.domain
        ))
    
    print(f"📚 Knowledge Base:")
    print(f"  Items: {len(knowledge_base)}")
    print(f"  Domains: {len(set(item.domain for item in kb_items))}")
    print(f"  Cross-domain connections: {sum(len(item.connects_to) for item in kb_items)}")
    
    # Test configurations
    configurations = [
        {
            'name': '1-hop Conservative',
            'params': {
                'k': 0.4,
                'node_weight': 0.35,
                'edge_weight': 0.2,
                'novelty_weight': 0.45,
                'threshold': 0.35,
                'enable_multihop': False,
                'max_hops': 1
            }
        },
        {
            'name': '2-hop Conservative',
            'params': {
                'k': 0.4,
                'node_weight': 0.35,
                'edge_weight': 0.2,
                'edge_weight_2hop': 0.1,
                'novelty_weight': 0.45,
                'threshold': 0.38,  # Slightly higher for multi-hop
                'enable_multihop': True,
                'max_hops': 2,
                'decay_factor': 0.7
            }
        },
        {
            'name': '3-hop Conservative',
            'params': {
                'k': 0.4,
                'node_weight': 0.35,
                'edge_weight': 0.2,
                'edge_weight_2hop': 0.1,
                'edge_weight_3hop': 0.05,
                'novelty_weight': 0.45,
                'threshold': 0.40,  # Even higher for 3-hop
                'enable_multihop': True,
                'max_hops': 3,
                'decay_factor': 0.7
            }
        },
        {
            'name': '2-hop Liberal',
            'params': {
                'k': 0.2,
                'node_weight': 0.4,
                'edge_weight': 0.25,
                'edge_weight_2hop': 0.15,
                'novelty_weight': 0.4,
                'threshold': 0.25,
                'enable_multihop': True,
                'max_hops': 2,
                'decay_factor': 0.6  # Less decay
            }
        }
    ]
    
    config = ExperimentConfig()
    results = []
    detailed_logs = []
    
    for config_data in configurations:
        print(f"\n📊 Testing: {config_data['name']}")
        print("-" * 50)
        
        # Create system
        system = MultiHopGeDIGSystem(config, config_data['params'])
        system.add_initial_knowledge(knowledge_base)
        
        # Track query types
        query_stats = {
            'multi_hop': {'accepted': 0, 'total': 0},
            'cross_domain': {'accepted': 0, 'total': 0},
            'single_domain': {'accepted': 0, 'total': 0},
            'novel': {'accepted': 0, 'total': 0},
            'redundant': {'accepted': 0, 'total': 0}
        }
        
        # Process queries
        decisions = []
        hop_analysis = []
        
        for i, (query, depth) in enumerate(queries):
            result = system.process_query(query, depth)
            updated = result.get('updated', False)
            decisions.append(updated)
            
            # Analyze query type
            query_lower = query.lower()
            if '->' in query or 'relate' in query_lower or 'compare' in query_lower:
                query_stats['multi_hop']['total'] += 1
                if updated:
                    query_stats['multi_hop']['accepted'] += 1
            
            if 'bridge' in query_lower or 'cross' in query_lower:
                query_stats['cross_domain']['total'] += 1
                if updated:
                    query_stats['cross_domain']['accepted'] += 1
            
            if depth == 'basic' and not any(word in query_lower for word in ['relate', 'compare', 'how']):
                query_stats['single_domain']['total'] += 1
                if updated:
                    query_stats['single_domain']['accepted'] += 1
            
            # Check if redundant (basic questions about existing concepts)
            is_redundant = any(concept in query_lower 
                              for item in kb_items 
                              for concept in item.concepts)
            if is_redundant and depth == 'basic':
                query_stats['redundant']['total'] += 1
                if updated:
                    query_stats['redundant']['accepted'] += 1
            elif depth == 'advanced':
                query_stats['novel']['total'] += 1
                if updated:
                    query_stats['novel']['accepted'] += 1
            
            # Collect hop statistics
            hop_stats = analyze_domain_hops(system, result)
            if hop_stats:
                hop_analysis.append(hop_stats)
            
            # Debug output for interesting queries
            if i < 3 and config_data['name'] == '3-hop Conservative':
                print(f"\n  Query {i+1}: {query[:60]}...")
                if system.decision_log:
                    log = system.decision_log[-1]
                    print(f"    1-hop edges: {log.get('edges_1hop', 0)}")
                    print(f"    2-hop impact: {log.get('edges_2hop', 0):.2f}")
                    print(f"    3-hop impact: {log.get('edges_3hop', 0):.2f}")
                    print(f"    geDIG score: {log.get('gedig_score', 0):.3f}")
                    print(f"    Decision: {'✅ Accept' if log.get('decision') else '❌ Reject'}")
        
        # Calculate statistics
        acceptance_rate = sum(decisions) / len(decisions) * 100
        
        # Multi-hop specific metrics
        multi_hop_rate = (query_stats['multi_hop']['accepted'] / 
                         query_stats['multi_hop']['total'] * 100 
                         if query_stats['multi_hop']['total'] > 0 else 0)
        
        cross_domain_rate = (query_stats['cross_domain']['accepted'] / 
                            query_stats['cross_domain']['total'] * 100 
                            if query_stats['cross_domain']['total'] > 0 else 0)
        
        novel_rate = (query_stats['novel']['accepted'] / 
                     query_stats['novel']['total'] * 100 
                     if query_stats['novel']['total'] > 0 else 0)
        
        redundant_rate = (query_stats['redundant']['accepted'] / 
                         query_stats['redundant']['total'] * 100 
                         if query_stats['redundant']['total'] > 0 else 0)
        
        # Quality score: favor accepting novel and rejecting redundant
        quality_score = (novel_rate + (100 - redundant_rate)) / 2
        
        result_data = {
            'config': config_data['name'],
            'max_hops': config_data['params']['max_hops'],
            'overall_acceptance': acceptance_rate,
            'multi_hop_acceptance': multi_hop_rate,
            'cross_domain_acceptance': cross_domain_rate,
            'novel_acceptance': novel_rate,
            'redundant_acceptance': redundant_rate,
            'quality_score': quality_score,
            'query_stats': query_stats,
            'hop_analysis': hop_analysis
        }
        
        results.append(result_data)
        detailed_logs.append({
            'config': config_data['name'],
            'decision_log': system.decision_log if hasattr(system, 'decision_log') else []
        })
        
        # Print summary
        print(f"\n  📈 Results:")
        print(f"    Overall acceptance: {acceptance_rate:.1f}%")
        print(f"    Multi-hop queries: {multi_hop_rate:.1f}%")
        print(f"    Cross-domain queries: {cross_domain_rate:.1f}%")
        print(f"    Novel acceptance: {novel_rate:.1f}%")
        print(f"    Redundant acceptance: {redundant_rate:.1f}%")
        print(f"    Quality score: {quality_score:.1f}%")
    
    return results, detailed_logs


def analyze_multihop_benefits(results):
    """Analyze benefits of multi-hop in multi-domain setting."""
    print("\n" + "=" * 60)
    print("🔍 MULTI-HOP BENEFITS ANALYSIS")
    print("=" * 60)
    
    # Compare 1-hop vs multi-hop
    one_hop = [r for r in results if r['max_hops'] == 1]
    two_hop = [r for r in results if r['max_hops'] == 2]
    three_hop = [r for r in results if r['max_hops'] == 3]
    
    print("\n📊 Performance by Hop Count:")
    print("-" * 40)
    
    metrics = ['overall_acceptance', 'multi_hop_acceptance', 'cross_domain_acceptance', 
               'novel_acceptance', 'redundant_acceptance', 'quality_score']
    
    for metric in metrics:
        print(f"\n{metric.replace('_', ' ').title()}:")
        if one_hop:
            avg_1 = np.mean([r[metric] for r in one_hop])
            print(f"  1-hop: {avg_1:.1f}%")
        if two_hop:
            avg_2 = np.mean([r[metric] for r in two_hop])
            print(f"  2-hop: {avg_2:.1f}%")
        if three_hop:
            avg_3 = np.mean([r[metric] for r in three_hop])
            print(f"  3-hop: {avg_3:.1f}%")
    
    # Find configuration with best multi-hop performance
    best_multihop = max(results, 
                       key=lambda x: x['multi_hop_acceptance'] if x['max_hops'] > 1 else 0)
    
    print(f"\n🏆 Best Multi-Hop Configuration: {best_multihop['config']}")
    print(f"   Multi-hop acceptance: {best_multihop['multi_hop_acceptance']:.1f}%")
    print(f"   Cross-domain acceptance: {best_multihop['cross_domain_acceptance']:.1f}%")
    print(f"   Quality score: {best_multihop['quality_score']:.1f}%")
    
    # Analyze hop utilization
    print("\n🔗 Hop Utilization Analysis:")
    for result in results:
        if result['hop_analysis'] and result['max_hops'] > 1:
            hop_data = result['hop_analysis']
            avg_2hop = np.mean([h.get('2-hop', 0) for h in hop_data])
            avg_3hop = np.mean([h.get('3-hop', 0) for h in hop_data])
            avg_affected = np.mean([h.get('affected_2hop', 0) for h in hop_data])
            
            print(f"\n  {result['config']}:")
            print(f"    Avg 2-hop edges: {avg_2hop:.2f}")
            if result['max_hops'] >= 3:
                print(f"    Avg 3-hop edges: {avg_3hop:.2f}")
            print(f"    Avg nodes affected at 2-hop: {avg_affected:.1f}")


def create_visualization(results):
    """Create comprehensive visualization of multi-domain multi-hop results."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Quality Score by Configuration
    ax = axes[0, 0]
    configs = [r['config'] for r in results]
    quality_scores = [r['quality_score'] for r in results]
    colors = ['#1f77b4' if '1-hop' in c else '#ff7f0e' if '2-hop' in c else '#2ca02c' 
              for c in configs]
    ax.bar(range(len(configs)), quality_scores, color=colors, alpha=0.7)
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.set_ylabel('Quality Score (%)')
    ax.set_title('Quality Score by Configuration')
    ax.grid(True, alpha=0.3)
    
    # 2. Acceptance Rates Comparison
    ax = axes[0, 1]
    metrics = ['multi_hop_acceptance', 'cross_domain_acceptance', 
               'novel_acceptance', 'redundant_acceptance']
    x = np.arange(len(configs))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = [r[metric] for r in results]
        ax.bar(x + i * width, values, width, 
               label=metric.replace('_', ' ').title(), alpha=0.7)
    
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.set_ylabel('Acceptance Rate (%)')
    ax.set_title('Acceptance Rates by Query Type')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 3. Multi-hop vs Single-hop Performance
    ax = axes[0, 2]
    hop_counts = [r['max_hops'] for r in results]
    quality = [r['quality_score'] for r in results]
    multi_hop_acc = [r['multi_hop_acceptance'] for r in results]
    
    ax.scatter(hop_counts, quality, s=100, alpha=0.6, label='Quality Score')
    ax.scatter(hop_counts, multi_hop_acc, s=100, alpha=0.6, label='Multi-hop Acceptance')
    ax.set_xlabel('Max Hops')
    ax.set_ylabel('Score (%)')
    ax.set_title('Performance vs Hop Count')
    ax.set_xticks([1, 2, 3])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Novel vs Redundant Trade-off
    ax = axes[1, 0]
    novel = [r['novel_acceptance'] for r in results]
    redundant = [r['redundant_acceptance'] for r in results]
    
    for i, (n, r, c) in enumerate(zip(novel, redundant, configs)):
        color = '#1f77b4' if '1-hop' in c else '#ff7f0e' if '2-hop' in c else '#2ca02c'
        ax.scatter(n, 100-r, s=150, alpha=0.7, color=color)
        ax.annotate(c.split()[0], (n, 100-r), fontsize=8, ha='center')
    
    ax.set_xlabel('Novel Acceptance Rate (%)')
    ax.set_ylabel('Redundant Rejection Rate (%)')
    ax.set_title('Novel vs Redundant Trade-off')
    ax.plot([0, 100], [0, 100], 'k--', alpha=0.2)  # Ideal line
    ax.grid(True, alpha=0.3)
    
    # 5. Cross-domain Performance
    ax = axes[1, 1]
    cross_domain = [r['cross_domain_acceptance'] for r in results]
    ax.barh(range(len(configs)), cross_domain, color=colors, alpha=0.7)
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels(configs)
    ax.set_xlabel('Cross-domain Acceptance Rate (%)')
    ax.set_title('Cross-domain Query Performance')
    ax.grid(True, alpha=0.3, axis='x')
    
    # 6. Hop Utilization Heatmap
    ax = axes[1, 2]
    hop_matrix = []
    for r in results:
        if r['hop_analysis']:
            hop_data = r['hop_analysis']
            avg_1hop = np.mean([h.get('1-hop', 0) for h in hop_data])
            avg_2hop = np.mean([h.get('2-hop', 0) for h in hop_data])
            avg_3hop = np.mean([h.get('3-hop', 0) for h in hop_data])
            hop_matrix.append([avg_1hop, avg_2hop, avg_3hop])
        else:
            hop_matrix.append([0, 0, 0])
    
    im = ax.imshow(hop_matrix, aspect='auto', cmap='YlOrRd')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['1-hop', '2-hop', '3-hop'])
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels(configs)
    ax.set_title('Average Hop Utilization')
    plt.colorbar(im, ax=ax)
    
    plt.suptitle('Multi-Domain Multi-Hop geDIG Analysis', fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("../results/multidomain_multihop")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    viz_path = results_dir / f"multidomain_analysis_{timestamp}.png"
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Saved visualization to: {viz_path}")
    
    return viz_path


def save_results(results, detailed_logs):
    """Save analysis results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("../results/multidomain_multihop")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save main results
    results_path = results_dir / f"multidomain_results_{timestamp}.json"
    
    # Convert numpy types for JSON serialization
    clean_results = []
    for r in results:
        clean_r = {}
        for k, v in r.items():
            if isinstance(v, (np.integer, np.floating)):
                clean_r[k] = float(v)
            elif isinstance(v, np.ndarray):
                clean_r[k] = v.tolist()
            elif k == 'hop_analysis':
                # Clean hop analysis data
                clean_hops = []
                for hop in v:
                    clean_hop = {}
                    for hk, hv in hop.items():
                        if isinstance(hv, (np.integer, np.floating)):
                            clean_hop[hk] = float(hv)
                        else:
                            clean_hop[hk] = hv
                    clean_hops.append(clean_hop)
                clean_r[k] = clean_hops
            else:
                clean_r[k] = v
        clean_results.append(clean_r)
    
    with open(results_path, 'w') as f:
        json.dump(clean_results, f, indent=2)
    
    print(f"✅ Saved results to: {results_path}")
    
    # Save detailed logs separately (can be large)
    logs_path = results_dir / f"detailed_logs_{timestamp}.json"
    
    # Clean logs for JSON
    clean_logs = []
    for log_entry in detailed_logs:
        clean_entry = {'config': log_entry['config'], 'decision_log': []}
        for decision in log_entry['decision_log']:
            clean_decision = {}
            for k, v in decision.items():
                if isinstance(v, (np.integer, np.floating, np.bool_)):
                    clean_decision[k] = float(v) if not isinstance(v, np.bool_) else bool(v)
                elif isinstance(v, np.ndarray):
                    clean_decision[k] = v.tolist()
                else:
                    clean_decision[k] = v
            clean_entry['decision_log'].append(clean_decision)
        clean_logs.append(clean_entry)
    
    with open(logs_path, 'w') as f:
        json.dump(clean_logs, f, indent=2)
    
    print(f"✅ Saved detailed logs to: {logs_path}")


def main():
    """Run multi-domain multi-hop analysis."""
    print("\n🚀 Starting Multi-Domain Multi-Hop Analysis")
    print("=" * 60)
    
    try:
        # Run the study
        results, detailed_logs = run_multidomain_study()
        
        # Analyze benefits
        analyze_multihop_benefits(results)
        
        # Create visualization
        create_visualization(results)
        
        # Save results
        save_results(results, detailed_logs)
        
        print("\n" + "=" * 60)
        print("✅ MULTI-DOMAIN ANALYSIS COMPLETE")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())