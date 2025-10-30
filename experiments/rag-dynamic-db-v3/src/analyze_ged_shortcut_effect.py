#!/usr/bin/env python3
"""Analyze GED shortcut effect in multi-hop evaluation."""

import json
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd

from multidomain_knowledge_base import (
    create_multidomain_knowledge_base,
    create_multidomain_queries
)
from analyze_threshold_sensitivity import ThresholdSensitiveMultiHopSystem
from run_experiment_improved import ExperimentConfig, HighQualityKnowledge

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 12)


class DetailedGEDTrackingSystem(ThresholdSensitiveMultiHopSystem):
    """System that tracks detailed GED/IG values for each hop level."""
    
    def _evaluate_with_gedig(self, query: str, response: str, 
                            similar_nodes: list) -> tuple:
        """Multi-hop geDIG evaluation with detailed GED/IG tracking."""
        # Simulate adding new node
        new_node_id = f"hypothetical_{self.queries_processed}"
        
        # === 1-HOP EVALUATION ===
        edges_1hop = 0
        connected_nodes_1hop = []
        for node_id, similarity in similar_nodes[:5]:
            if similarity > self.edge_threshold_1hop:
                edges_1hop += 1
                connected_nodes_1hop.append((node_id, similarity))
        
        # Calculate 1-hop only GED/IG
        nodes_added = 1
        ged_1hop_only = (nodes_added * self.params['node_weight'] + 
                        edges_1hop * self.params['edge_weight'])
        
        max_similarity = similar_nodes[0][1] if similar_nodes else 0.0
        novelty = 1.0 - max_similarity
        connectivity_1hop = edges_1hop * 0.1
        ig_1hop_only = novelty * self.params['novelty_weight'] + connectivity_1hop
        
        gedig_1hop_only = ged_1hop_only - self.params['k'] * ig_1hop_only
        
        # === 2-HOP EVALUATION ===
        edges_2hop = 0
        affected_nodes_2hop = set()
        decay_factor = self.params.get('decay_factor', 0.7)
        
        if self.params.get('enable_multihop', True) and self.params.get('max_hops', 2) >= 2:
            for node_id, sim1 in connected_nodes_1hop:
                node_embedding = self.knowledge_graph.nodes[node_id].embedding
                neighbors = self.knowledge_graph.find_similar_nodes(
                    node_embedding, k=5
                )
                
                for neighbor_id, neighbor_sim in neighbors:
                    if neighbor_id != node_id and neighbor_id != new_node_id:
                        if neighbor_sim > self.edge_threshold_2hop:
                            impact = sim1 * neighbor_sim * decay_factor
                            if impact > self.params.get('min_impact_2hop', 0.1):
                                affected_nodes_2hop.add(neighbor_id)
                                edges_2hop += impact
        
        # Calculate 2-hop cumulative GED/IG
        ged_2hop = (nodes_added * self.params['node_weight'] + 
                   edges_1hop * self.params['edge_weight'] +
                   edges_2hop * self.params.get('edge_weight_2hop', self.params['edge_weight'] * 0.5))
        
        connectivity_2hop = edges_1hop * 0.1 + len(affected_nodes_2hop) * 0.05
        ig_2hop = novelty * self.params['novelty_weight'] + connectivity_2hop
        
        gedig_2hop = ged_2hop - self.params['k'] * ig_2hop
        
        # === 3-HOP EVALUATION ===
        edges_3hop = 0
        affected_nodes_3hop = set()
        shortcut_detected = False
        
        if self.params.get('max_hops', 2) >= 3 and self.params.get('enable_multihop', True):
            decay_3hop = decay_factor ** 2
            
            # Check for shortcuts - nodes that connect back to 1-hop nodes
            shortcuts_found = []
            
            for node_id in affected_nodes_2hop:
                node_embedding = self.knowledge_graph.nodes[node_id].embedding
                far_neighbors = self.knowledge_graph.find_similar_nodes(
                    node_embedding, k=3
                )
                
                for far_id, far_sim in far_neighbors:
                    # Check if this creates a shortcut to original nodes
                    if far_id in [n[0] for n in connected_nodes_1hop[:2]]:
                        shortcuts_found.append((node_id, far_id, far_sim))
                        shortcut_detected = True
                    
                    if far_id not in affected_nodes_2hop and far_id not in [n[0] for n in connected_nodes_1hop]:
                        if far_sim > self.edge_threshold_3hop:
                            impact = far_sim * decay_3hop * 0.5
                            if impact > self.params.get('min_impact_3hop', 0.05):
                                affected_nodes_3hop.add(far_id)
                                edges_3hop += impact
        
        # Calculate 3-hop cumulative GED/IG
        ged_3hop = (nodes_added * self.params['node_weight'] + 
                   edges_1hop * self.params['edge_weight'] +
                   edges_2hop * self.params.get('edge_weight_2hop', self.params['edge_weight'] * 0.5) +
                   edges_3hop * self.params.get('edge_weight_3hop', self.params['edge_weight'] * 0.25))
        
        connectivity_3hop = (edges_1hop * 0.1 + 
                            len(affected_nodes_2hop) * 0.05 +
                            len(affected_nodes_3hop) * 0.02)
        ig_3hop = novelty * self.params['novelty_weight'] + connectivity_3hop
        
        gedig_3hop = ged_3hop - self.params['k'] * ig_3hop
        
        # === SHORTCUT EFFECT ===
        # If shortcuts are found, GED might actually decrease due to redundant paths
        if shortcut_detected:
            # Reduce GED due to shortcut (graph becomes more connected, less change needed)
            shortcut_reduction = len(shortcuts_found) * 0.1 * self.params['edge_weight']
            ged_3hop_adjusted = ged_3hop - shortcut_reduction
            gedig_3hop_adjusted = ged_3hop_adjusted - self.params['k'] * ig_3hop
        else:
            ged_3hop_adjusted = ged_3hop
            gedig_3hop_adjusted = gedig_3hop
        
        # Use final values for decision
        final_ged = ged_3hop_adjusted if self.params.get('max_hops', 1) >= 3 else (
                   ged_2hop if self.params.get('max_hops', 1) >= 2 else ged_1hop_only)
        final_ig = ig_3hop if self.params.get('max_hops', 1) >= 3 else (
                  ig_2hop if self.params.get('max_hops', 1) >= 2 else ig_1hop_only)
        final_gedig = gedig_3hop_adjusted if self.params.get('max_hops', 1) >= 3 else (
                     gedig_2hop if self.params.get('max_hops', 1) >= 2 else gedig_1hop_only)
        
        threshold = self.params['threshold']
        should_update = final_gedig > threshold
        
        # Detailed logging
        self.decision_log.append({
            'query': query[:80],
            'novelty': novelty,
            'similarity': max_similarity,
            # 1-hop values
            'edges_1hop': edges_1hop,
            'ged_1hop': ged_1hop_only,
            'ig_1hop': ig_1hop_only,
            'gedig_1hop': gedig_1hop_only,
            # 2-hop values
            'edges_2hop': float(edges_2hop),
            'ged_2hop': ged_2hop,
            'ig_2hop': ig_2hop,
            'gedig_2hop': gedig_2hop,
            'affected_2hop': len(affected_nodes_2hop),
            # 3-hop values
            'edges_3hop': float(edges_3hop),
            'ged_3hop': ged_3hop,
            'ig_3hop': ig_3hop,
            'gedig_3hop': gedig_3hop,
            'affected_3hop': len(affected_nodes_3hop),
            # Shortcut detection
            'shortcut_detected': shortcut_detected,
            'shortcuts_count': len(shortcuts_found) if shortcut_detected else 0,
            'ged_3hop_adjusted': ged_3hop_adjusted,
            'gedig_3hop_adjusted': gedig_3hop_adjusted,
            # Delta values (changes between hops)
            'delta_ged_1to2': ged_2hop - ged_1hop_only,
            'delta_ged_2to3': ged_3hop - ged_2hop,
            'delta_ig_1to2': ig_2hop - ig_1hop_only,
            'delta_ig_2to3': ig_3hop - ig_2hop,
            'delta_gedig_1to2': gedig_2hop - gedig_1hop_only,
            'delta_gedig_2to3': gedig_3hop - gedig_2hop,
            # Final values
            'final_ged': final_ged,
            'final_ig': final_ig,
            'final_gedig': final_gedig,
            'threshold': threshold,
            'decision': should_update
        })
        
        # Store metrics
        self.gedig_scores.append(final_gedig)
        self.ig_values.append(final_ig)
        self.ged_values.append(final_ged)
        
        metadata = {
            'gedig_score': final_gedig,
            'ged': final_ged,
            'ig': final_ig,
            'novelty': novelty,
            'threshold_used': threshold,
            'shortcut_detected': shortcut_detected,
            'ged_values': {
                '1hop': ged_1hop_only,
                '2hop': ged_2hop,
                '3hop': ged_3hop,
                '3hop_adjusted': ged_3hop_adjusted
            },
            'ig_values': {
                '1hop': ig_1hop_only,
                '2hop': ig_2hop,
                '3hop': ig_3hop
            },
            'gedig_values': {
                '1hop': gedig_1hop_only,
                '2hop': gedig_2hop,
                '3hop': gedig_3hop,
                '3hop_adjusted': gedig_3hop_adjusted
            }
        }
        
        return should_update, metadata


def run_ged_shortcut_analysis():
    """Run analysis focusing on GED shortcut effects."""
    print("🔬 GED Shortcut Effect Analysis")
    print("=" * 60)
    
    # Create multi-domain knowledge base
    kb_items = create_multidomain_knowledge_base()
    queries = create_multidomain_queries()
    
    knowledge_base = []
    for item in kb_items:
        knowledge_base.append(HighQualityKnowledge(
            text=item.text,
            concepts=item.concepts,
            depth=item.depth,
            domain=item.domain
        ))
    
    print(f"📚 Knowledge Base: {len(knowledge_base)} items")
    
    # Test configurations focusing on multi-hop effects
    configurations = [
        {
            'name': '1-hop Standard',
            'params': {
                'k': 0.3,
                'node_weight': 0.4,
                'edge_weight': 0.25,
                'novelty_weight': 0.45,
                'threshold': 0.25,
                'enable_multihop': False,
                'max_hops': 1,
                'edge_threshold_1hop': 0.15,
                'edge_threshold_2hop': 0.20,
                'edge_threshold_3hop': 0.25
            }
        },
        {
            'name': '2-hop Standard',
            'params': {
                'k': 0.3,
                'node_weight': 0.4,
                'edge_weight': 0.25,
                'edge_weight_2hop': 0.15,
                'novelty_weight': 0.45,
                'threshold': 0.25,
                'enable_multihop': True,
                'max_hops': 2,
                'decay_factor': 0.7,
                'edge_threshold_1hop': 0.15,
                'edge_threshold_2hop': 0.20,
                'edge_threshold_3hop': 0.25,
                'min_impact_2hop': 0.1
            }
        },
        {
            'name': '3-hop Standard',
            'params': {
                'k': 0.3,
                'node_weight': 0.4,
                'edge_weight': 0.25,
                'edge_weight_2hop': 0.15,
                'edge_weight_3hop': 0.08,
                'novelty_weight': 0.45,
                'threshold': 0.25,
                'enable_multihop': True,
                'max_hops': 3,
                'decay_factor': 0.7,
                'edge_threshold_1hop': 0.15,
                'edge_threshold_2hop': 0.20,
                'edge_threshold_3hop': 0.25,
                'min_impact_2hop': 0.1,
                'min_impact_3hop': 0.05
            }
        },
        {
            'name': '3-hop Relaxed',
            'params': {
                'k': 0.3,
                'node_weight': 0.4,
                'edge_weight': 0.25,
                'edge_weight_2hop': 0.15,
                'edge_weight_3hop': 0.08,
                'novelty_weight': 0.45,
                'threshold': 0.20,
                'enable_multihop': True,
                'max_hops': 3,
                'decay_factor': 0.7,
                'edge_threshold_1hop': 0.05,
                'edge_threshold_2hop': 0.08,
                'edge_threshold_3hop': 0.10,
                'min_impact_2hop': 0.05,
                'min_impact_3hop': 0.02
            }
        }
    ]
    
    config = ExperimentConfig()
    all_results = []
    
    for config_data in configurations:
        print(f"\n📊 Testing: {config_data['name']}")
        print("-" * 50)
        
        # Create system
        system = DetailedGEDTrackingSystem(config, config_data['params'])
        system.add_initial_knowledge(knowledge_base)
        
        # Process queries
        for i, (query, depth) in enumerate(queries[:10]):  # Focus on first 10 queries
            result = system.process_query(query, depth)
            
            # Extract detailed metrics
            if system.decision_log:
                log = system.decision_log[-1]
                
                # Print interesting cases
                if config_data['name'] == '3-hop Relaxed' and i < 5:
                    print(f"\n  Query {i+1}: {query[:50]}...")
                    print(f"    GED: 1hop={log['ged_1hop']:.3f}, "
                          f"2hop={log['ged_2hop']:.3f}, "
                          f"3hop={log['ged_3hop']:.3f}")
                    
                    if log['shortcut_detected']:
                        print(f"    🔗 SHORTCUT DETECTED! Adjusted GED: {log['ged_3hop_adjusted']:.3f}")
                    
                    print(f"    IG:  1hop={log['ig_1hop']:.3f}, "
                          f"2hop={log['ig_2hop']:.3f}, "
                          f"3hop={log['ig_3hop']:.3f}")
                    print(f"    geDIG: 1hop={log['gedig_1hop']:.3f}, "
                           f"2hop={log['gedig_2hop']:.3f}, "
                           f"3hop={log['gedig_3hop']:.3f}")
                    
                    # Show deltas
                    delta_ged_2to3 = log['delta_ged_2to3']
                    if abs(delta_ged_2to3) < 0.01 and log['edges_3hop'] > 0:
                        print(f"    ⚠️ GED PLATEAU: 3-hop adds edges but GED barely changes!")
                    elif delta_ged_2to3 < -0.05:
                        print(f"    📉 GED DROP: Multi-hop reduces GED by {-delta_ged_2to3:.3f}")
                
                # Collect all results
                all_results.append({
                    'config': config_data['name'],
                    'query_idx': i,
                    'query': query[:50],
                    **log
                })
    
    return all_results


def analyze_shortcut_patterns(results):
    """Analyze patterns in GED shortcuts."""
    print("\n" + "=" * 60)
    print("🔍 SHORTCUT PATTERN ANALYSIS")
    print("=" * 60)
    
    df = pd.DataFrame(results)
    
    # 1. Shortcut detection rate
    print("\n📊 Shortcut Detection Rate by Configuration:")
    for config in df['config'].unique():
        config_data = df[df['config'] == config]
        shortcut_rate = config_data['shortcut_detected'].mean() * 100
        shortcuts_count = config_data['shortcuts_count'].sum()
        print(f"  {config}: {shortcut_rate:.1f}% queries, {shortcuts_count} total shortcuts")
    
    # 2. GED progression analysis
    print("\n📈 Average GED Progression:")
    for config in df['config'].unique():
        config_data = df[df['config'] == config]
        avg_ged_1hop = config_data['ged_1hop'].mean()
        avg_ged_2hop = config_data['ged_2hop'].mean()
        avg_ged_3hop = config_data['ged_3hop'].mean()
        
        print(f"\n  {config}:")
        print(f"    1-hop: {avg_ged_1hop:.3f}")
        if '2-hop' in config or '3-hop' in config:
            print(f"    2-hop: {avg_ged_2hop:.3f} (Δ={avg_ged_2hop-avg_ged_1hop:+.3f})")
        if '3-hop' in config:
            print(f"    3-hop: {avg_ged_3hop:.3f} (Δ={avg_ged_3hop-avg_ged_2hop:+.3f})")
    
    # 3. Find queries with significant GED drops
    print("\n⚡ Queries with Significant GED Drops (2→3 hop):")
    three_hop_data = df[df['config'].str.contains('3-hop')]
    
    for _, row in three_hop_data.iterrows():
        delta = row['delta_ged_2to3']
        if delta < -0.01:  # Significant drop
            print(f"  {row['query'][:40]}...")
            print(f"    Config: {row['config']}")
            print(f"    GED drop: {delta:.3f}")
            if row['shortcut_detected']:
                print(f"    Shortcuts found: {row['shortcuts_count']}")
    
    # 4. IG vs GED relationship
    print("\n🔄 IG vs GED Correlation:")
    for config in df['config'].unique():
        config_data = df[df['config'] == config]
        if len(config_data) > 1:
            corr_1hop = config_data[['ged_1hop', 'ig_1hop']].corr().iloc[0, 1]
            print(f"  {config} (1-hop): {corr_1hop:.3f}")
            
            if '2-hop' in config or '3-hop' in config:
                corr_2hop = config_data[['ged_2hop', 'ig_2hop']].corr().iloc[0, 1]
                print(f"  {config} (2-hop): {corr_2hop:.3f}")
            
            if '3-hop' in config:
                corr_3hop = config_data[['ged_3hop', 'ig_3hop']].corr().iloc[0, 1]
                print(f"  {config} (3-hop): {corr_3hop:.3f}")
    
    # 5. GED plateau detection
    print("\n🏔️ GED Plateau Detection (minimal change despite new edges):")
    plateau_count = 0
    for _, row in df.iterrows():
        if '3-hop' in row['config']:
            if abs(row['delta_ged_2to3']) < 0.01 and row['edges_3hop'] > 0:
                plateau_count += 1
                if plateau_count <= 3:  # Show first 3 examples
                    print(f"  {row['query'][:40]}...")
                    print(f"    3-hop edges: {row['edges_3hop']:.3f}, GED change: {row['delta_ged_2to3']:.4f}")
    
    if plateau_count > 3:
        print(f"  ... and {plateau_count - 3} more cases")


def create_ged_visualization(results):
    """Create comprehensive GED analysis visualization."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    
    df = pd.DataFrame(results)
    
    # 1. GED Progression by Hop Count
    ax = axes[0, 0]
    for config in df['config'].unique():
        config_data = df[df['config'] == config]
        
        x = []
        y_mean = []
        y_std = []
        
        if '1-hop' in config or '2-hop' in config or '3-hop' in config:
            x.append(1)
            y_mean.append(config_data['ged_1hop'].mean())
            y_std.append(config_data['ged_1hop'].std())
        
        if '2-hop' in config or '3-hop' in config:
            x.append(2)
            y_mean.append(config_data['ged_2hop'].mean())
            y_std.append(config_data['ged_2hop'].std())
        
        if '3-hop' in config:
            x.append(3)
            y_mean.append(config_data['ged_3hop'].mean())
            y_std.append(config_data['ged_3hop'].std())
        
        ax.errorbar(x, y_mean, yerr=y_std, marker='o', label=config, alpha=0.7, capsize=5)
    
    ax.set_xlabel('Hop Count')
    ax.set_ylabel('Average GED')
    ax.set_title('GED Progression with Multi-Hop')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 2. IG Progression by Hop Count
    ax = axes[0, 1]
    for config in df['config'].unique():
        config_data = df[df['config'] == config]
        
        x = []
        y_mean = []
        
        if '1-hop' in config or '2-hop' in config or '3-hop' in config:
            x.append(1)
            y_mean.append(config_data['ig_1hop'].mean())
        
        if '2-hop' in config or '3-hop' in config:
            x.append(2)
            y_mean.append(config_data['ig_2hop'].mean())
        
        if '3-hop' in config:
            x.append(3)
            y_mean.append(config_data['ig_3hop'].mean())
        
        ax.plot(x, y_mean, marker='s', label=config, alpha=0.7)
    
    ax.set_xlabel('Hop Count')
    ax.set_ylabel('Average IG')
    ax.set_title('IG Progression with Multi-Hop')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 3. geDIG Score Progression
    ax = axes[0, 2]
    for config in df['config'].unique():
        config_data = df[df['config'] == config]
        
        x = []
        y_mean = []
        
        if '1-hop' in config or '2-hop' in config or '3-hop' in config:
            x.append(1)
            y_mean.append(config_data['gedig_1hop'].mean())
        
        if '2-hop' in config or '3-hop' in config:
            x.append(2)
            y_mean.append(config_data['gedig_2hop'].mean())
        
        if '3-hop' in config:
            x.append(3)
            y_mean.append(config_data['gedig_3hop'].mean())
        
        ax.plot(x, y_mean, marker='^', label=config, alpha=0.7)
    
    ax.set_xlabel('Hop Count')
    ax.set_ylabel('Average geDIG Score')
    ax.set_title('geDIG Score Progression')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 4. Delta GED Distribution (1→2 hop)
    ax = axes[1, 0]
    multi_hop_data = df[df['config'].str.contains('2-hop|3-hop')]
    ax.hist(multi_hop_data['delta_ged_1to2'], bins=20, alpha=0.6, color='blue', edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('ΔGED (1→2 hop)')
    ax.set_ylabel('Frequency')
    ax.set_title('GED Change Distribution (1→2 hop)')
    
    # 5. Delta GED Distribution (2→3 hop)
    ax = axes[1, 1]
    three_hop_data = df[df['config'].str.contains('3-hop')]
    if not three_hop_data.empty:
        ax.hist(three_hop_data['delta_ged_2to3'], bins=20, alpha=0.6, color='green', edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', alpha=0.5)
        
        # Mark shortcut region
        shortcut_region = three_hop_data[three_hop_data['shortcut_detected']]
        if not shortcut_region.empty:
            ax.axvspan(shortcut_region['delta_ged_2to3'].min(), 
                      shortcut_region['delta_ged_2to3'].max(),
                      alpha=0.2, color='orange', label='Shortcut region')
    
    ax.set_xlabel('ΔGED (2→3 hop)')
    ax.set_ylabel('Frequency')
    ax.set_title('GED Change Distribution (2→3 hop)')
    ax.legend()
    
    # 6. GED vs Edges Scatter
    ax = axes[1, 2]
    for config in ['3-hop Standard', '3-hop Relaxed']:
        if config in df['config'].values:
            config_data = df[df['config'] == config]
            total_edges = (config_data['edges_1hop'] + 
                          config_data['edges_2hop'] + 
                          config_data['edges_3hop'])
            ax.scatter(total_edges, config_data['ged_3hop'], 
                      label=config, alpha=0.6, s=50)
    
    ax.set_xlabel('Total Edges (all hops)')
    ax.set_ylabel('Final GED')
    ax.set_title('GED vs Total Edge Count')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 7. Shortcut Effect Visualization
    ax = axes[2, 0]
    shortcut_data = df[df['shortcut_detected'] == True]
    no_shortcut_data = df[(df['shortcut_detected'] == False) & (df['config'].str.contains('3-hop'))]
    
    if not shortcut_data.empty:
        ax.scatter(shortcut_data['ged_3hop'], shortcut_data['ged_3hop_adjusted'],
                  color='red', alpha=0.6, label='With shortcut', s=100)
    if not no_shortcut_data.empty:
        ax.scatter(no_shortcut_data['ged_3hop'], no_shortcut_data['ged_3hop_adjusted'],
                  color='blue', alpha=0.4, label='No shortcut', s=50)
    
    # Add diagonal line
    if not df.empty:
        max_ged = max(df['ged_3hop'].max(), df['ged_3hop_adjusted'].max())
        ax.plot([0, max_ged], [0, max_ged], 'k--', alpha=0.3)
    
    ax.set_xlabel('GED (3-hop)')
    ax.set_ylabel('GED (3-hop adjusted)')
    ax.set_title('Shortcut Effect on GED')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 8. Per-Query GED Evolution
    ax = axes[2, 1]
    # Show first 5 queries for 3-hop Relaxed
    relaxed_data = df[df['config'] == '3-hop Relaxed'].head(5)
    if not relaxed_data.empty:
        x = np.arange(len(relaxed_data))
        width = 0.25
        
        ax.bar(x - width, relaxed_data['ged_1hop'], width, label='1-hop', alpha=0.8)
        ax.bar(x, relaxed_data['ged_2hop'], width, label='2-hop', alpha=0.8)
        ax.bar(x + width, relaxed_data['ged_3hop'], width, label='3-hop', alpha=0.8)
        
        ax.set_xlabel('Query Index')
        ax.set_ylabel('GED Value')
        ax.set_title('Per-Query GED Evolution (3-hop Relaxed)')
        ax.set_xticks(x)
        ax.set_xticklabels([f"Q{i+1}" for i in range(len(relaxed_data))])
        ax.legend()
    
    # 9. Summary Statistics Table
    ax = axes[2, 2]
    ax.axis('off')
    
    # Calculate key statistics
    stats_text = "GED SHORTCUT ANALYSIS SUMMARY\n" + "=" * 30 + "\n\n"
    
    for config in df['config'].unique():
        config_data = df[df['config'] == config]
        stats_text += f"{config}:\n"
        stats_text += f"  Avg GED: {config_data['final_ged'].mean():.3f}\n"
        stats_text += f"  Avg IG: {config_data['final_ig'].mean():.3f}\n"
        stats_text += f"  Avg geDIG: {config_data['final_gedig'].mean():.3f}\n"
        
        if '3-hop' in config:
            shortcut_rate = config_data['shortcut_detected'].mean() * 100
            stats_text += f"  Shortcuts: {shortcut_rate:.1f}%\n"
        stats_text += "\n"
    
    ax.text(0.1, 0.5, stats_text, fontsize=9, 
           verticalalignment='center', fontfamily='monospace')
    
    plt.suptitle('GED Shortcut Effect Analysis', fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("../results/ged_shortcut")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    viz_path = results_dir / f"ged_shortcut_analysis_{timestamp}.png"
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Saved visualization to: {viz_path}")
    return viz_path


def save_detailed_results(results):
    """Save detailed GED/IG tracking results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("../results/ged_shortcut")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean for JSON
    clean_results = []
    for r in results:
        clean_r = {}
        for k, v in r.items():
            if isinstance(v, (np.integer, np.floating, np.bool_)):
                clean_r[k] = float(v) if not isinstance(v, np.bool_) else bool(v)
            else:
                clean_r[k] = v
        clean_results.append(clean_r)
    
    # Save JSON
    json_path = results_dir / f"ged_detailed_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(clean_results, f, indent=2)
    print(f"✅ Saved detailed results to: {json_path}")
    
    # Save CSV for analysis
    df = pd.DataFrame(clean_results)
    csv_path = results_dir / f"ged_detailed_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved CSV to: {csv_path}")


def main():
    """Run GED shortcut effect analysis."""
    print("\n🚀 Starting GED Shortcut Effect Analysis")
    print("=" * 60)
    
    try:
        # Run analysis
        results = run_ged_shortcut_analysis()
        
        # Analyze patterns
        analyze_shortcut_patterns(results)
        
        # Create visualizations
        create_ged_visualization(results)
        
        # Save results
        save_detailed_results(results)
        
        print("\n" + "=" * 60)
        print("✅ GED SHORTCUT ANALYSIS COMPLETE")
        print("=" * 60)
        
        # Key findings
        df = pd.DataFrame(results)
        print("\n💡 KEY FINDINGS:")
        print("-" * 40)
        
        # Check for GED drops
        three_hop = df[df['config'].str.contains('3-hop')]
        if not three_hop.empty:
            avg_drop = three_hop['delta_ged_2to3'].mean()
            if avg_drop < 0:
                print(f"1. GED DROP detected: Average {avg_drop:.3f} from 2→3 hop")
            
            shortcut_rate = three_hop['shortcut_detected'].mean() * 100
            if shortcut_rate > 0:
                print(f"2. Shortcuts found in {shortcut_rate:.1f}% of 3-hop queries")
            
            plateau_cases = sum((abs(three_hop['delta_ged_2to3']) < 0.01) & 
                              (three_hop['edges_3hop'] > 0))
            if plateau_cases > 0:
                print(f"3. GED plateau effect in {plateau_cases} cases")
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())