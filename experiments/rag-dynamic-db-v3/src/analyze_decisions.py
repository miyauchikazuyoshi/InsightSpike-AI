#!/usr/bin/env python3
"""Analyze which queries were accepted/rejected and why."""

import json
import csv
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

from run_optimal_gedig import OptimalGeDIGSystem, run_optimal_experiment
from run_experiment_improved import (
    create_high_quality_knowledge_base,
    create_meaningful_queries,
    ExperimentConfig
)


def analyze_decisions_detailed():
    """Analyze accept/reject decisions in detail."""
    print("🔍 Analyzing geDIG Decisions")
    print("=" * 60)
    
    # Setup
    config = ExperimentConfig()
    knowledge_base = create_high_quality_knowledge_base()
    test_queries = create_meaningful_queries()
    
    # Run optimal configuration (36.8% acceptance)
    params = {
        'k': 0.18,
        'node_weight': 0.5,
        'edge_weight': 0.15,
        'novelty_weight': 0.45,
        'connectivity_weight': 0.08,
        'base_threshold': 0.42,
        'target_rate': 0.30,
        'adaptive_k': False
    }
    
    system = OptimalGeDIGSystem("gedig", config, params)
    system.add_initial_knowledge(knowledge_base)
    
    # Collect detailed decision data
    decision_data = []
    
    for i, (query, depth) in enumerate(test_queries):
        # Process query
        result = system.process_query(query, depth)
        
        # Get metadata
        metadata = system.last_gedig_metadata if hasattr(system, 'last_gedig_metadata') else {}
        
        # Determine query category
        if i < 5:
            category = "Direct"
        elif i < 10:
            category = "Synthesis"
        elif i < 15:
            category = "Extension"
        else:
            category = "Novel"
        
        # Determine decision reason
        accepted = result.get('updated', False)
        if accepted:
            if metadata.get('novelty', 0) > 0.8:
                reason = "High novelty"
            elif metadata.get('gedig_score', 0) > metadata.get('threshold_used', 0) + 0.1:
                reason = "Strong geDIG score"
            elif i < 3:
                reason = "Bootstrap phase"
            else:
                reason = "Moderate value"
        else:
            if metadata.get('novelty', 0) < 0.5:
                reason = "Low novelty"
            elif metadata.get('gedig_score', 0) < metadata.get('threshold_used', 0) - 0.1:
                reason = "Weak geDIG score"
            else:
                reason = "Below threshold"
        
        decision_data.append({
            'query_id': i + 1,
            'query': query,
            'category': category,
            'depth': depth,
            'decision': 'ACCEPTED' if accepted else 'REJECTED',
            'gedig_score': round(metadata.get('gedig_score', 0), 3),
            'threshold': round(metadata.get('threshold_used', 0), 3),
            'novelty': round(metadata.get('novelty', 0), 3),
            'similarity': round(metadata.get('max_similarity', 0), 3),
            'edges_added': metadata.get('edges_added', 0),
            'reason': reason,
            'current_nodes': result.get('graph_size', 0),
            'current_edges': result.get('graph_edges', 0)
        })
    
    return decision_data


def evaluate_decision_quality(decision_data):
    """Evaluate if decisions were good based on various criteria."""
    print("\n📊 Evaluating Decision Quality")
    print("-" * 60)
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(decision_data)
    
    # 1. Acceptance rate by category
    print("\n1. Acceptance Rate by Query Category:")
    for category in ['Direct', 'Synthesis', 'Extension', 'Novel']:
        cat_data = df[df['category'] == category]
        accepted = len(cat_data[cat_data['decision'] == 'ACCEPTED'])
        total = len(cat_data)
        rate = accepted / total * 100 if total > 0 else 0
        print(f"   {category:12s}: {accepted}/{total} ({rate:.0f}%)")
    
    # 2. Quality metrics for accepted queries
    accepted_df = df[df['decision'] == 'ACCEPTED']
    rejected_df = df[df['decision'] == 'REJECTED']
    
    print("\n2. Quality Metrics:")
    print(f"   Accepted queries (n={len(accepted_df)}):")
    print(f"      Avg novelty: {accepted_df['novelty'].mean():.3f}")
    print(f"      Avg similarity: {accepted_df['similarity'].mean():.3f}")
    print(f"      Avg geDIG score: {accepted_df['gedig_score'].mean():.3f}")
    print(f"      Avg edges added: {accepted_df['edges_added'].mean():.1f}")
    
    print(f"   Rejected queries (n={len(rejected_df)}):")
    print(f"      Avg novelty: {rejected_df['novelty'].mean():.3f}")
    print(f"      Avg similarity: {rejected_df['similarity'].mean():.3f}")
    print(f"      Avg geDIG score: {rejected_df['gedig_score'].mean():.3f}")
    
    # 3. Decision quality assessment
    print("\n3. Decision Quality Assessment:")
    
    # Good accepts: High novelty (>0.7) or low similarity (<0.1)
    good_accepts = accepted_df[(accepted_df['novelty'] > 0.7) | (accepted_df['similarity'] < 0.1)]
    print(f"   Good accepts: {len(good_accepts)}/{len(accepted_df)} ({len(good_accepts)/len(accepted_df)*100:.0f}%)")
    
    # Good rejects: Low novelty (<0.6) and high similarity (>0.15)
    good_rejects = rejected_df[(rejected_df['novelty'] < 0.6) & (rejected_df['similarity'] > 0.15)]
    print(f"   Good rejects: {len(good_rejects)}/{len(rejected_df)} ({len(good_rejects)/max(1,len(rejected_df))*100:.0f}%)")
    
    # Overall quality score
    total_good = len(good_accepts) + len(good_rejects)
    total_decisions = len(df)
    quality_score = total_good / total_decisions * 100
    print(f"\n   📈 Overall Decision Quality: {quality_score:.1f}%")
    
    # 4. Identify questionable decisions
    print("\n4. Questionable Decisions:")
    
    # Accepted but low novelty and high similarity
    questionable_accepts = accepted_df[(accepted_df['novelty'] < 0.5) & (accepted_df['similarity'] > 0.2)]
    if not questionable_accepts.empty:
        print("   Questionable accepts (low novelty, high similarity):")
        for _, row in questionable_accepts.iterrows():
            print(f"      Query {row['query_id']}: {row['query'][:50]}...")
    
    # Rejected but high novelty and low similarity  
    questionable_rejects = rejected_df[(rejected_df['novelty'] > 0.8) & (rejected_df['similarity'] < 0.1)]
    if not questionable_rejects.empty:
        print("   Questionable rejects (high novelty, low similarity):")
        for _, row in questionable_rejects.iterrows():
            print(f"      Query {row['query_id']}: {row['query'][:50]}...")
    
    if questionable_accepts.empty and questionable_rejects.empty:
        print("   ✅ No questionable decisions found!")
    
    return df


def save_decision_analysis(decision_data, output_dir=None):
    """Save decision analysis to CSV and JSON."""
    if output_dir is None:
        output_dir = Path("../results/decision_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save to CSV
    csv_path = output_dir / f"decisions_{timestamp}.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        if decision_data:
            writer = csv.DictWriter(f, fieldnames=decision_data[0].keys())
            writer.writeheader()
            writer.writerows(decision_data)
    
    print(f"\n✅ Saved decision analysis to: {csv_path}")
    
    # Save summary JSON
    df = pd.DataFrame(decision_data)
    
    summary = {
        'timestamp': timestamp,
        'total_queries': len(df),
        'accepted': len(df[df['decision'] == 'ACCEPTED']),
        'rejected': len(df[df['decision'] == 'REJECTED']),
        'acceptance_rate': len(df[df['decision'] == 'ACCEPTED']) / len(df) * 100,
        'by_category': {},
        'quality_metrics': {}
    }
    
    # Acceptance by category
    for category in ['Direct', 'Synthesis', 'Extension', 'Novel']:
        cat_data = df[df['category'] == category]
        accepted = len(cat_data[cat_data['decision'] == 'ACCEPTED'])
        total = len(cat_data)
        summary['by_category'][category] = {
            'accepted': accepted,
            'total': total,
            'rate': accepted / total * 100 if total > 0 else 0
        }
    
    # Quality metrics
    accepted_df = df[df['decision'] == 'ACCEPTED']
    rejected_df = df[df['decision'] == 'REJECTED']
    
    summary['quality_metrics'] = {
        'accepted': {
            'avg_novelty': float(accepted_df['novelty'].mean()),
            'avg_similarity': float(accepted_df['similarity'].mean()),
            'avg_gedig_score': float(accepted_df['gedig_score'].mean()),
            'avg_edges_added': float(accepted_df['edges_added'].mean())
        },
        'rejected': {
            'avg_novelty': float(rejected_df['novelty'].mean()),
            'avg_similarity': float(rejected_df['similarity'].mean()),
            'avg_gedig_score': float(rejected_df['gedig_score'].mean())
        }
    }
    
    json_path = output_dir / f"summary_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved summary to: {json_path}")
    
    return csv_path, json_path


def create_human_readable_report(decision_data):
    """Create a human-readable report of decisions."""
    print("\n" + "="*60)
    print("📝 HUMAN-READABLE DECISION REPORT")
    print("="*60)
    
    df = pd.DataFrame(decision_data)
    
    print("\n🟢 ACCEPTED QUERIES (Knowledge Added):")
    print("-" * 60)
    
    accepted = df[df['decision'] == 'ACCEPTED']
    for i, (_, row) in enumerate(accepted.iterrows(), 1):
        print(f"\n{i}. Query #{row['query_id']}: {row['query'][:60]}...")
        print(f"   Category: {row['category']} | Novelty: {row['novelty']:.2f} | Similarity: {row['similarity']:.2f}")
        print(f"   geDIG Score: {row['gedig_score']:.3f} > Threshold: {row['threshold']:.3f}")
        print(f"   Reason: {row['reason']}")
        print(f"   Impact: Added {row['edges_added']} edges")
    
    print("\n\n🔴 REJECTED QUERIES (Knowledge Skipped):")
    print("-" * 60)
    
    rejected = df[df['decision'] == 'REJECTED']
    for i, (_, row) in enumerate(rejected.iterrows(), 1):
        print(f"\n{i}. Query #{row['query_id']}: {row['query'][:60]}...")
        print(f"   Category: {row['category']} | Novelty: {row['novelty']:.2f} | Similarity: {row['similarity']:.2f}")
        print(f"   geDIG Score: {row['gedig_score']:.3f} < Threshold: {row['threshold']:.3f}")
        print(f"   Reason: {row['reason']}")
    
    print("\n" + "="*60)
    print("📊 SUMMARY STATISTICS")
    print("="*60)
    
    print(f"\nAcceptance Rate: {len(accepted)}/{len(df)} ({len(accepted)/len(df)*100:.1f}%)")
    print(f"Average Novelty - Accepted: {accepted['novelty'].mean():.3f}, Rejected: {rejected['novelty'].mean():.3f}")
    print(f"Average Similarity - Accepted: {accepted['similarity'].mean():.3f}, Rejected: {rejected['similarity'].mean():.3f}")
    
    # Determine if decisions were good
    good_decision_rate = evaluate_goodness(df)
    
    if good_decision_rate > 80:
        print(f"\n✅ EXCELLENT: {good_decision_rate:.0f}% of decisions were appropriate!")
    elif good_decision_rate > 60:
        print(f"\n⚠️ GOOD: {good_decision_rate:.0f}% of decisions were appropriate.")
    else:
        print(f"\n❌ NEEDS IMPROVEMENT: Only {good_decision_rate:.0f}% of decisions were appropriate.")


def evaluate_goodness(df):
    """Evaluate what percentage of decisions were good."""
    good_decisions = 0
    total_decisions = len(df)
    
    for _, row in df.iterrows():
        if row['decision'] == 'ACCEPTED':
            # Good accept: novelty > 0.6 or similarity < 0.12
            if row['novelty'] > 0.6 or row['similarity'] < 0.12:
                good_decisions += 1
        else:  # REJECTED
            # Good reject: novelty < 0.7 and similarity > 0.1
            if row['novelty'] < 0.7 and row['similarity'] > 0.1:
                good_decisions += 1
    
    return good_decisions / total_decisions * 100


def main():
    """Main execution."""
    try:
        # Analyze decisions
        decision_data = analyze_decisions_detailed()
        
        # Evaluate quality
        df = evaluate_decision_quality(decision_data)
        
        # Save to files
        csv_path, json_path = save_decision_analysis(decision_data)
        
        # Create human-readable report
        create_human_readable_report(decision_data)
        
        print("\n" + "="*60)
        print("✅ ANALYSIS COMPLETE")
        print("="*60)
        print(f"\nFiles saved:")
        print(f"  • CSV: {csv_path}")
        print(f"  • JSON: {json_path}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)