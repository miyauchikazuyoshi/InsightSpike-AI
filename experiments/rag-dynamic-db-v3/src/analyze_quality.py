#!/usr/bin/env python3
"""Qualitative analysis of geDIG decisions - focusing on actual usefulness."""

import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

from run_optimal_gedig import OptimalGeDIGSystem
from run_experiment_improved import (
    create_high_quality_knowledge_base,
    create_meaningful_queries,
    ExperimentConfig,
    ImprovedEmbedder
)


def analyze_query_usefulness():
    """Analyze if truly useful queries were accepted and simple ones rejected."""
    print("🎯 Qualitative Analysis of geDIG Decisions")
    print("=" * 60)
    print("Focus: Are we accepting truly novel/useful queries?")
    print("       Are we rejecting queries answerable with existing knowledge?")
    print()
    
    # Get initial knowledge base
    knowledge_base = create_high_quality_knowledge_base()
    test_queries = create_meaningful_queries()
    
    # Categorize queries by their ACTUAL usefulness
    query_analysis = []
    
    # Define what existing knowledge can answer
    existing_knowledge_coverage = {
        'python_gil': ['GIL', 'threading', 'parallelism'],
        'memory_management': ['reference counting', 'garbage collection'],
        'overfitting': ['regularization', 'L1/L2'],
        'gradient_descent': ['optimization', 'loss function', 'learning rate'],
        'cnn': ['convolution', 'spatial features', 'pooling'],
        'transformers': ['self-attention', 'parallel processing', 'RNN comparison'],
        'backpropagation': ['automatic differentiation', 'gradients'],
        'bert': ['bidirectional', 'masked language modeling'],
        'tokenization': ['BPE', 'WordPiece', 'subword'],
        'gpu': ['CUDA', 'parallel matrix operations'],
        'mixed_precision': ['FP16', 'FP32', 'memory optimization'],
        'feature_engineering': ['encoding', 'scaling', 'polynomial features'],
        'normalization': ['StandardScaler', 'MinMaxScaler', 'gradient stability']
    }
    
    # Analyze each query
    for i, (query, depth) in enumerate(test_queries):
        query_id = i + 1
        
        # Determine if query is answerable with existing knowledge
        answerable_with_existing = False
        relevant_knowledge = []
        
        query_lower = query.lower()
        
        # Check if query is about something we already know
        if 'gil' in query_lower and ('affect' in query_lower or 'threading' in query_lower):
            answerable_with_existing = True
            relevant_knowledge.append('python_gil')
            
        elif 'overfitting' in query_lower and 'causes' in query_lower:
            answerable_with_existing = True
            relevant_knowledge.append('overfitting')
            
        elif 'transformer' in query_lower and 'rnn' in query_lower and 'compare' in query_lower:
            answerable_with_existing = True
            relevant_knowledge.append('transformers')
            
        elif 'gradient descent' in query_lower and 'backpropagation' in query_lower:
            answerable_with_existing = True
            relevant_knowledge.extend(['gradient_descent', 'backpropagation'])
            
        elif 'overcome gil' in query_lower:
            # This extends existing knowledge but is partially answerable
            answerable_with_existing = 'partial'
            relevant_knowledge.append('python_gil')
            
        # Determine actual novelty/usefulness
        if query_id <= 5:  # Direct questions
            if answerable_with_existing == True:
                actual_usefulness = 'LOW'  # Can be answered with existing
                usefulness_reason = 'Directly answerable with existing knowledge'
            elif answerable_with_existing == 'partial':
                actual_usefulness = 'MEDIUM'
                usefulness_reason = 'Extends existing knowledge'
            else:
                actual_usefulness = 'HIGH'
                usefulness_reason = 'Genuinely new information needed'
                
        elif query_id <= 10:  # Synthesis questions
            # These require combining knowledge
            if 'advanced regularization' in query_lower:
                actual_usefulness = 'HIGH'
                usefulness_reason = 'Goes beyond basic L1/L2 knowledge'
            elif 'attention mechanism' in query_lower:
                actual_usefulness = 'MEDIUM'
                usefulness_reason = 'Deepens transformer understanding'
            elif 'transfer learning' in query_lower:
                actual_usefulness = 'HIGH'
                usefulness_reason = 'New concept not in knowledge base'
            elif 'vanishing gradient' in query_lower:
                actual_usefulness = 'HIGH'
                usefulness_reason = 'Important concept not covered'
            elif 'gans' in query_lower:
                actual_usefulness = 'HIGH'
                usefulness_reason = 'Completely new architecture type'
            else:
                actual_usefulness = 'MEDIUM'
                usefulness_reason = 'Synthesis of existing concepts'
                
        elif query_id <= 15:  # Extension questions
            if 'lstm vs gru vs transformer' in query_lower:
                actual_usefulness = 'HIGH'
                usefulness_reason = 'Practical comparison not in KB'
            elif 'automatic differentiation' in query_lower and 'pytorch' in query_lower:
                actual_usefulness = 'HIGH'
                usefulness_reason = 'Implementation details not covered'
            elif 'batch normalization' in query_lower:
                actual_usefulness = 'HIGH'
                usefulness_reason = 'Important technique not in KB'
            elif 'deploying ml models' in query_lower:
                actual_usefulness = 'HIGH'
                usefulness_reason = 'Production aspects not covered'
            else:
                actual_usefulness = 'MEDIUM'
                usefulness_reason = 'Extends existing knowledge'
                
        else:  # Novel questions
            if 'imbalanced datasets' in query_lower:
                actual_usefulness = 'HIGH'
                usefulness_reason = 'Practical problem not addressed'
            elif 'memory management internals' in query_lower:
                actual_usefulness = 'MEDIUM'
                usefulness_reason = 'Deepens existing Python memory knowledge'
            elif 'optimization techniques' in query_lower:
                actual_usefulness = 'MEDIUM'
                usefulness_reason = 'Extends gradient descent knowledge'
            elif 'beyond bert' in query_lower:
                actual_usefulness = 'HIGH'
                usefulness_reason = 'Cutting-edge developments'
            else:
                actual_usefulness = 'MEDIUM'
                usefulness_reason = 'Novel but not critical'
        
        query_analysis.append({
            'query_id': query_id,
            'query': query[:80] + '...' if len(query) > 80 else query,
            'category': get_category(i),
            'answerable_with_existing': answerable_with_existing,
            'relevant_existing_knowledge': ', '.join(relevant_knowledge) if relevant_knowledge else 'None',
            'actual_usefulness': actual_usefulness,
            'usefulness_reason': usefulness_reason
        })
    
    return query_analysis


def get_category(index):
    """Get query category from index."""
    if index < 5:
        return 'Direct'
    elif index < 10:
        return 'Synthesis'
    elif index < 15:
        return 'Extension'
    else:
        return 'Novel'


def evaluate_decision_quality(query_analysis):
    """Evaluate if geDIG made good decisions based on actual usefulness."""
    print("\n📊 Evaluating Decision Quality Based on Usefulness")
    print("-" * 60)
    
    # Run the actual system to get decisions
    config = ExperimentConfig()
    knowledge_base = create_high_quality_knowledge_base()
    test_queries = create_meaningful_queries()
    
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
    
    decisions = []
    for query, depth in test_queries:
        result = system.process_query(query, depth)
        decisions.append(result.get('updated', False))
    
    # Combine with usefulness analysis
    for i, analysis in enumerate(query_analysis):
        analysis['decision'] = 'ACCEPTED' if decisions[i] else 'REJECTED'
        
        # Evaluate if decision was correct
        if analysis['actual_usefulness'] == 'HIGH':
            # Should definitely accept
            analysis['correct_decision'] = analysis['decision'] == 'ACCEPTED'
            analysis['decision_quality'] = 'GOOD' if analysis['correct_decision'] else 'BAD (missed valuable)'
        elif analysis['actual_usefulness'] == 'LOW':
            # Should definitely reject
            analysis['correct_decision'] = analysis['decision'] == 'REJECTED'
            analysis['decision_quality'] = 'GOOD' if analysis['correct_decision'] else 'BAD (accepted redundant)'
        else:  # MEDIUM
            # Either is acceptable
            analysis['correct_decision'] = True
            analysis['decision_quality'] = 'ACCEPTABLE'
    
    return query_analysis, decisions


def print_qualitative_report(query_analysis):
    """Print detailed qualitative analysis report."""
    print("\n" + "=" * 60)
    print("📝 QUALITATIVE DECISION ANALYSIS REPORT")
    print("=" * 60)
    
    df = pd.DataFrame(query_analysis)
    
    # Group by usefulness
    print("\n🎯 HIGH USEFULNESS QUERIES (Should Accept):")
    print("-" * 60)
    high_useful = df[df['actual_usefulness'] == 'HIGH']
    
    for _, row in high_useful.iterrows():
        symbol = "✅" if row['decision'] == 'ACCEPTED' else "❌"
        print(f"\n{symbol} Query {row['query_id']}: {row['query']}")
        print(f"   Decision: {row['decision']}")
        print(f"   Reason for usefulness: {row['usefulness_reason']}")
        if row['decision'] == 'REJECTED':
            print(f"   ⚠️ MISSED OPPORTUNITY - This was valuable knowledge!")
    
    print("\n\n📉 LOW USEFULNESS QUERIES (Should Reject):")
    print("-" * 60)
    low_useful = df[df['actual_usefulness'] == 'LOW']
    
    for _, row in low_useful.iterrows():
        symbol = "✅" if row['decision'] == 'REJECTED' else "❌"
        print(f"\n{symbol} Query {row['query_id']}: {row['query']}")
        print(f"   Decision: {row['decision']}")
        print(f"   Can be answered with: {row['relevant_existing_knowledge']}")
        if row['decision'] == 'ACCEPTED':
            print(f"   ⚠️ REDUNDANT - Already covered in knowledge base!")
    
    print("\n\n📊 MEDIUM USEFULNESS QUERIES (Either OK):")
    print("-" * 60)
    medium_useful = df[df['actual_usefulness'] == 'MEDIUM']
    
    for _, row in medium_useful.iterrows():
        symbol = "✓"
        print(f"\n{symbol} Query {row['query_id']}: {row['query']}")
        print(f"   Decision: {row['decision']}")
        print(f"   Reason: {row['usefulness_reason']}")
    
    # Calculate quality metrics
    print("\n\n" + "=" * 60)
    print("📈 QUALITY METRICS")
    print("=" * 60)
    
    # High usefulness acceptance rate
    high_accepted = len(high_useful[high_useful['decision'] == 'ACCEPTED'])
    high_total = len(high_useful)
    high_rate = high_accepted / high_total * 100 if high_total > 0 else 0
    
    # Low usefulness rejection rate  
    low_rejected = len(low_useful[low_useful['decision'] == 'REJECTED'])
    low_total = len(low_useful)
    low_rate = low_rejected / low_total * 100 if low_total > 0 else 0
    
    # Overall correct decisions
    correct = len(df[df['correct_decision'] == True])
    total = len(df)
    correct_rate = correct / total * 100
    
    print(f"\n✨ High-Value Acceptance Rate: {high_accepted}/{high_total} ({high_rate:.0f}%)")
    print(f"🛡️ Low-Value Rejection Rate: {low_rejected}/{low_total} ({low_rate:.0f}%)")
    print(f"📊 Overall Correct Decisions: {correct}/{total} ({correct_rate:.0f}%)")
    
    # Final verdict
    print("\n" + "=" * 60)
    print("🏆 FINAL VERDICT")
    print("=" * 60)
    
    if high_rate >= 70 and low_rate >= 70:
        print("✅ EXCELLENT: System effectively distinguishes valuable from redundant knowledge!")
    elif high_rate >= 50 and low_rate >= 50:
        print("⚠️ GOOD: System shows reasonable discrimination but could be improved.")
    else:
        print("❌ NEEDS IMPROVEMENT: System struggles to identify truly valuable knowledge.")
    
    # Specific issues
    missed_valuable = high_useful[high_useful['decision'] == 'REJECTED']
    accepted_redundant = low_useful[low_useful['decision'] == 'ACCEPTED']
    
    if not missed_valuable.empty:
        print(f"\n⚠️ Missed {len(missed_valuable)} valuable queries:")
        for _, row in missed_valuable.iterrows():
            print(f"   - Query {row['query_id']}: {row['usefulness_reason']}")
    
    if not accepted_redundant.empty:
        print(f"\n⚠️ Accepted {len(accepted_redundant)} redundant queries:")
        for _, row in accepted_redundant.iterrows():
            print(f"   - Query {row['query_id']}: {row['relevant_existing_knowledge']}")
    
    return {
        'high_value_acceptance': high_rate,
        'low_value_rejection': low_rate,
        'overall_quality': correct_rate,
        'missed_valuable': len(missed_valuable),
        'accepted_redundant': len(accepted_redundant)
    }


def save_qualitative_analysis(query_analysis, metrics):
    """Save qualitative analysis to files."""
    output_dir = Path("../results/qualitative_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed analysis
    csv_path = output_dir / f"qualitative_analysis_{timestamp}.csv"
    df = pd.DataFrame(query_analysis)
    df.to_csv(csv_path, index=False)
    
    # Save summary
    summary = {
        'timestamp': timestamp,
        'metrics': metrics,
        'usefulness_distribution': {
            'high': len(df[df['actual_usefulness'] == 'HIGH']),
            'medium': len(df[df['actual_usefulness'] == 'MEDIUM']),
            'low': len(df[df['actual_usefulness'] == 'LOW'])
        },
        'decision_distribution': {
            'accepted': len(df[df['decision'] == 'ACCEPTED']),
            'rejected': len(df[df['decision'] == 'REJECTED'])
        }
    }
    
    json_path = output_dir / f"qualitative_summary_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Saved analysis to: {csv_path}")
    print(f"✅ Saved summary to: {json_path}")
    
    return csv_path, json_path


def main():
    """Main execution."""
    try:
        # Analyze query usefulness
        query_analysis = analyze_query_usefulness()
        
        # Get actual decisions and evaluate
        query_analysis, decisions = evaluate_decision_quality(query_analysis)
        
        # Print detailed report
        metrics = print_qualitative_report(query_analysis)
        
        # Save results
        save_qualitative_analysis(query_analysis, metrics)
        
        print("\n" + "=" * 60)
        print("✅ QUALITATIVE ANALYSIS COMPLETE")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)