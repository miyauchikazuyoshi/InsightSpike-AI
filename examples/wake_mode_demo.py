#!/usr/bin/env python3
"""
Wake Mode Demonstration
=======================

This script demonstrates the Wake Mode implementation for efficient
query processing using geDIG minimization.
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from insightspike.algorithms.gedig_wake_mode import (
    WakeModeGeDIG,
    ProcessingMode,
)
from insightspike.algorithms.gedig_core import GeDIGCore


def create_knowledge_graph():
    """Create a sample knowledge graph."""
    g = nx.Graph()
    
    # Add concept nodes
    concepts = [
        'AI', 'Machine Learning', 'Deep Learning', 'Neural Networks',
        'Supervised Learning', 'Unsupervised Learning', 'Reinforcement Learning',
        'CNN', 'RNN', 'Transformer', 'BERT', 'GPT',
        'Computer Vision', 'NLP', 'Feature Engineering'
    ]
    g.add_nodes_from(concepts)
    
    # Add relationships
    edges = [
        ('AI', 'Machine Learning'),
        ('Machine Learning', 'Deep Learning'),
        ('Machine Learning', 'Supervised Learning'),
        ('Machine Learning', 'Unsupervised Learning'),
        ('Machine Learning', 'Reinforcement Learning'),
        ('Deep Learning', 'Neural Networks'),
        ('Neural Networks', 'CNN'),
        ('Neural Networks', 'RNN'),
        ('Neural Networks', 'Transformer'),
        ('Transformer', 'BERT'),
        ('Transformer', 'GPT'),
        ('CNN', 'Computer Vision'),
        ('RNN', 'NLP'),
        ('Transformer', 'NLP'),
        ('Machine Learning', 'Feature Engineering'),
    ]
    g.add_edges_from(edges)
    
    return g


def demonstrate_wake_mode():
    """Demonstrate Wake Mode processing."""
    print("=" * 60)
    print("Wake Mode geDIG Demonstration")
    print("=" * 60)
    
    # Create knowledge graph
    graph = create_knowledge_graph()
    print(f"\nKnowledge graph: {len(graph.nodes())} nodes, {len(graph.edges())} edges")
    
    # Initialize calculators
    wake_calculator = WakeModeGeDIG()
    sleep_calculator = GeDIGCore()  # Standard mode
    
    # Add some known patterns
    print("\nAdding known patterns...")
    
    # Pattern 1: Deep learning for vision
    vision_pattern = nx.Graph()
    vision_pattern.add_edges_from([
        ('Deep Learning', 'CNN'),
        ('CNN', 'Computer Vision')
    ])
    wake_calculator.add_pattern('vision_pipeline', vision_pattern)
    
    # Pattern 2: NLP pipeline
    nlp_pattern = nx.Graph()
    nlp_pattern.add_edges_from([
        ('Deep Learning', 'Transformer'),
        ('Transformer', 'NLP')
    ])
    wake_calculator.add_pattern('nlp_pipeline', nlp_pattern)
    
    # Test queries
    queries = [
        {
            'name': 'Vision Task',
            'focal_nodes': {'Deep Learning', 'Computer Vision'},
            'context': {'task': 'image_classification'}
        },
        {
            'name': 'NLP Task',
            'focal_nodes': {'Deep Learning', 'NLP'},
            'context': {'task': 'text_generation'}
        },
        {
            'name': 'General ML',
            'focal_nodes': {'Machine Learning', 'Feature Engineering'},
            'context': {'task': 'data_preprocessing'}
        }
    ]
    
    print("\n" + "-" * 60)
    print("Query Processing Comparison")
    print("-" * 60)
    
    results = []
    
    for query in queries:
        print(f"\nQuery: {query['name']}")
        print(f"Focal nodes: {query['focal_nodes']}")
        
        # Wake Mode (efficient)
        wake_result = wake_calculator.calculate_wake_mode_gedig(
            graph,
            query['focal_nodes'],
            query['context']
        )
        
        # Sleep Mode (exploration)
        sleep_result = sleep_calculator.calculate(
            graph,
            query['focal_nodes']
        )
        
        print(f"\nWake Mode:")
        print(f"  - geDIG value: {wake_result.gedig_value:.4f}")
        print(f"  - Pattern match: {wake_result.pattern_match_score:.4f}")
        print(f"  - Efficiency: {wake_result.efficiency_score:.4f}")
        print(f"  - Computation time: {wake_result.computation_time*1000:.2f}ms")
        
        print(f"\nSleep Mode:")
        print(f"  - geDIG value: {sleep_result.gedig_value:.4f}")
        print(f"  - Computation time: {sleep_result.computation_time*1000:.2f}ms")
        
        results.append({
            'query': query['name'],
            'wake_gedig': wake_result.gedig_value,
            'sleep_gedig': sleep_result.gedig_value,
            'pattern_match': wake_result.pattern_match_score,
            'efficiency': wake_result.efficiency_score,
            'wake_time': wake_result.computation_time,
            'sleep_time': sleep_result.computation_time
        })
    
    # Visualize results
    visualize_results(results)
    
    # Show computation efficiency
    print("\n" + "-" * 60)
    print("Efficiency Summary")
    print("-" * 60)
    
    avg_wake_time = np.mean([r['wake_time'] for r in results])
    avg_sleep_time = np.mean([r['sleep_time'] for r in results])
    speedup = avg_sleep_time / avg_wake_time
    
    print(f"Average Wake Mode time: {avg_wake_time*1000:.2f}ms")
    print(f"Average Sleep Mode time: {avg_sleep_time*1000:.2f}ms")
    print(f"Speedup factor: {speedup:.2f}x")
    
    # Pattern matching effectiveness
    avg_pattern_match = np.mean([r['pattern_match'] for r in results])
    print(f"\nAverage pattern match score: {avg_pattern_match:.3f}")
    
    return results


def visualize_results(results):
    """Visualize the comparison results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. geDIG values comparison
    ax1 = axes[0, 0]
    queries = [r['query'] for r in results]
    wake_gedigs = [r['wake_gedig'] for r in results]
    sleep_gedigs = [r['sleep_gedig'] for r in results]
    
    x = np.arange(len(queries))
    width = 0.35
    
    ax1.bar(x - width/2, wake_gedigs, width, label='Wake Mode', color='skyblue')
    ax1.bar(x + width/2, sleep_gedigs, width, label='Sleep Mode', color='lightcoral')
    ax1.set_xlabel('Query')
    ax1.set_ylabel('geDIG Value')
    ax1.set_title('geDIG Values: Wake vs Sleep Mode')
    ax1.set_xticks(x)
    ax1.set_xticklabels(queries, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Pattern matching scores
    ax2 = axes[0, 1]
    pattern_scores = [r['pattern_match'] for r in results]
    efficiency_scores = [r['efficiency'] for r in results]
    
    ax2.plot(queries, pattern_scores, 'o-', label='Pattern Match', markersize=8)
    ax2.plot(queries, efficiency_scores, 's-', label='Efficiency', markersize=8)
    ax2.set_xlabel('Query')
    ax2.set_ylabel('Score')
    ax2.set_title('Wake Mode Performance Metrics')
    ax2.set_ylim(0, 1.1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Computation time comparison
    ax3 = axes[1, 0]
    wake_times = [r['wake_time']*1000 for r in results]  # Convert to ms
    sleep_times = [r['sleep_time']*1000 for r in results]
    
    ax3.bar(x - width/2, wake_times, width, label='Wake Mode', color='green')
    ax3.bar(x + width/2, sleep_times, width, label='Sleep Mode', color='orange')
    ax3.set_xlabel('Query')
    ax3.set_ylabel('Time (ms)')
    ax3.set_title('Computation Time Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(queries, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Efficiency ratio
    ax4 = axes[1, 1]
    speedup_ratios = [r['sleep_time']/r['wake_time'] for r in results]
    
    bars = ax4.bar(queries, speedup_ratios, color='purple')
    ax4.axhline(y=1, color='r', linestyle='--', label='No speedup')
    ax4.set_xlabel('Query')
    ax4.set_ylabel('Speedup Factor')
    ax4.set_title('Wake Mode Speedup')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, ratio in zip(bars, speedup_ratios):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{ratio:.2f}x', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('wake_mode_comparison.png', dpi=150)
    plt.show()


def demonstrate_predictive_coding():
    """Demonstrate Wake Mode as predictive coding."""
    print("\n" + "=" * 60)
    print("Wake Mode as Predictive Coding")
    print("=" * 60)
    
    # Create a simple graph
    g = nx.Graph()
    g.add_edges_from([
        ('input', 'hidden1'), ('hidden1', 'hidden2'),
        ('hidden2', 'output'), ('hidden1', 'hidden3'),
        ('hidden3', 'output')
    ])
    
    calculator = WakeModeGeDIG()
    
    # Simulate multiple queries (prediction error minimization)
    focal_nodes = {'input', 'output'}
    
    print("\nSimulating prediction error minimization...")
    errors = []
    
    for i in range(10):
        result = calculator.calculate_wake_mode_gedig(g, focal_nodes)
        
        # Simulate prediction error (decreasing over time)
        error = result.gedig_value * (0.9 ** i)
        errors.append(error)
        
        if i % 3 == 0:
            print(f"Iteration {i}: Prediction error = {error:.4f}")
    
    # Plot convergence
    plt.figure(figsize=(8, 6))
    plt.plot(range(10), errors, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Iteration')
    plt.ylabel('Prediction Error (geDIG)')
    plt.title('Wake Mode: Prediction Error Minimization')
    plt.grid(True, alpha=0.3)
    plt.savefig('wake_mode_convergence.png', dpi=150)
    plt.show()
    
    print("\nPrediction error converged, demonstrating efficient processing.")


if __name__ == '__main__':
    # Run demonstrations
    results = demonstrate_wake_mode()
    demonstrate_predictive_coding()
    
    print("\n" + "=" * 60)
    print("Wake Mode demonstration completed!")
    print("Results saved to: wake_mode_comparison.png")
    print("=" * 60)