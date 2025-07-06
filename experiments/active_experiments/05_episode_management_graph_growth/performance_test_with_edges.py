#!/usr/bin/env python3
"""
Performance test with proper edge creation
"""

import os
import sys
import time
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from insightspike.core.layers.layer3_graph_reasoner import GraphBuilder
from insightspike.core.layers.scalable_graph_builder import ScalableGraphBuilder


def generate_clustered_documents(n: int):
    """Generate documents with clear clusters for better edge creation"""
    documents = []
    
    # Create fewer, tighter clusters
    n_clusters = min(10, max(3, n // 20))
    docs_per_cluster = n // n_clusters
    
    for cluster in range(n_clusters):
        # Base embedding for this cluster
        base = np.random.randn(384)
        base = base / np.linalg.norm(base)
        
        for i in range(docs_per_cluster):
            # Small noise for tight clusters
            noise = np.random.randn(384) * 0.1  # Reduced noise
            embedding = base + noise
            embedding = embedding / np.linalg.norm(embedding)
            
            doc = {
                'text': f'Cluster {cluster}: Document about {"AI" if cluster % 2 == 0 else "ML"} - item {i}',
                'embedding': embedding.astype(np.float32),
                'id': cluster * docs_per_cluster + i
            }
            documents.append(doc)
    
    return documents[:n]


def run_comparison():
    """Run performance comparison with edge creation"""
    print("=== Performance Test with Edge Creation ===")
    print(f"Start time: {datetime.now()}\n")
    
    # Test sizes
    test_sizes = [50, 100, 200, 300, 500, 1000, 2000]
    
    results = {
        'original': {},
        'scalable': {},
        'scalable_low_threshold': {}
    }
    
    for size in test_sizes:
        print(f"\n{'='*70}")
        print(f"Testing with {size} documents")
        print(f"{'='*70}")
        
        # Generate clustered data
        documents = generate_clustered_documents(size)
        
        # Test original GraphBuilder (only for small sizes)
        if size <= 300:
            try:
                print(f"\n1. Original GraphBuilder:")
                builder = GraphBuilder()
                
                start = time.time()
                graph = builder.build_graph(documents)
                build_time = time.time() - start
                
                results['original'][size] = {
                    'time': build_time,
                    'nodes': graph.num_nodes if hasattr(graph, 'num_nodes') else size,
                    'edges': graph.edge_index.size(1) if hasattr(graph, 'edge_index') else 0
                }
                
                print(f"   Build time: {build_time:.3f}s")
                print(f"   Edges: {results['original'][size]['edges']}")
                print(f"   Avg edges/node: {results['original'][size]['edges']/size:.1f}")
                
            except Exception as e:
                print(f"   Failed: {e}")
        else:
            print(f"\n1. Skipping Original (too large)")
        
        # Test ScalableGraphBuilder with default threshold
        try:
            print(f"\n2. Scalable GraphBuilder (threshold=0.3):")
            builder = ScalableGraphBuilder()
            
            start = time.time()
            graph = builder.build_graph(documents)
            build_time = time.time() - start
            
            results['scalable'][size] = {
                'time': build_time,
                'nodes': graph.num_nodes,
                'edges': graph.edge_index.size(1)
            }
            
            print(f"   Build time: {build_time:.3f}s")
            print(f"   Edges: {graph.edge_index.size(1)}")
            if graph.edge_index.size(1) > 0:
                print(f"   Avg edges/node: {graph.edge_index.size(1)/size:.1f}")
                
        except Exception as e:
            print(f"   Failed: {e}")
        
        # Test with lower threshold for more edges
        try:
            print(f"\n3. Scalable GraphBuilder (threshold=0.1):")
            builder = ScalableGraphBuilder()
            builder.similarity_threshold = 0.1  # Lower threshold
            
            start = time.time()
            graph = builder.build_graph(documents)
            build_time = time.time() - start
            
            results['scalable_low_threshold'][size] = {
                'time': build_time,
                'nodes': graph.num_nodes,
                'edges': graph.edge_index.size(1)
            }
            
            print(f"   Build time: {build_time:.3f}s")
            print(f"   Edges: {graph.edge_index.size(1)}")
            print(f"   Avg edges/node: {graph.edge_index.size(1)/size:.1f}")
            
            # Compare with original if available
            if size in results['original']:
                speedup = results['original'][size]['time'] / build_time
                print(f"\n   🚀 Speedup vs Original: {speedup:.2f}x")
                
        except Exception as e:
            print(f"   Failed: {e}")
    
    # Final summary
    print(f"\n{'='*70}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*70}")
    
    print("\nBuild Time Comparison:")
    print(f"{'Size':>6} | {'Original':>10} | {'Scalable':>10} | {'Scalable-0.1':>13} | {'Speedup':>10}")
    print("-" * 65)
    
    for size in test_sizes:
        orig = results['original'].get(size, {}).get('time', None)
        scale = results['scalable'].get(size, {}).get('time', None)
        scale_low = results['scalable_low_threshold'].get(size, {}).get('time', None)
        
        orig_str = f"{orig:.3f}s" if orig else "N/A"
        scale_str = f"{scale:.3f}s" if scale else "N/A"
        scale_low_str = f"{scale_low:.3f}s" if scale_low else "N/A"
        
        speedup = "N/A"
        if orig and scale_low:
            speedup = f"{orig/scale_low:.2f}x"
        
        print(f"{size:>6} | {orig_str:>10} | {scale_str:>10} | {scale_low_str:>13} | {speedup:>10}")
    
    print("\nEdge Count Comparison:")
    print(f"{'Size':>6} | {'Original':>10} | {'Scalable-0.3':>13} | {'Scalable-0.1':>13}")
    print("-" * 50)
    
    for size in test_sizes:
        orig = results['original'].get(size, {}).get('edges', 'N/A')
        scale = results['scalable'].get(size, {}).get('edges', 'N/A')
        scale_low = results['scalable_low_threshold'].get(size, {}).get('edges', 'N/A')
        
        print(f"{size:>6} | {str(orig):>10} | {str(scale):>13} | {str(scale_low):>13}")
    
    # Save results
    import json
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'edge_performance_results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to edge_performance_results_{timestamp}.json")


if __name__ == "__main__":
    run_comparison()