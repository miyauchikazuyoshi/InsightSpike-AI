#!/usr/bin/env python3
"""
Simple performance comparison between GraphBuilder implementations
"""

import os
import sys
import time
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from insightspike.core.layers.layer3_graph_reasoner import GraphBuilder
from insightspike.core.layers.scalable_graph_builder import ScalableGraphBuilder


def generate_test_documents(n: int):
    """Generate test documents with realistic embeddings"""
    documents = []
    
    # Create clusters of similar documents
    n_clusters = max(5, n // 50)
    docs_per_cluster = n // n_clusters
    
    for cluster in range(n_clusters):
        # Base embedding for this cluster
        base = np.random.randn(384)
        base = base / np.linalg.norm(base)
        
        for i in range(docs_per_cluster):
            # Add noise to create variation
            noise = np.random.randn(384) * 0.3
            embedding = base + noise
            embedding = embedding / np.linalg.norm(embedding)
            
            doc = {
                'text': f'Document {cluster*docs_per_cluster + i} in cluster {cluster}',
                'embedding': embedding.astype(np.float32),
                'id': cluster * docs_per_cluster + i
            }
            documents.append(doc)
    
    # Add remaining documents
    for i in range(len(documents), n):
        embedding = np.random.randn(384)
        embedding = embedding / np.linalg.norm(embedding)
        doc = {
            'text': f'Random document {i}',
            'embedding': embedding.astype(np.float32),
            'id': i
        }
        documents.append(doc)
    
    return documents


def test_performance():
    """Compare performance of GraphBuilder implementations"""
    print("=== Performance Comparison: Original vs Scalable GraphBuilder ===")
    print(f"Start time: {datetime.now()}\n")
    
    # Test sizes
    test_sizes = [50, 100, 200, 300, 500, 1000]
    
    results = {
        'original': {},
        'scalable': {}
    }
    
    for size in test_sizes:
        print(f"\n{'='*60}")
        print(f"Testing with {size} documents")
        print(f"{'='*60}")
        
        # Generate test data
        documents = generate_test_documents(size)
        
        # Test original GraphBuilder (skip for large sizes)
        if size <= 300:
            try:
                print(f"\n1. Original GraphBuilder:")
                builder = GraphBuilder()
                
                start = time.time()
                graph = builder.build_graph(documents)
                build_time = time.time() - start
                
                n_edges = graph.edge_index.size(1) if hasattr(graph, 'edge_index') else 0
                
                results['original'][size] = {
                    'time': build_time,
                    'nodes': graph.num_nodes if hasattr(graph, 'num_nodes') else size,
                    'edges': n_edges
                }
                
                print(f"   Build time: {build_time:.3f}s")
                print(f"   Nodes: {results['original'][size]['nodes']}")
                print(f"   Edges: {n_edges}")
                
            except Exception as e:
                print(f"   Failed: {e}")
                results['original'][size] = {'error': str(e)}
        else:
            print(f"\n1. Skipping Original GraphBuilder (too large)")
        
        # Test ScalableGraphBuilder
        try:
            print(f"\n2. Scalable GraphBuilder (FAISS):")
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
            print(f"   Nodes: {graph.num_nodes}")
            print(f"   Edges: {graph.edge_index.size(1)}")
            
            # Calculate speedup if we have both results
            if size in results['original'] and 'time' in results['original'][size]:
                speedup = results['original'][size]['time'] / build_time
                print(f"\n   🚀 Speedup: {speedup:.2f}x faster!")
                
        except Exception as e:
            print(f"   Failed: {e}")
            results['scalable'][size] = {'error': str(e)}
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    print("\nBuild Times:")
    print(f"{'Size':>8} | {'Original':>12} | {'Scalable':>12} | {'Speedup':>10}")
    print("-" * 50)
    
    for size in test_sizes:
        orig_time = results['original'].get(size, {}).get('time', None)
        scale_time = results['scalable'].get(size, {}).get('time', None)
        
        orig_str = f"{orig_time:.3f}s" if orig_time else "N/A"
        scale_str = f"{scale_time:.3f}s" if scale_time else "N/A"
        
        speedup_str = "N/A"
        if orig_time and scale_time:
            speedup = orig_time / scale_time
            speedup_str = f"{speedup:.2f}x"
        
        print(f"{size:>8} | {orig_str:>12} | {scale_str:>12} | {speedup_str:>10}")
    
    # Save results
    import json
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'performance_comparison_{timestamp}.json'
    
    with open(filename, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'results': results
        }, f, indent=2)
    
    print(f"\nResults saved to {filename}")
    
    # Graph density analysis
    print("\nGraph Density Analysis:")
    print(f"{'Size':>8} | {'Original Edges':>15} | {'Scalable Edges':>15}")
    print("-" * 45)
    
    for size in test_sizes:
        orig_edges = results['original'].get(size, {}).get('edges', 'N/A')
        scale_edges = results['scalable'].get(size, {}).get('edges', 'N/A')
        print(f"{size:>8} | {str(orig_edges):>15} | {str(scale_edges):>15}")


if __name__ == "__main__":
    test_performance()