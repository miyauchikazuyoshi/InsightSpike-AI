#!/usr/bin/env python3
"""
Simple scalability test comparing GraphBuilder implementations
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from insightspike.core.layers.layer3_graph_reasoner import GraphBuilder
from insightspike.core.layers.scalable_graph_builder import ScalableGraphBuilder


def generate_test_documents(n: int):
    """Generate test documents with embeddings"""
    documents = []
    embedding_dim = 384
    
    # Generate random embeddings
    embeddings = np.random.randn(n, embedding_dim)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    for i in range(n):
        doc = {
            'text': f'Document {i}',
            'embedding': embeddings[i],
            'id': i
        }
        documents.append(doc)
        
    return documents


def test_builders():
    """Test both builders with increasing document sizes"""
    test_sizes = [100, 300, 500, 1000]
    
    print("=== Scalability Performance Test ===\n")
    
    for size in test_sizes:
        print(f"\nTesting with {size} documents:")
        
        # Generate test data
        documents = generate_test_documents(size)
        
        # Test original builder (only for smaller sizes)
        if size <= 500:
            try:
                builder = GraphBuilder()
                start = time.time()
                graph = builder.build_graph(documents)
                orig_time = time.time() - start
                print(f"  Original GraphBuilder: {orig_time:.2f}s")
                print(f"    Nodes: {graph.num_nodes if hasattr(graph, 'num_nodes') else 'N/A'}")
                print(f"    Edges: {graph.edge_index.size(1) if hasattr(graph, 'edge_index') else 'N/A'}")
            except Exception as e:
                print(f"  Original GraphBuilder failed: {e}")
                orig_time = None
        else:
            print(f"  Skipping Original GraphBuilder (too large)")
            orig_time = None
        
        # Test scalable builder
        try:
            builder = ScalableGraphBuilder()
            start = time.time()
            graph = builder.build_graph(documents)
            scale_time = time.time() - start
            print(f"  Scalable GraphBuilder: {scale_time:.2f}s")
            print(f"    Nodes: {graph.num_nodes}")
            print(f"    Edges: {graph.edge_index.size(1)}")
            
            if orig_time:
                speedup = orig_time / scale_time
                print(f"  Speedup: {speedup:.2f}x")
        except Exception as e:
            print(f"  Scalable GraphBuilder failed: {e}")
    
    print("\n=== Test Complete ===")


if __name__ == "__main__":
    test_builders()