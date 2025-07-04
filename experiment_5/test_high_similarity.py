#!/usr/bin/env python3
"""
Test with very high similarity documents
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from insightspike.core.layers.scalable_graph_builder import ScalableGraphBuilder


def test_high_similarity():
    """Test with documents that have very high similarity"""
    print("Testing ScalableGraphBuilder with high similarity documents...")
    
    builder = ScalableGraphBuilder()
    print(f"Similarity threshold: {builder.similarity_threshold}")
    
    # Create almost identical documents
    base_embedding = np.random.randn(384)
    base_embedding = base_embedding / np.linalg.norm(base_embedding)
    
    documents = []
    for i in range(10):
        # Very small noise
        noise = np.random.randn(384) * 0.01
        embedding = base_embedding + noise
        embedding = embedding / np.linalg.norm(embedding)
        
        doc = {
            'text': f'Document {i}: Machine learning is transforming industries',
            'embedding': embedding.astype(np.float32),
            'id': i
        }
        documents.append(doc)
    
    # Check similarities
    from sklearn.metrics.pairwise import cosine_similarity
    embeddings = np.array([d['embedding'] for d in documents])
    sim_matrix = cosine_similarity(embeddings)
    
    print(f"\nSimilarity matrix (first 5x5):")
    print(sim_matrix[:5, :5])
    print(f"\nMin similarity (excluding diagonal): {np.min(sim_matrix[sim_matrix < 0.999]):.3f}")
    print(f"Max similarity (excluding diagonal): {np.max(sim_matrix[sim_matrix < 0.999]):.3f}")
    
    # Build graph
    graph = builder.build_graph(documents)
    print(f"\n✓ Graph built!")
    print(f"  Nodes: {graph.num_nodes}")
    print(f"  Edges: {graph.edge_index.size(1)}")
    
    # Test with lower threshold
    builder.similarity_threshold = 0.1
    print(f"\nTesting with lower threshold: {builder.similarity_threshold}")
    graph2 = builder.build_graph(documents)
    print(f"  Nodes: {graph2.num_nodes}")
    print(f"  Edges: {graph2.edge_index.size(1)}")


if __name__ == "__main__":
    test_high_similarity()