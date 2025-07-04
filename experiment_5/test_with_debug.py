#!/usr/bin/env python3
"""
Test ScalableGraphBuilder with debug info
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from insightspike.core.layers.scalable_graph_builder import ScalableGraphBuilder
from insightspike.core.config import get_config


def test_with_similar_documents():
    """Test with documents that should have high similarity"""
    print("Testing ScalableGraphBuilder with similar documents...")
    
    # Create builder
    config = get_config()
    print(f"Config similarity threshold: {config.reasoning.similarity_threshold}")
    
    builder = ScalableGraphBuilder()
    print(f"Builder similarity threshold: {builder.similarity_threshold}")
    print(f"Builder top_k: {builder.top_k}")
    
    # Create similar documents (high cosine similarity)
    base_embedding = np.random.randn(384)
    base_embedding = base_embedding / np.linalg.norm(base_embedding)
    
    documents = []
    for i in range(10):
        # Add small noise to create similar but not identical embeddings
        noise = np.random.randn(384) * 0.1
        embedding = base_embedding + noise
        embedding = embedding / np.linalg.norm(embedding)
        
        doc = {
            'text': f'Similar document {i} about machine learning',
            'embedding': embedding.astype(np.float32),
            'id': i
        }
        documents.append(doc)
    
    # Add some different documents
    for i in range(5):
        embedding = np.random.randn(384)
        embedding = embedding / np.linalg.norm(embedding)
        
        doc = {
            'text': f'Different document {i} about quantum physics',
            'embedding': embedding.astype(np.float32),
            'id': 10 + i
        }
        documents.append(doc)
    
    print(f"✓ Created {len(documents)} documents (10 similar, 5 different)")
    
    # Test similarity manually
    from sklearn.metrics.pairwise import cosine_similarity
    embeddings = np.array([d['embedding'] for d in documents])
    sim_matrix = cosine_similarity(embeddings)
    
    print(f"\nSimilarity matrix stats:")
    print(f"  Min similarity: {sim_matrix.min():.3f}")
    print(f"  Max similarity: {sim_matrix.max():.3f}")
    print(f"  Mean similarity: {sim_matrix.mean():.3f}")
    
    # Count how many pairs exceed threshold
    threshold = builder.similarity_threshold
    above_threshold = (sim_matrix > threshold).sum() - len(documents)  # Subtract diagonal
    print(f"  Pairs above threshold ({threshold}): {above_threshold}")
    
    # Build graph
    try:
        graph = builder.build_graph(documents)
        print(f"\n✓ Graph built successfully!")
        print(f"  Nodes: {graph.num_nodes}")
        print(f"  Edges: {graph.edge_index.size(1)}")
        
        # Check if edges match expectations
        if graph.edge_index.size(1) == 0:
            print("\n⚠️  No edges created! Debugging...")
            print(f"  FAISS index: {builder.index}")
            print(f"  Embeddings shape in builder: {builder.embeddings.shape if builder.embeddings is not None else 'None'}")
            
    except Exception as e:
        print(f"✗ Graph building failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_with_similar_documents()