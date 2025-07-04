#!/usr/bin/env python3
"""Simple comparison test"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from insightspike.core.layers.scalable_graph_builder import ScalableGraphBuilder

print("=== Simple Comparison Test ===")

# Generate test documents
n_docs = 50
documents = []

for i in range(n_docs):
    topic_id = i % 5  # 5 topics
    
    # Create topic-based embedding
    embedding = np.zeros(384, dtype=np.float32)
    embedding[topic_id * 70:(topic_id + 1) * 70] = np.random.randn(70)
    embedding = embedding / np.linalg.norm(embedding)
    
    doc = {
        'text': f'Document {i} about topic {topic_id}',
        'embedding': embedding,
        'id': i
    }
    documents.append(doc)

# Test ScalableGraphBuilder
print("\nTesting ScalableGraphBuilder...")
builder = ScalableGraphBuilder()
builder.similarity_threshold = 0.3

start = time.time()
graph = builder.build_graph(documents)
elapsed = time.time() - start

print(f"Time: {elapsed:.3f}s")
print(f"Nodes: {graph.num_nodes}")
print(f"Edges: {graph.edge_index.size(1)}")
print(f"Avg edges/node: {graph.edge_index.size(1) / graph.num_nodes:.1f}")

# Show some connections
if graph.edge_index.size(1) > 0:
    print("\nSample connections:")
    edges = graph.edge_index.t().numpy()
    for i in range(min(5, len(edges))):
        src, dst = edges[i]
        print(f"  Doc {src} <-> Doc {dst}")

print("\n✅ Test complete!")