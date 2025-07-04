#!/usr/bin/env python3
"""
Test ScalableGraphBuilder in isolation
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from insightspike.core.layers.scalable_graph_builder import ScalableGraphBuilder


def test_scalable_builder():
    """Test basic functionality"""
    print("Testing ScalableGraphBuilder...")
    
    # Create builder
    builder = ScalableGraphBuilder()
    print(f"✓ Builder created")
    
    # Test empty graph
    empty_graph = builder._empty_graph()
    print(f"✓ Empty graph created: nodes={empty_graph.num_nodes}, edges={empty_graph.edge_index.shape}")
    
    # Create test documents
    documents = []
    for i in range(10):
        embedding = np.random.randn(384).astype(np.float32)
        doc = {
            'text': f'Test document {i}',
            'embedding': embedding,
            'id': i
        }
        documents.append(doc)
    
    print(f"✓ Created {len(documents)} test documents")
    
    # Build graph
    try:
        graph = builder.build_graph(documents)
        print(f"✓ Graph built successfully!")
        print(f"  Nodes: {graph.num_nodes}")
        print(f"  Edges: {graph.edge_index.size(1)}")
        print(f"  Node features shape: {graph.x.shape}")
    except Exception as e:
        print(f"✗ Graph building failed: {e}")
        import traceback
        traceback.print_exc()
        
    # Test incremental update
    try:
        new_docs = []
        for i in range(5):
            embedding = np.random.randn(384).astype(np.float32)
            doc = {
                'text': f'New document {i}',
                'embedding': embedding,
                'id': 10 + i
            }
            new_docs.append(doc)
            
        graph2 = builder.build_graph(new_docs, incremental=True)
        print(f"✓ Incremental update successful!")
        print(f"  Total nodes: {graph2.num_nodes}")
        print(f"  Total edges: {graph2.edge_index.size(1)}")
    except Exception as e:
        print(f"✗ Incremental update failed: {e}")
        

if __name__ == "__main__":
    test_scalable_builder()