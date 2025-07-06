#!/usr/bin/env python3
"""
PoC Test with torch-geometric enabled
Test the core functionality with GNN processing capabilities
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import torch
from torch_geometric.data import Data

try:
    from insightspike.core.learning.knowledge_graph_memory import KnowledgeGraphMemory
    print("✓ KnowledgeGraphMemory imported successfully")
except ImportError as e:
    print(f"✗ Failed to import KnowledgeGraphMemory: {e}")
    sys.exit(1)

try:
    from insightspike.core.layers.layer3_graph_reasoner import L3GraphReasoner
    print("✓ L3GraphReasoner imported successfully")
except ImportError as e:
    print(f"✗ Failed to import L3GraphReasoner: {e}")
    sys.exit(1)

def test_knowledge_graph_memory():
    """Test KnowledgeGraphMemory with torch-geometric."""
    print("\n=== Testing KnowledgeGraphMemory ===")
    
    # Initialize memory
    memory = KnowledgeGraphMemory(embedding_dim=64, similarity_threshold=0.3)
    print(f"✓ Initialized with embedding_dim=64, threshold=0.3")
    
    # Add some episode nodes
    embeddings = [
        np.random.randn(64).astype(np.float32),
        np.random.randn(64).astype(np.float32),
        np.random.randn(64).astype(np.float32),
    ]
    
    for i, emb in enumerate(embeddings):
        memory.add_episode_node(emb, i)
        print(f"✓ Added episode node {i}")
    
    print(f"✓ Graph has {memory.graph.x.size(0)} nodes")
    print(f"✓ Graph has {memory.graph.edge_index.size(1)} edges")
    
    # Test subgraph extraction
    subgraph = memory.get_subgraph([0, 1])
    print(f"✓ Subgraph extracted with {subgraph.x.size(0)} nodes")
    
    return True

def test_torch_geometric_integration():
    """Test torch-geometric integration."""
    print("\n=== Testing torch-geometric Integration ===")
    
    # Skip if using mocked torch (CI environment)
    if not hasattr(torch, '__file__'):
        print("✓ Skipping torch-specific test in mocked environment")
        return True
    
    # Create test graph
    x = torch.randn(4, 64)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)
    
    print(f"✓ Created test graph with {data.x.size(0)} nodes")
    print(f"✓ Graph has {data.edge_index.size(1)} edges")
    
    # Test basic torch-geometric operations
    from torch_geometric.utils import degree
    degrees = degree(data.edge_index[0])
    print(f"✓ Node degrees: {degrees.tolist()}")  # Convert to list to avoid SymBool issues
    
    return True

def test_memory_efficiency():
    """Test memory efficiency with larger graphs."""
    print("\n=== Testing Memory Efficiency ===")
    
    memory = KnowledgeGraphMemory(embedding_dim=128, similarity_threshold=0.5)
    
    # Add more nodes to test efficiency
    n_nodes = 50
    embeddings = [np.random.randn(128).astype(np.float32) for _ in range(n_nodes)]
    
    for i, emb in enumerate(embeddings):
        memory.add_episode_node(emb, i)
        if (i + 1) % 10 == 0:
            print(f"✓ Added {i + 1}/{n_nodes} nodes")
    
    print(f"✓ Final graph: {memory.graph.x.size(0)} nodes, {memory.graph.edge_index.size(1)} edges")
    
    # Test subgraph efficiency
    large_subgraph = memory.get_subgraph(list(range(0, min(20, n_nodes))))
    print(f"✓ Large subgraph: {large_subgraph.x.size(0)} nodes")
    
    return True

def main():
    """Run PoC tests."""
    print("InsightSpike-AI PoC Test with torch-geometric")
    print("=" * 50)
    
    try:
        # Test torch-geometric availability
        import torch_geometric
        print(f"✓ torch-geometric version: {torch_geometric.__version__}")
        print(f"✓ PyTorch version: {torch.__version__}")
        
        # Run tests
        test_torch_geometric_integration()
        test_knowledge_graph_memory()
        test_memory_efficiency()
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed! torch-geometric integration successful")
        print("GNN processing capabilities are now available")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
