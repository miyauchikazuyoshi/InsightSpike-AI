#!/usr/bin/env python3
"""
Validation script for torch-geometric integration in CI environment
Tests both torch-geometric availability and fallback functionality
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

def test_torch_geometric_availability():
    """Test if torch-geometric is available in CI environment."""
    print("🔍 Testing torch-geometric availability...")
    
    try:
        import torch_geometric
        from torch_geometric.data import Data
        from torch_geometric.utils import subgraph
        
        print(f"✅ Torch-geometric version: {torch_geometric.__version__}")
        
        # Basic functionality test
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).t()
        data = Data(x=x, edge_index=edge_index)
        
        print(f"✅ Basic Data object created: {data.num_nodes} nodes, {data.num_edges} edges")
        return True
        
    except ImportError as e:
        print(f"⚠️ Torch-geometric not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Torch-geometric test failed: {e}")
        return False

def test_knowledge_graph_memory():
    """Test KnowledgeGraphMemory with or without torch-geometric."""
    print("\n🧠 Testing KnowledgeGraphMemory...")
    
    try:
        from insightspike.core.learning.knowledge_graph_memory import KnowledgeGraphMemory
        
        # Test basic functionality
        kg_memory = KnowledgeGraphMemory(embedding_dim=4)
        print("✅ KnowledgeGraphMemory initialized successfully")
        
        # Test adding nodes
        import numpy as np
        embedding1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        embedding2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        
        kg_memory.add_episode_node(embedding1, 0)
        kg_memory.add_episode_node(embedding2, 1)
        
        print(f"✅ Added 2 nodes, graph has {kg_memory.graph.num_nodes} nodes")
        
        # Test subgraph extraction
        subgraph = kg_memory.get_subgraph([0])
        print(f"✅ Subgraph extraction successful: {subgraph.num_nodes} nodes")
        
        return True
        
    except Exception as e:
        print(f"❌ KnowledgeGraphMemory test failed: {e}")
        return False

def main():
    """Main validation function."""
    print("🚀 Torch-geometric CI Validation")
    print("=" * 50)
    
    import torch
    print(f"PyTorch version: {torch.__version__}")
    
    # Test torch-geometric availability
    torch_geo_available = test_torch_geometric_availability()
    
    # Test knowledge graph memory (should work with or without torch-geometric)
    kg_memory_working = test_knowledge_graph_memory()
    
    print("\n" + "=" * 50)
    print("📊 Validation Results:")
    print(f"   Torch-geometric available: {'✅' if torch_geo_available else '⚠️'}")
    print(f"   KnowledgeGraphMemory working: {'✅' if kg_memory_working else '❌'}")
    
    if kg_memory_working:
        print("\n🎉 Validation PASSED - Core functionality working")
        if not torch_geo_available:
            print("   (Running in fallback mode without torch-geometric)")
        return 0
    else:
        print("\n❌ Validation FAILED - Core functionality broken")
        return 1

if __name__ == "__main__":
    import torch
    sys.exit(main())
