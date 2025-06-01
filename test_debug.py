#!/usr/bin/env python3
"""Debug test for GraphBuilder import issue"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_import():
    """Test GraphBuilder import directly"""
    try:
        print("Testing torch import...")
        import torch
        print(f"✓ torch imported successfully: {torch.__version__}")
        
        print("Testing torch_geometric import...")
        import torch_geometric
        print(f"✓ torch_geometric imported successfully: {torch_geometric.__version__}")
        
        print("Testing GraphBuilder import...")
        from insightspike.core.layers.layer3_graph_reasoner import GraphBuilder
        print("✓ GraphBuilder imported successfully")
        
        print("Testing GraphBuilder creation...")
        gb = GraphBuilder()
        print(f"✓ GraphBuilder created successfully with similarity_threshold: {gb.similarity_threshold}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_import()
    if success:
        print("\n🎉 All imports successful!")
    else:
        print("\n❌ Import failed!")
        sys.exit(1)
