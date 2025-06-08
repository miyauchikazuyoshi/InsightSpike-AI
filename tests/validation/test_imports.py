#!/usr/bin/env python3
"""Test imports for memory reconstitution demo"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    print("Testing imports...")
    
    # Test config
    from insightspike.config import get_config
    print("✓ Config import successful")
    
    # Test memory manager
    from insightspike.core.layers.layer2_memory_manager import L2MemoryManager, Episode
    print("✓ Memory manager import successful")
    
    # Test graph reasoner
    from insightspike.core.layers.layer3_graph_reasoner import L3GraphReasoner
    print("✓ Graph reasoner import successful")
    
    # Test main agent
    from insightspike.core.agents.main_agent import MainAgent
    print("✓ Main agent import successful")
    
    print("\n🎉 All imports successful!")
    
    # Test basic initialization
    config = get_config()
    print(f"✓ Config loaded: {type(config)}")
    
    memory_manager = L2MemoryManager(dim=384, config=config)
    print(f"✓ Memory manager created: {type(memory_manager)}")
    
    graph_reasoner = L3GraphReasoner(config)
    print(f"✓ Graph reasoner created: {type(graph_reasoner)}")
    
    print("\n✅ Basic initialization test complete!")

except Exception as e:
    print(f"❌ Import/initialization failed: {e}")
    import traceback
    traceback.print_exc()
