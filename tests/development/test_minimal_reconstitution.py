#!/usr/bin/env python3
"""Minimal memory reconstitution test"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    print("🧠 Minimal Memory Reconstitution Test")
    print("=" * 40)
    
    try:
        # Test 1: Import config
        print("Step 1: Testing config import...")
        from insightspike.core.config import get_config
        config = get_config()
        print(f"✓ Config loaded: {type(config).__name__}")
        
        # Test 2: Import memory manager
        print("Step 2: Testing memory manager import...")
        from insightspike.core.layers.layer2_memory_manager import L2MemoryManager, Episode
        memory_manager = L2MemoryManager(dim=384, config=config)
        print(f"✓ Memory manager created: {type(memory_manager).__name__}")
        
        # Test 3: Create sample episode
        print("Step 3: Creating sample episode...")
        import numpy as np
        embedding = np.random.random(384).astype(np.float32)
        episode = Episode(embedding, "Test episode", 0.5, {"test": True})
        memory_manager.episodes.append(episode)
        print(f"✓ Episode created and added: C-value={episode.c}")
        
        # Test 4: Import graph reasoner
        print("Step 4: Testing graph reasoner import...")
        from insightspike.core.layers.layer3_graph_reasoner import L3GraphReasoner
        graph_reasoner = L3GraphReasoner(config)
        print(f"✓ Graph reasoner created: {type(graph_reasoner).__name__}")
        
        # Test 5: Test memory sync callback
        print("Step 5: Testing memory sync setup...")
        def memory_sync_callback(documents, updated_c_values):
            return memory_manager.sync_c_values_from_graph(documents, updated_c_values)
        
        graph_reasoner.set_memory_sync_callback(memory_sync_callback)
        print("✓ Memory sync callback set")
        
        print("\n🎉 All basic tests passed!")
        print(f"Memory episodes: {len(memory_manager.episodes)}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
