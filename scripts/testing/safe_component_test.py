#!/usr/bin/env python3
"""
Safe component test - avoids MainAgent to prevent segfaults
Tests individual components in isolation
"""
import sys
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from insightspike.core.config import get_config
from insightspike.utils.embedder import Embedder
from insightspike.core.layers.layer2_memory_manager import L2MemoryManager

def test_individual_components():
    """Test individual components to identify segfault source"""
    print("🔍 Testing individual components safely...")
    
    try:
        # Test 1: Configuration
        print("\n📋 Testing configuration...")
        config = get_config()
        print(f"✅ Config loaded successfully. Model: {config.model_config.default_model}")
        
        # Test 2: Embedder
        print("\n🧠 Testing embedder...")
        embedder = Embedder()
        test_text = "This is a test sentence for embedding."
        embedding = embedder.embed_text(test_text)
        print(f"✅ Embedder working. Embedding shape: {embedding.shape}")
        
        # Test 3: Memory Manager
        print("\n💾 Testing memory manager...")
        memory = L2MemoryManager()
        
        # Add some test episodes
        test_episodes = [
            "The cat sat on the mat",
            "Machine learning algorithms require data",
            "Python is a programming language",
            "Neural networks can learn patterns"
        ]
        
        for i, episode in enumerate(test_episodes):
            embedding = embedder.embed_text(episode)
            memory.store_episode(
                episode_id=f"test_{i}",
                content=episode,
                embedding=embedding,
                metadata={"source": "test", "index": i}
            )
        
        print(f"✅ Memory manager working. Stored {len(test_episodes)} episodes")
        
        # Test 4: Search functionality
        print("\n🔍 Testing search...")
        query = "programming language"
        query_embedding = embedder.embed_text(query)
        results = memory.search_similar_episodes(query_embedding, top_k=2)
        
        print(f"✅ Search working. Found {len(results)} similar episodes:")
        for result in results:
            print(f"  - Score: {result['similarity']:.3f}, Content: {result['content'][:50]}...")
        
        print("\n🎉 All individual components working successfully!")
        print("❌ Segfault likely occurs in MainAgent initialization or LLM components")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in component test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_individual_components()
    sys.exit(0 if success else 1)
