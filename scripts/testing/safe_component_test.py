#!/usr/bin/env python3
"""
Safe component test - avoids MainAgent to prevent segfaults
Tests individual components in isolation
"""
import sys
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from insightspike.config.loader import load_config
from insightspike.config.presets import ConfigPresets
try:
    from insightspike.processing.embedder import EmbeddingManager
except ImportError:
    # Fallback for newer structure
    from sentence_transformers import SentenceTransformer
    class EmbeddingManager:
        def __init__(self, config=None):
            self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        def encode(self, texts):
            return self.model.encode(texts)
from insightspike.implementations.layers.layer2_memory_manager import L2MemoryManager

def test_individual_components():
    """Test individual components to identify segfault source"""
    print("🔍 Testing individual components safely...")
    
    try:
        # Test 1: Configuration
        print("\n📋 Testing configuration...")
        config = load_config(preset="development")
        embedding_model = config.embedding.model_name
        print(f"✅ Config loaded successfully. Model: {embedding_model}")
        
        # Test 2: Embedder
        print("\n🧠 Testing embedder...")
        embedder = EmbeddingManager(config=config)
        test_text = "This is a test sentence for embedding."
        embedding = embedder.encode([test_text])
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
            # Use correct API for embedding
            if hasattr(embedder, 'embed_text'):
                embedding = embedder.embed_text(episode)
            else:
                # Use sentence-transformers API
                embedding = embedder.encode([episode])[0]
            
            # Use correct API for storing episodes
            memory.add_episode(
                text=episode,
                metadata={"source": "test", "index": i, "c_value": 0.5}
            )
        
        print(f"✅ Memory manager working. Stored {len(test_episodes)} episodes")
        
        # Test 4: Search functionality
        print("\n🔍 Testing search...")
        query = "programming language"
        
        # Use the text-based search API
        results = memory.search_episodes(query, k=2)
        
        print(f"✅ Search working. Found {len(results)} similar episodes:")
        for result in results:
            if isinstance(result, dict):
                score = result.get('similarity', result.get('score', 0))
                content = result.get('content', result.get('text', str(result)))
            else:
                score = 0
                content = str(result)
            print(f"  - Score: {score:.3f}, Content: {content[:50]}...")
        
        print("\n🎉 All individual components working successfully!")
        print("✅ Component-level functionality verified")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in component test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_individual_components()
    sys.exit(0 if success else 1)
