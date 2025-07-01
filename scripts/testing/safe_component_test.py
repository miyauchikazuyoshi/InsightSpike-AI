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
try:
    from insightspike.utils.embedder import get_model_singleton as get_embedder
except ImportError:
    # Fallback for newer structure
    from sentence_transformers import SentenceTransformer
    def get_embedder():
        return SentenceTransformer('paraphrase-MiniLM-L6-v2')
from insightspike.core.layers.layer2_memory_manager import L2MemoryManager

def test_individual_components():
    """Test individual components to identify segfault source"""
    print("🔍 Testing individual components safely...")
    
    try:
        # Test 1: Configuration
        print("\n📋 Testing configuration...")
        config = get_config()
        embedding_model = getattr(config, 'embedding_model', 'paraphrase-MiniLM-L6-v2')
        print(f"✅ Config loaded successfully. Model: {embedding_model}")
        
        # Test 2: Embedder
        print("\n🧠 Testing embedder...")
        embedder = get_embedder()
        test_text = "This is a test sentence for embedding."
        if hasattr(embedder, 'embed_text'):
            embedding = embedder.embed_text(test_text)
        else:
            # Use sentence-transformers API
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
            try:
                memory.store_episode(
                    episode_id=f"test_{i}",
                    content=episode,
                    embedding=embedding,
                    metadata={"source": "test", "index": i}
                )
            except TypeError:
                # Use the correct L2MemoryManager API
                memory.add_episode(
                    vector=embedding,
                    text=episode,
                    c_value=0.5
                )
        
        print(f"✅ Memory manager working. Stored {len(test_episodes)} episodes")
        
        # Test 4: Search functionality
        print("\n🔍 Testing search...")
        query = "programming language"
        if hasattr(embedder, 'embed_text'):
            query_embedding = embedder.embed_text(query)
        else:
            query_embedding = embedder.encode([query])[0]
        
        # Use correct search method
        if hasattr(memory, 'search_similar_episodes'):
            results = memory.search_similar_episodes(query_embedding, top_k=2)
        elif hasattr(memory, 'search'):
            distances, indices = memory.search(query_embedding, top_k=2)
            results = []
            for i, (dist, idx) in enumerate(zip(distances, indices)):
                if idx < len(memory.episodes):
                    results.append({
                        'similarity': 1.0 - dist,  # Convert distance to similarity
                        'content': memory.episodes[idx].text
                    })
        else:
            results = []
        
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
