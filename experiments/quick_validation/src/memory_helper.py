"""
Helper functions for working with InsightSpike memory system
"""

import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer

from insightspike.core.layers.layer2_memory_manager import L2MemoryManager
from insightspike.processing.embedder import get_model


class MemoryHelper:
    """Helper class for proper memory management"""
    
    def __init__(self):
        self.embedder = get_model()
        self.memory = L2MemoryManager()
    
    def add_documents(self, documents: List[str], context_prefix: str = "doc") -> int:
        """Add documents to memory with proper embeddings"""
        success_count = 0
        
        for i, doc in enumerate(documents):
            try:
                # Generate embedding
                embedding = self.embedder.encode(doc, show_progress_bar=False)
                embedding = embedding.astype(np.float32)
                
                # Add to memory (vector, text, c_value)
                episode_id = self.memory.add_episode(
                    embedding, 
                    doc,
                    0.5  # Default C-value
                )
                
                if episode_id >= 0:
                    success_count += 1
                    print(f"✓ Added document {i+1}: {doc[:50]}...")
                else:
                    print(f"✗ Failed to add document {i+1}")
                    
            except Exception as e:
                print(f"✗ Error adding document {i+1}: {e}")
        
        print(f"\nSuccessfully added {success_count}/{len(documents)} documents")
        return success_count
    
    def search(self, query: str, top_k: int = 3) -> List[dict]:
        """Search memory for relevant documents"""
        try:
            # Generate query embedding
            query_embedding = self.embedder.encode(query, show_progress_bar=False)
            query_embedding = query_embedding.astype(np.float32)
            
            # Use search_episodes method which works with text queries
            results = self.memory.search_episodes(query, k=top_k)
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "text": result["text"],
                    "score": result["similarity"],
                    "c_value": result.get("c_value", 0.5)
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def get_memory_stats(self) -> dict:
        """Get memory statistics"""
        return {
            "num_episodes": len(self.memory.episodes),
            "is_trained": self.memory.is_trained,
            "embedding_dim": self.memory.dim,
            "index_type": "FAISS" if hasattr(self.memory, 'index') else "None"
        }


def test_memory_system():
    """Test the memory system with sample data"""
    print("Testing InsightSpike Memory System...")
    
    helper = MemoryHelper()
    
    # Test documents
    test_docs = [
        "Sleep is crucial for memory consolidation and learning.",
        "REM sleep specifically helps with procedural memory.",
        "Exercise improves brain health through BDNF production.",
        "Neuroplasticity allows the brain to adapt and change."
    ]
    
    # Add documents
    print("\n1. Adding documents to memory:")
    helper.add_documents(test_docs)
    
    # Check stats
    print("\n2. Memory statistics:")
    stats = helper.get_memory_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Test search
    print("\n3. Testing search:")
    queries = [
        "How does sleep affect memory?",
        "What improves brain health?"
    ]
    
    for query in queries:
        print(f"\n   Query: {query}")
        results = helper.search(query, top_k=2)
        for i, result in enumerate(results):
            print(f"   Result {i+1}: {result['text'][:60]}...")
            print(f"            Score: {result['score']:.3f}")
    
    return helper


if __name__ == "__main__":
    test_memory_system()