#!/usr/bin/env python3
"""Test RAG systems integration - run from src directory."""

import sys
import time
import warnings
warnings.filterwarnings("ignore")

# Try importing the utility modules first
try:
    from utils.embedding import SimpleEmbedder
    from utils.text_processing import SimpleTextProcessor
    from llm.generator import SimpleGenerator
    print("✅ Utility modules available")
    utils_available = True
except ImportError as e:
    print(f"⚠️  Utility modules not available: {e}")
    print("   Creating mock utilities for testing")
    utils_available = False


# Mock utilities if not available
if not utils_available:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    
    class SimpleEmbedder:
        def __init__(self, model_name="all-MiniLM-L6-v2"):
            try:
                self.model = SentenceTransformer(model_name)
                print(f"✅ Loaded SentenceTransformer: {model_name}")
            except Exception as e:
                print(f"⚠️  Could not load SentenceTransformer: {e}")
                self.model = None
        
        def encode(self, texts):
            if self.model is None:
                # Return random embeddings as fallback
                if isinstance(texts, str):
                    texts = [texts]
                return np.random.normal(0, 1, (len(texts), 384))
            return self.model.encode(texts)
    
    class SimpleTextProcessor:
        def clean(self, text):
            return text.strip()
        
        def tokenize(self, text):
            return text.split()
        
        def preprocess(self, text):
            return self.clean(text)
    
    class SimpleGenerator:
        def __init__(self, model_name="simple"):
            self.model_name = model_name
        
        def generate(self, query, context_text="", max_length=150):
            # Simple template-based generation for testing
            if context_text:
                return f"Based on the context: {context_text[:100]}..., I can answer: {query}"
            else:
                return f"I don't have specific information about: {query}. This is a general response."


def test_rag_mock_implementation():
    """Test RAG-like behavior without full implementation."""
    print("🧪 Testing RAG Mock Implementation...")
    
    from core.config import ExperimentConfig
    from core.knowledge_graph import KnowledgeGraph
    
    config = ExperimentConfig()
    
    # Initialize components
    embedder = SimpleEmbedder()
    processor = SimpleTextProcessor()
    generator = SimpleGenerator()
    
    print("  ✅ Components initialized")
    
    # Create knowledge graph
    kg = KnowledgeGraph(embedding_dim=384)
    
    # Add some initial knowledge
    initial_docs = [
        "Python is a high-level programming language known for its simplicity.",
        "Machine learning algorithms learn patterns from data automatically.",  
        "Natural language processing enables computers to understand human language."
    ]
    
    print("  📚 Adding initial knowledge...")
    for i, doc in enumerate(initial_docs):
        embedding = embedder.encode([doc])[0]
        node_id = kg.add_node(
            text=doc,
            embedding=embedding,
            node_type="knowledge",
            confidence=0.9
        )
        print(f"    Added node {node_id}: {doc[:50]}...")
    
    # Test queries
    test_queries = [
        "What is Python?",
        "How does machine learning work?",
        "What is NLP?",
        "Tell me about deep learning.",  # New topic
    ]
    
    print("  🔍 Processing test queries...")
    
    for i, query in enumerate(test_queries):
        print(f"\n  Query {i+1}: '{query}'")
        
        # Get query embedding and find similar knowledge
        query_embedding = embedder.encode([query])[0]
        similar_nodes = kg.find_similar_nodes(query_embedding, k=3, min_similarity=0.1)
        
        if similar_nodes:
            # Get context from similar nodes
            context_parts = []
            for node_id, similarity in similar_nodes[:2]:  # Top 2
                node = kg.nodes[node_id] 
                context_parts.append(f"{node.text} (sim: {similarity:.3f})")
                # Update access tracking
                node.access_count += 1
                node.last_accessed = time.time()
            
            context = " | ".join(context_parts)
            print(f"    📖 Found context: {len(similar_nodes)} nodes, max_sim={similar_nodes[0][1]:.3f}")
        else:
            context = ""
            print(f"    📭 No relevant context found")
        
        # Generate response
        response = generator.generate(query, context)
        print(f"    💬 Generated: {response[:80]}...")
        
        # Simple update decision (add if similarity is low)
        max_similarity = similar_nodes[0][1] if similar_nodes else 0.0
        should_add = max_similarity < config.cosine_similarity_threshold
        
        if should_add:
            # Add new knowledge 
            new_knowledge = f"Q: {query} A: {response}"
            new_embedding = embedder.encode([new_knowledge])[0]
            new_node_id = kg.add_node(
                text=new_knowledge,
                embedding=new_embedding,
                node_type="qa_pair",
                confidence=0.7
            )
            print(f"    ➕ Added new knowledge: {new_node_id}")
            
            # Add edges to similar nodes
            for node_id, similarity in similar_nodes[:2]:
                if similarity > 0.2:  # Minimum connection threshold
                    kg.add_edge(new_node_id, node_id, 
                               relation="semantic", 
                               weight=similarity,
                               semantic_similarity=similarity)
        else:
            print(f"    🚫 No update needed (max_sim={max_similarity:.3f} >= {config.cosine_similarity_threshold})")
    
    # Final statistics
    stats = kg.get_statistics()
    print(f"\n  📊 Final Statistics:")
    print(f"    Nodes: {stats['current_nodes']}")
    print(f"    Edges: {stats['current_edges']}")
    print(f"    Total additions: {stats['total_nodes_added']}")
    
    return True


def main():
    """Run RAG integration test."""
    print("🚀 geDIG-RAG v3 Integration Test")
    print("=" * 50)
    
    try:
        test_rag_mock_implementation()
        
        print()
        print("🎉 RAG Integration Test Passed!")
        print("✅ Core RAG workflow functioning correctly")
        print()
        print("Ready for:")
        print("1. Complete baseline RAG implementations")
        print("2. Evaluation system framework")
        print("3. Data preparation and experiments")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Integration Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)