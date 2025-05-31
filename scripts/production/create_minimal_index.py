#!/usr/bin/env python3
"""
Quick FAISS Index Creator for InsightSpike-AI
===========================================

Creates a minimal FAISS index for testing purposes.
"""

import os
import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    
    def create_minimal_faiss_index():
        """Create a minimal FAISS index for testing"""
        print("🔧 Creating minimal FAISS index...")
        
        # Initialize sentence transformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create some test sentences
        test_sentences = [
            "The weather is nice today.",
            "Machine learning is fascinating.",
            "Python is a powerful programming language.",
            "Artificial intelligence will transform the future.",
            "Data science requires statistical knowledge."
        ]
        
        # Generate embeddings
        print("📝 Generating embeddings...")
        embeddings = model.encode(test_sentences)
        embeddings = np.array(embeddings).astype('float32')
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        
        # Save the index
        index_path = project_root / "data" / "index.faiss"
        faiss.write_index(index, str(index_path))
        
        print(f"✅ FAISS index created with {len(test_sentences)} documents")
        print(f"📁 Saved to: {index_path}")
        print(f"📊 Dimension: {dimension}")
        
        return True
        
    if __name__ == "__main__":
        success = create_minimal_faiss_index()
        if success:
            print("🎉 Index creation completed successfully!")
        else:
            print("❌ Index creation failed!")
            sys.exit(1)
            
except ImportError as e:
    print(f"❌ Required dependencies not available: {e}")
    print("Please run: pip install faiss-cpu sentence-transformers")
    sys.exit(1)
