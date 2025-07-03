#!/usr/bin/env python3
"""
Baseline RAG System for Comparison
Standard FAISS + Sentence Transformers implementation
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

import faiss
from sentence_transformers import SentenceTransformer

class BaselineRAG:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.encoder = SentenceTransformer(model_name)
        self.dimension = 384
        self.index = None
        self.documents = []
        self.embeddings = []
        
    def initialize(self):
        """Initialize FAISS index"""
        # Use simple flat index for fair comparison
        self.index = faiss.IndexFlatL2(self.dimension)
        print(f"Initialized baseline RAG with dimension {self.dimension}")
    
    def add_documents(self, texts: List[str]) -> Dict[str, Any]:
        """Add documents to the RAG system"""
        start_time = time.time()
        
        # Encode texts
        new_embeddings = self.encoder.encode(texts, show_progress_bar=False)
        
        # Add to index
        self.index.add(new_embeddings.astype('float32'))
        
        # Store documents
        for text in texts:
            self.documents.append({
                'text': text,
                'id': len(self.documents)
            })
        
        # Store embeddings
        self.embeddings.extend(new_embeddings)
        
        processing_time = time.time() - start_time
        
        return {
            'added': len(texts),
            'total_documents': len(self.documents),
            'processing_time': processing_time
        }
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents"""
        start_time = time.time()
        
        # Encode query
        query_embedding = self.encoder.encode([query], show_progress_bar=False)
        
        # Search
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Prepare results
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.documents):
                results.append({
                    'text': self.documents[idx]['text'],
                    'score': float(1 / (1 + dist)),  # Convert distance to similarity
                    'index': int(idx)
                })
        
        search_time = time.time() - start_time
        
        return {
            'results': results,
            'search_time': search_time
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            'total_documents': len(self.documents),
            'index_size': self.index.ntotal if self.index else 0,
            'memory_usage': self._estimate_memory_usage()
        }
    
    def save(self, path: str = 'baseline_rag_data'):
        """Save the RAG system"""
        Path(path).mkdir(exist_ok=True)
        
        # Save index
        faiss.write_index(self.index, f"{path}/index.faiss")
        
        # Save documents
        with open(f"{path}/documents.json", 'w') as f:
            json.dump(self.documents, f)
        
        # Save embeddings
        np.save(f"{path}/embeddings.npy", np.array(self.embeddings))
        
        print(f"Saved baseline RAG to {path}")
    
    def load(self, path: str = 'baseline_rag_data'):
        """Load the RAG system"""
        # Load index
        self.index = faiss.read_index(f"{path}/index.faiss")
        
        # Load documents
        with open(f"{path}/documents.json", 'r') as f:
            self.documents = json.load(f)
        
        # Load embeddings
        self.embeddings = np.load(f"{path}/embeddings.npy").tolist()
        
        print(f"Loaded baseline RAG from {path}")
    
    def _estimate_memory_usage(self) -> int:
        """Estimate memory usage in bytes"""
        # Index memory
        index_memory = len(self.embeddings) * self.dimension * 4  # float32
        
        # Document memory (rough estimate)
        doc_memory = sum(len(doc['text'].encode('utf-8')) for doc in self.documents)
        
        # Embedding list memory
        embedding_memory = len(self.embeddings) * self.dimension * 8  # Python float
        
        return index_memory + doc_memory + embedding_memory


def run_baseline_comparison(dataset: List[str]):
    """Run comparison experiment with baseline RAG"""
    print("\n=== Baseline RAG Comparison ===")
    
    # Initialize baseline
    baseline = BaselineRAG()
    baseline.initialize()
    
    # Add documents in batches
    batch_size = 100
    total_time = 0
    
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        result = baseline.add_documents(batch)
        total_time += result['processing_time']
        
        if (i + batch_size) % 500 == 0:
            print(f"Added {i + batch_size} documents...")
    
    # Get final stats
    stats = baseline.get_stats()
    
    print(f"\nBaseline RAG Results:")
    print(f"  Total documents: {stats['total_documents']}")
    print(f"  Total processing time: {total_time:.2f}s")
    print(f"  Average time per document: {total_time/len(dataset)*1000:.2f}ms")
    print(f"  Estimated memory usage: {stats['memory_usage']/1024/1024:.2f} MB")
    
    # Test search performance
    test_queries = [
        "artificial intelligence in healthcare",
        "machine learning algorithms",
        "quantum computing applications",
        "blockchain technology benefits",
        "deep learning neural networks"
    ]
    
    total_search_time = 0
    for query in test_queries:
        result = baseline.search(query, k=5)
        total_search_time += result['search_time']
        print(f"\nQuery: '{query}'")
        print(f"  Search time: {result['search_time']*1000:.2f}ms")
        print(f"  Top result: {result['results'][0]['text'][:50]}...")
    
    avg_search_time = total_search_time / len(test_queries)
    print(f"\nAverage search time: {avg_search_time*1000:.2f}ms")
    
    # Save for later analysis
    baseline.save('experiment_4/baseline_rag_data')
    
    return {
        'total_documents': stats['total_documents'],
        'processing_time': total_time,
        'memory_usage': stats['memory_usage'],
        'avg_search_time': avg_search_time
    }


if __name__ == "__main__":
    # Generate test dataset
    from dynamic_graph_growth_experiment import DynamicGraphExperiment
    
    exp = DynamicGraphExperiment()
    dataset = exp.generate_dataset(1000)
    
    # Run baseline comparison
    results = run_baseline_comparison(dataset)