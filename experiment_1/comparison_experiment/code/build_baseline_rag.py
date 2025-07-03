#!/usr/bin/env python3
"""
Build Baseline RAG System for Comparison
Implements traditional RAG using FAISS and sentence-transformers
"""

import os
import json
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

import faiss
from sentence_transformers import SentenceTransformer
import torch

class BaselineRAG:
    """Traditional RAG system for comparison"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.encoder = SentenceTransformer(model_name)
        self.dimension = 384  # all-MiniLM-L6-v2 dimension
        self.index = None
        self.documents = []
        self.metadata = []
        
    def build_index(self, documents: List[str], metadata: List[Dict] = None):
        """Build FAISS index from documents"""
        print(f"Building index for {len(documents)} documents...")
        
        # Encode documents
        start_time = time.time()
        embeddings = self.encoder.encode(documents, show_progress_bar=True)
        encode_time = time.time() - start_time
        
        # Create FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings.astype('float32'))
        
        # Store documents and metadata
        self.documents = documents
        self.metadata = metadata or [{} for _ in documents]
        
        index_time = time.time() - start_time - encode_time
        
        return {
            'num_documents': len(documents),
            'encode_time': encode_time,
            'index_time': index_time,
            'total_time': encode_time + index_time,
            'index_size': self.get_index_size()
        }
    
    def search(self, query: str, k: int = 5) -> List[Tuple[str, float, Dict]]:
        """Search for relevant documents"""
        if self.index is None:
            return []
        
        # Encode query
        start_time = time.time()
        query_embedding = self.encoder.encode([query])
        encode_time = time.time() - start_time
        
        # Search
        search_start = time.time()
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        search_time = time.time() - search_start
        
        # Prepare results
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.documents):
                results.append((
                    self.documents[idx],
                    float(dist),
                    self.metadata[idx]
                ))
        
        return results, {
            'encode_time': encode_time,
            'search_time': search_time,
            'total_time': encode_time + search_time,
            'num_results': len(results)
        }
    
    def get_index_size(self) -> int:
        """Estimate index size in bytes"""
        if self.index is None:
            return 0
        
        # Estimate size: embeddings + overhead
        num_vectors = self.index.ntotal
        size = num_vectors * self.dimension * 4  # float32
        
        # Add document storage estimate
        doc_size = sum(len(doc.encode('utf-8')) for doc in self.documents)
        metadata_size = len(json.dumps(self.metadata).encode('utf-8'))
        
        return size + doc_size + metadata_size
    
    def save(self, path: str):
        """Save index and data"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        if self.index:
            faiss.write_index(self.index, str(path / "baseline_index.faiss"))
        
        # Save documents and metadata
        data = {
            'documents': self.documents,
            'metadata': self.metadata,
            'dimension': self.dimension
        }
        
        with open(path / "baseline_data.json", 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: str):
        """Load index and data"""
        path = Path(path)
        
        # Load FAISS index
        index_path = path / "baseline_index.faiss"
        if index_path.exists():
            self.index = faiss.read_index(str(index_path))
        
        # Load documents and metadata
        data_path = path / "baseline_data.json"
        if data_path.exists():
            with open(data_path, 'r') as f:
                data = json.load(f)
                self.documents = data['documents']
                self.metadata = data['metadata']
                self.dimension = data['dimension']


class HybridBaselineRAG(BaselineRAG):
    """Enhanced baseline with BM25 + dense retrieval"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        super().__init__(model_name)
        from rank_bm25 import BM25Okapi
        self.bm25 = None
        self.tokenized_docs = []
    
    def build_index(self, documents: List[str], metadata: List[Dict] = None):
        """Build both dense and sparse indices"""
        # Build dense index
        dense_stats = super().build_index(documents, metadata)
        
        # Build BM25 index
        from rank_bm25 import BM25Okapi
        start_time = time.time()
        self.tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)
        bm25_time = time.time() - start_time
        
        dense_stats['bm25_time'] = bm25_time
        dense_stats['total_time'] += bm25_time
        
        return dense_stats
    
    def search(self, query: str, k: int = 5, alpha: float = 0.5) -> List[Tuple[str, float, Dict]]:
        """Hybrid search combining dense and sparse retrieval"""
        if self.index is None:
            return [], {}
        
        # Dense search
        dense_results, dense_stats = super().search(query, k=k*2)
        
        # BM25 search
        start_time = time.time()
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(bm25_scores)[-k*2:][::-1]
        bm25_time = time.time() - start_time
        
        # Combine scores
        doc_scores = {}
        
        # Add dense scores
        for doc, dist, meta in dense_results:
            # Convert distance to similarity score
            score = 1 / (1 + dist)
            doc_scores[doc] = alpha * score
        
        # Add BM25 scores
        for idx in top_indices:
            if idx < len(self.documents):
                doc = self.documents[idx]
                if doc in doc_scores:
                    doc_scores[doc] += (1 - alpha) * bm25_scores[idx]
                else:
                    doc_scores[doc] = (1 - alpha) * bm25_scores[idx]
        
        # Sort by combined score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        # Prepare results
        results = []
        for doc, score in sorted_docs:
            idx = self.documents.index(doc)
            results.append((doc, score, self.metadata[idx]))
        
        stats = {
            'dense_stats': dense_stats,
            'bm25_time': bm25_time,
            'total_time': dense_stats['total_time'] + bm25_time,
            'num_results': len(results)
        }
        
        return results, stats


def prepare_test_data():
    """Prepare test data from existing datasets"""
    test_documents = []
    test_metadata = []
    
    # Load from HuggingFace datasets if available
    dataset_path = Path("experiments/gedig_embedding_evaluation/data/huggingface_datasets")
    
    if dataset_path.exists():
        # Load MS MARCO data
        marco_path = dataset_path / "ms_marco_20"
        if marco_path.exists():
            try:
                import pyarrow.parquet as pq
                # Simple loading for demonstration
                arrow_files = list(marco_path.glob("*.arrow"))
                if arrow_files:
                    print(f"Found {len(arrow_files)} MS MARCO files")
                    # Would load actual data here
            except ImportError:
                pass
    
    # If no real data, create synthetic test data
    if not test_documents:
        print("Creating synthetic test data...")
        for i in range(100):
            test_documents.append(
                f"Document {i}: This is test content about topic {i % 10}. "
                f"It contains information about subject {i % 5} and concept {i % 3}."
            )
            test_metadata.append({
                'id': i,
                'topic': i % 10,
                'type': 'synthetic'
            })
    
    return test_documents, test_metadata


def main():
    """Build and test baseline RAG systems"""
    print("Building Baseline RAG Systems for Comparison")
    
    # Prepare test data
    documents, metadata = prepare_test_data()
    print(f"Loaded {len(documents)} documents")
    
    # Build standard baseline
    print("\n=== Building Standard Baseline RAG ===")
    baseline_rag = BaselineRAG()
    build_stats = baseline_rag.build_index(documents, metadata)
    print(f"Build stats: {build_stats}")
    
    # Test search
    test_queries = [
        "What is topic 5?",
        "Information about subject 2",
        "Tell me about concept 1"
    ]
    
    for query in test_queries:
        results, stats = baseline_rag.search(query, k=5)
        print(f"\nQuery: {query}")
        print(f"Search time: {stats['total_time']:.3f}s")
        print(f"Top result: {results[0][0][:50]}..." if results else "No results")
    
    # Save baseline
    baseline_rag.save("experiment_1/comparison_experiment/data/baseline_rag")
    
    # Build hybrid baseline
    print("\n=== Building Hybrid Baseline RAG ===")
    hybrid_rag = HybridBaselineRAG()
    hybrid_stats = hybrid_rag.build_index(documents, metadata)
    print(f"Build stats: {hybrid_stats}")
    
    # Save hybrid baseline
    hybrid_rag.save("experiment_1/comparison_experiment/data/hybrid_baseline_rag")
    
    # Save build statistics
    stats_file = Path("experiment_1/comparison_experiment/results/baseline_build_stats.json")
    stats_file.parent.mkdir(exist_ok=True)
    
    with open(stats_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'num_documents': len(documents),
            'standard_baseline': build_stats,
            'hybrid_baseline': hybrid_stats
        }, f, indent=2)
    
    print(f"\nBaseline systems built and saved!")


if __name__ == "__main__":
    main()