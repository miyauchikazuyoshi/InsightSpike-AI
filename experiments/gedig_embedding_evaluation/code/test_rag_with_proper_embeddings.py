#!/usr/bin/env python3
"""
Test RAG Performance with Proper Embeddings
==========================================

Tests the regenerated data structure with SentenceTransformer embeddings.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'src'))

from insightspike.core.agents.main_agent import MainAgent
from insightspike.core.layers.layer2_memory_manager import L2MemoryManager

import numpy as np
import json
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Test RAG with properly embedded data"""
    print("="*60)
    print("Testing RAG with Proper Embeddings")
    print("="*60)
    
    # Check data state
    data_dir = Path("data")
    print("\nData state:")
    for filename in ["episodes.json", "index.faiss", "graph_pyg.pt"]:
        filepath = data_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size
            print(f"  {filename}: {size:,} bytes")
            if filename == "episodes.json":
                with open(filepath, 'r') as f:
                    episodes = json.load(f)
                print(f"    Episodes: {len(episodes)}")
                # Check first episode
                if episodes:
                    first_ep = episodes[0]
                    print(f"    First episode text: '{first_ep['text'][:50]}...'")
                    print(f"    Vector length: {len(first_ep['vec'])}")
    
    # Initialize agent
    print("\nInitializing agent...")
    agent = MainAgent()
    if not agent.initialize():
        logger.error("Failed to initialize agent")
        return
    
    # Test queries
    test_queries = [
        ("What is machine learning?", ["machine", "learning", "data", "systems"]),
        ("How does deep learning work?", ["deep", "learning", "layers", "neural"]),
        ("Explain neural networks", ["neural", "networks", "brain", "biological"]),
        ("What are the applications of computer vision?", ["computer", "vision", "applications", "pattern"]),
        ("Tell me about reinforcement learning", ["reinforcement", "learning", "trial", "error"]),
        ("What is data science?", ["data", "science", "insights", "statistical"]),
        ("How do algorithms work?", ["algorithms", "computational", "efficiency", "models"]),
        ("Explain artificial intelligence", ["artificial", "intelligence", "systems", "learn"])
    ]
    
    print(f"\nTesting {len(test_queries)} queries...")
    results = []
    
    for query, expected_keywords in test_queries:
        print(f"\nQuery: '{query}'")
        
        # Process query
        result = agent.process_question(query, max_cycles=2)
        
        # Extract metrics
        quality = result.get('reasoning_quality', 0)
        documents = result.get('documents', [])
        response = result.get('response', '')
        
        print(f"  Quality score: {quality:.3f}")
        print(f"  Documents retrieved: {len(documents)}")
        
        if documents:
            # Check document relevance
            relevant_count = 0
            for doc in documents[:5]:  # Check top 5
                doc_text = doc['text'].lower()
                if any(kw in doc_text for kw in expected_keywords):
                    relevant_count += 1
                print(f"    - Score {doc['similarity']:.3f}: {doc['text'][:60]}...")
            
            relevance_rate = relevant_count / min(5, len(documents))
            print(f"  Relevance rate (top 5): {relevance_rate:.1%}")
        else:
            relevance_rate = 0
        
        # Check response relevance
        response_lower = response.lower()
        response_has_keywords = sum(1 for kw in expected_keywords if kw in response_lower)
        print(f"  Response keywords found: {response_has_keywords}/{len(expected_keywords)}")
        
        results.append({
            'query': query,
            'quality': quality,
            'doc_count': len(documents),
            'relevance_rate': relevance_rate,
            'response_keywords': response_has_keywords / len(expected_keywords)
        })
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    avg_quality = np.mean([r['quality'] for r in results])
    avg_docs = np.mean([r['doc_count'] for r in results])
    avg_relevance = np.mean([r['relevance_rate'] for r in results])
    avg_response_keywords = np.mean([r['response_keywords'] for r in results])
    
    print(f"Average quality score: {avg_quality:.3f}")
    print(f"Average documents retrieved: {avg_docs:.1f}")
    print(f"Average relevance rate: {avg_relevance:.1%}")
    print(f"Average response keyword match: {avg_response_keywords:.1%}")
    
    # Compare with hash-based (expected values)
    print("\nComparison with hash-based embeddings:")
    print("  Hash-based relevance: ~4%")
    print(f"  Semantic embeddings relevance: {avg_relevance:.1%}")
    print(f"  Improvement: {(avg_relevance - 0.04) / 0.04 * 100:.0f}%")
    
    print("\nTest complete!")

if __name__ == "__main__":
    main()