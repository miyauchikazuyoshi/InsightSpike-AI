#!/usr/bin/env python3
"""Analyze similarity between knowledge base and queries."""

from run_experiment_improved import (
    ImprovedEmbedder,
    create_high_quality_knowledge_base,
    create_meaningful_queries
)
import numpy as np

def analyze_similarities():
    """Analyze similarity distribution."""
    print("🔍 Analyzing Similarity Distribution")
    print("=" * 60)
    
    # Setup
    embedder = ImprovedEmbedder()
    knowledge_base = create_high_quality_knowledge_base()
    test_queries = create_meaningful_queries()
    
    # Encode knowledge base
    kb_texts = [kb.text for kb in knowledge_base]
    kb_embeddings = embedder.encode(kb_texts)
    
    # Analyze each query
    similarities = []
    
    for i, (query, depth) in enumerate(test_queries):
        query_embedding = embedder.encode(query)[0]
        
        # Calculate similarities with all KB items
        query_sims = []
        for kb_emb in kb_embeddings:
            sim = np.dot(query_embedding, kb_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(kb_emb)
            )
            query_sims.append(sim)
        
        max_sim = max(query_sims)
        similarities.append(max_sim)
        
        if i < 5:  # Show first 5
            print(f"\nQuery {i+1}: {query[:50]}...")
            print(f"  Max similarity: {max_sim:.3f}")
            print(f"  Novelty: {1-max_sim:.3f}")
            best_match_idx = query_sims.index(max_sim)
            print(f"  Best match: {kb_texts[best_match_idx][:50]}...")
    
    # Statistics
    print("\n" + "-" * 60)
    print("Similarity Statistics:")
    print(f"  Mean: {np.mean(similarities):.3f}")
    print(f"  Median: {np.median(similarities):.3f}")
    print(f"  Min: {np.min(similarities):.3f}")
    print(f"  Max: {np.max(similarities):.3f}")
    
    # Distribution
    high_novelty = sum(1 for s in similarities if s < 0.3)
    moderate_novelty = sum(1 for s in similarities if 0.3 <= s < 0.6)
    low_novelty = sum(1 for s in similarities if s >= 0.6)
    
    print(f"\nNovelty Distribution:")
    print(f"  High (sim < 0.3): {high_novelty}/{len(similarities)} ({high_novelty/len(similarities)*100:.0f}%)")
    print(f"  Moderate (0.3-0.6): {moderate_novelty}/{len(similarities)} ({moderate_novelty/len(similarities)*100:.0f}%)")
    print(f"  Low (sim >= 0.6): {low_novelty}/{len(similarities)} ({low_novelty/len(similarities)*100:.0f}%)")
    
    print("\n💡 Insight:")
    if np.mean(similarities) < 0.2:
        print("  Queries have very low similarity to knowledge base.")
        print("  This explains why all queries appear highly novel.")
        print("  Solution: Use higher positive thresholds or create more overlap.")

if __name__ == "__main__":
    analyze_similarities()