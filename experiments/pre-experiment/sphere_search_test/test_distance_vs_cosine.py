"""
Test distance-based search vs cosine similarity with Sentence-BERT
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple


def test_search_methods():
    """Compare distance-based and cosine similarity search."""
    
    # Initialize Sentence-BERT
    print("Loading Sentence-BERT model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Test corpus - varied topics
    corpus = [
        # Technology cluster
        "Python is a popular programming language.",
        "JavaScript is used for web development.",
        "Machine learning requires programming skills.",
        "AI is transforming software development.",
        
        # Food cluster  
        "I love eating fresh apples.",
        "Oranges are rich in vitamin C.",
        "Fruits are healthy snacks.",
        "Apple pie is a delicious dessert.",
        
        # Mixed (Apple company)
        "Apple Inc. makes innovative products.",
        "The iPhone is Apple's flagship device.",
        
        # Random
        "The weather is nice today.",
        "Cats are independent animals.",
        "Books expand our knowledge.",
        "Exercise is good for health.",
    ]
    
    # Encode all sentences
    print("\nEncoding sentences...")
    embeddings = model.encode(corpus)
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Test queries
    queries = [
        "Tell me about programming languages",  # Should find tech cluster
        "What fruits should I eat?",           # Should find food cluster  
        "Apple products and technology",       # Ambiguous - company or fruit?
    ]
    
    print("\n" + "="*60)
    print("SEARCH COMPARISON")
    print("="*60)
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        print("-" * 40)
        
        # Encode query
        query_vec = model.encode([query])[0]
        query_vec_norm = query_vec / np.linalg.norm(query_vec)
        
        # Method 1: Euclidean distance
        distances = np.linalg.norm(embeddings - query_vec, axis=1)
        dist_indices = np.argsort(distances)[:5]
        
        # Method 2: Cosine similarity
        cos_sims = cosine_similarity([query_vec], embeddings)[0]
        cos_indices = np.argsort(cos_sims)[::-1][:5]
        
        # Method 3: Distance on normalized vectors (sphere)
        sphere_distances = np.linalg.norm(embeddings_norm - query_vec_norm, axis=1)
        sphere_indices = np.argsort(sphere_distances)[:5]
        
        # Display results
        print("\n1. Euclidean Distance (smaller = better):")
        for i, idx in enumerate(dist_indices):
            print(f"   {i+1}. [{distances[idx]:.3f}] {corpus[idx]}")
        
        print("\n2. Cosine Similarity (larger = better):")
        for i, idx in enumerate(cos_indices):
            print(f"   {i+1}. [{cos_sims[idx]:.3f}] {corpus[idx]}")
        
        print("\n3. Sphere Distance (normalized, smaller = better):")
        for i, idx in enumerate(sphere_indices):
            print(f"   {i+1}. [{sphere_distances[idx]:.3f}] {corpus[idx]}")
        
        # Check if results differ
        if not np.array_equal(dist_indices, cos_indices):
            print("\n⚠️  Distance and Cosine give different rankings!")
        
        if not np.array_equal(sphere_indices, cos_indices):
            print("⚠️  Sphere distance and Cosine give different rankings!")
    
    # Analyze vector properties
    print("\n" + "="*60)
    print("VECTOR ANALYSIS")
    print("="*60)
    
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"\nVector norms (before normalization):")
    print(f"  Min: {norms.min():.3f}")
    print(f"  Max: {norms.max():.3f}")
    print(f"  Mean: {norms.mean():.3f}")
    print(f"  Std: {norms.std():.3f}")
    
    # Visualize distance distribution
    print("\nComputing pairwise distances...")
    n = len(embeddings)
    
    # Sample pairs for visualization
    sample_size = min(100, n*(n-1)//2)
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            pairs.append((i, j))
    
    if len(pairs) > sample_size:
        import random
        pairs = random.sample(pairs, sample_size)
    
    euclidean_dists = []
    cosine_sims = []
    sphere_dists = []
    
    for i, j in pairs:
        # Euclidean
        euclidean_dists.append(np.linalg.norm(embeddings[i] - embeddings[j]))
        
        # Cosine
        cos_sim = np.dot(embeddings[i], embeddings[j]) / (norms[i] * norms[j])
        cosine_sims.append(cos_sim)
        
        # Sphere (normalized)
        sphere_dists.append(np.linalg.norm(embeddings_norm[i] - embeddings_norm[j]))
    
    # Plot distributions
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(euclidean_dists, bins=30, alpha=0.7, color='blue')
    plt.title('Euclidean Distance Distribution')
    plt.xlabel('Distance')
    plt.ylabel('Count')
    
    plt.subplot(1, 3, 2)
    plt.hist(cosine_sims, bins=30, alpha=0.7, color='green')
    plt.title('Cosine Similarity Distribution')
    plt.xlabel('Similarity')
    plt.ylabel('Count')
    
    plt.subplot(1, 3, 3)
    plt.hist(sphere_dists, bins=30, alpha=0.7, color='red')
    plt.title('Sphere Distance Distribution (Normalized)')
    plt.xlabel('Distance')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('distance_comparison.png', dpi=150)
    print("\nSaved distribution plot to 'distance_comparison.png'")
    
    # Test intuitive radius search
    print("\n" + "="*60)
    print("INTUITIVE RADIUS TEST")
    print("="*60)
    
    query = "programming and coding"
    query_vec = model.encode([query])[0]
    query_vec_norm = query_vec / np.linalg.norm(query_vec)
    
    # Different intuitive radii (3D-like)
    radii = [0.3, 0.5, 0.7]
    dimension = embeddings.shape[1]  # 384 for MiniLM
    
    print(f"\nQuery: '{query}'")
    print(f"Embedding dimension: {dimension}")
    
    for intuitive_r in radii:
        # Convert to actual radius for this dimension
        volume_fraction = intuitive_r ** 3
        actual_radius = np.power(volume_fraction, 1.0 / dimension)
        
        # Find points within radius
        sphere_distances = np.linalg.norm(embeddings_norm - query_vec_norm, axis=1)
        within_radius = sphere_distances < actual_radius
        
        print(f"\nIntuitive radius: {intuitive_r} (volume: {volume_fraction:.1%})")
        print(f"Actual radius in {dimension}D: {actual_radius:.6f}")
        print(f"Points found: {within_radius.sum()}")
        
        if within_radius.sum() > 0:
            indices = np.where(within_radius)[0]
            sorted_indices = indices[np.argsort(sphere_distances[indices])][:5]
            for idx in sorted_indices:
                print(f"  - [{sphere_distances[idx]:.3f}] {corpus[idx]}")


if __name__ == "__main__":
    test_search_methods()