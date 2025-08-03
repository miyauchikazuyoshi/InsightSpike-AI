"""
Detailed analysis of the Apple query results
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


def analyze_apple_query():
    """Analyze how distance vs cosine handles ambiguous queries."""
    
    # Initialize model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Corpus focused on Apple ambiguity
    corpus = [
        # Apple company cluster
        "Apple Inc. makes innovative products.",
        "The iPhone is Apple's flagship device.", 
        "MacBook is a laptop made by Apple.",
        "Steve Jobs founded Apple Computer.",
        
        # Apple fruit cluster
        "I love eating fresh apples.",
        "Apple pie is a delicious dessert.",
        "Green apples are sour.",
        "An apple a day keeps the doctor away.",
        
        # Mixed/ambiguous
        "Apple's market share is growing.",  # Could be company or fruit market
        "The apple logo is iconic.",         # Company, but mentions fruit word
        
        # Control (unrelated)
        "Python is a programming language.",
        "The weather is nice today.",
    ]
    
    # Encode
    embeddings = model.encode(corpus)
    
    # Query
    query = "Apple products and technology"
    query_vec = model.encode([query])[0]
    
    # Calculate both metrics
    distances = np.linalg.norm(embeddings - query_vec, axis=1)
    cos_sims = cosine_similarity([query_vec], embeddings)[0]
    
    # Convert cosine to cosine distance for comparison
    cos_distances = 1 - cos_sims
    
    # Sort by distance
    dist_order = np.argsort(distances)
    
    print("DETAILED APPLE QUERY ANALYSIS")
    print("=" * 70)
    print(f"Query: '{query}'")
    print("\n" + "-" * 70)
    print(f"{'Rank':<5} {'Euclidean':<12} {'Cosine Sim':<12} {'Cos Dist':<12} {'Text'}")
    print("-" * 70)
    
    for i, idx in enumerate(dist_order):
        print(f"{i+1:<5} {distances[idx]:<12.3f} {cos_sims[idx]:<12.3f} "
              f"{cos_distances[idx]:<12.3f} {corpus[idx][:40]}...")
    
    # Analyze gaps
    print("\n" + "=" * 70)
    print("GAP ANALYSIS (between consecutive results)")
    print("=" * 70)
    
    print(f"\n{'Between':<30} {'Euclidean Gap':<15} {'Cosine Gap':<15} {'Ratio'}")
    print("-" * 70)
    
    for i in range(len(dist_order)-1):
        idx1, idx2 = dist_order[i], dist_order[i+1]
        
        # Calculate gaps
        dist_gap = distances[idx2] - distances[idx1]
        cos_gap = cos_sims[idx1] - cos_sims[idx2]  # Note: reversed because similarity
        
        # Ratio of gaps (how much more sensitive distance is)
        ratio = dist_gap / cos_gap if cos_gap > 0.001 else float('inf')
        
        between = f"{i+1} → {i+2}"
        print(f"{between:<30} {dist_gap:<15.4f} {cos_gap:<15.4f} {ratio:<.2f}x")
    
    # Find semantic boundaries
    print("\n" + "=" * 70)
    print("SEMANTIC BOUNDARIES")
    print("=" * 70)
    
    # Group by semantic category
    company_indices = [0, 1, 2, 3, 9]
    fruit_indices = [4, 5, 6, 7]
    
    print("\nAverage distances by category:")
    company_dists = [distances[i] for i in company_indices if i < len(distances)]
    fruit_dists = [distances[i] for i in fruit_indices]
    
    print(f"  Company-related: {np.mean(company_dists):.3f} (±{np.std(company_dists):.3f})")
    print(f"  Fruit-related:   {np.mean(fruit_dists):.3f} (±{np.std(fruit_dists):.3f})")
    print(f"  Gap between categories: {np.mean(fruit_dists) - np.mean(company_dists):.3f}")
    
    print("\nAverage cosine similarities by category:")
    company_cos = [cos_sims[i] for i in company_indices if i < len(cos_sims)]
    fruit_cos = [cos_sims[i] for i in fruit_indices]
    
    print(f"  Company-related: {np.mean(company_cos):.3f} (±{np.std(company_cos):.3f})")
    print(f"  Fruit-related:   {np.mean(fruit_cos):.3f} (±{np.std(fruit_cos):.3f})")
    print(f"  Gap between categories: {np.mean(company_cos) - np.mean(fruit_cos):.3f}")
    
    # Visualize
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Distance distribution
    plt.subplot(1, 2, 1)
    x_pos = range(len(dist_order))
    colors = ['blue' if idx in company_indices else 'orange' if idx in fruit_indices else 'gray' 
              for idx in dist_order]
    
    plt.bar(x_pos, [distances[idx] for idx in dist_order], color=colors, alpha=0.7)
    plt.xlabel('Rank')
    plt.ylabel('Euclidean Distance')
    plt.title('Distance Distribution')
    plt.axhline(y=1.0, color='red', linestyle='--', label='Potential boundary')
    plt.legend()
    
    # Plot 2: Cosine similarity distribution
    plt.subplot(1, 2, 2)
    plt.bar(x_pos, [cos_sims[idx] for idx in dist_order], color=colors, alpha=0.7)
    plt.xlabel('Rank')
    plt.ylabel('Cosine Similarity')
    plt.title('Cosine Similarity Distribution')
    plt.axhline(y=0.4, color='red', linestyle='--', label='Potential boundary')
    plt.legend()
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', alpha=0.7, label='Company'),
        Patch(facecolor='orange', alpha=0.7, label='Fruit'),
        Patch(facecolor='gray', alpha=0.7, label='Other')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('apple_query_analysis.png', dpi=150)
    print("\nSaved visualization to 'apple_query_analysis.png'")
    
    # Test different thresholds
    print("\n" + "=" * 70)
    print("THRESHOLD TESTING")
    print("=" * 70)
    
    print("\nIf we set different thresholds:")
    
    # Distance thresholds
    for threshold in [0.8, 1.0, 1.2]:
        selected = distances < threshold
        company_selected = sum(1 for i in company_indices if i < len(distances) and selected[i])
        fruit_selected = sum(1 for i in fruit_indices if selected[i])
        
        print(f"\nDistance < {threshold}:")
        print(f"  Company: {company_selected}/5, Fruit: {fruit_selected}/4")
        print(f"  Precision: {company_selected/(company_selected+fruit_selected):.2f}" if (company_selected+fruit_selected) > 0 else "  No results")
    
    # Cosine thresholds
    for threshold in [0.6, 0.4, 0.3]:
        selected = cos_sims > threshold
        company_selected = sum(1 for i in company_indices if i < len(cos_sims) and selected[i])
        fruit_selected = sum(1 for i in fruit_indices if selected[i])
        
        print(f"\nCosine > {threshold}:")
        print(f"  Company: {company_selected}/5, Fruit: {fruit_selected}/4")
        print(f"  Precision: {company_selected/(company_selected+fruit_selected):.2f}" if (company_selected+fruit_selected) > 0 else "  No results")


if __name__ == "__main__":
    analyze_apple_query()