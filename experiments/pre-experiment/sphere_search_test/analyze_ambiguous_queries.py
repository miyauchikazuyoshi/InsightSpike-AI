"""
Test distance vs cosine with more ambiguous queries
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


def analyze_ambiguous_queries():
    """Test various ambiguous queries to see boundary clarity."""
    
    # Initialize model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Expanded corpus with ambiguous terms
    corpus = [
        # Python (programming)
        "Python is a versatile programming language.",
        "I write Python scripts for automation.",
        "Python has excellent data science libraries.",
        "Django is a Python web framework.",
        
        # Python (snake)
        "A python is a large constrictor snake.",
        "Pythons are found in tropical regions.",
        "The ball python is a popular pet snake.",
        "Pythons can grow over 20 feet long.",
        
        # Java (programming)
        "Java is an object-oriented programming language.",
        "Spring Boot simplifies Java development.",
        "Java runs on billions of devices worldwide.",
        "I'm learning Java for Android development.",
        
        # Java (island)
        "Java is the most populous island in Indonesia.",
        "Jakarta is the capital city on Java island.",
        "Java island has many active volcanoes.",
        "Coffee from Java is world-famous.",
        
        # Spring (framework)
        "Spring Framework is popular for enterprise Java.",
        "Spring Boot makes microservices easy.",
        "Spring Security handles authentication.",
        "I use Spring MVC for web applications.",
        
        # Spring (season)
        "Spring brings blooming flowers and warmer weather.",
        "In spring, the days get longer.",
        "Spring is between winter and summer.",
        "Cherry blossoms bloom in spring.",
        
        # Ruby (programming)
        "Ruby on Rails is a web framework.",
        "Ruby has elegant syntax.",
        "Many startups use Ruby for rapid development.",
        "Ruby is dynamically typed.",
        
        # Ruby (gemstone)
        "Ruby is a precious red gemstone.",
        "Rubies are valued for their deep red color.",
        "A ruby ring makes a beautiful gift.",
        "Rubies are harder than most gems.",
        
        # Go (programming)
        "Go is a language created by Google.",
        "Go has built-in concurrency support.",
        "Kubernetes is written in Go.",
        "Go compiles to native code.",
        
        # Go (game/verb)
        "Go is an ancient board game.",
        "Let's go to the park.",
        "Go ahead and start without me.",
        "The traffic light turned green, so go.",
    ]
    
    # Test queries
    test_cases = [
        {
            "query": "Python development and programming",
            "expected_type": "programming",
            "indices": {
                "programming": [0, 1, 2, 3],
                "snake": [4, 5, 6, 7]
            }
        },
        {
            "query": "Java enterprise applications", 
            "expected_type": "programming",
            "indices": {
                "programming": [8, 9, 10, 11],
                "island": [12, 13, 14, 15]
            }
        },
        {
            "query": "Spring framework configuration",
            "expected_type": "programming",
            "indices": {
                "framework": [16, 17, 18, 19],
                "season": [20, 21, 22, 23]
            }
        },
        {
            "query": "Ruby gems and libraries",
            "expected_type": "ambiguous",  # Could be either!
            "indices": {
                "programming": [24, 25, 26, 27],
                "gemstone": [28, 29, 30, 31]
            }
        },
        {
            "query": "Go concurrent programming",
            "expected_type": "programming",
            "indices": {
                "programming": [32, 33, 34, 35],
                "other": [36, 37, 38, 39]
            }
        }
    ]
    
    # Encode corpus
    embeddings = model.encode(corpus)
    
    print("AMBIGUOUS QUERY ANALYSIS")
    print("=" * 80)
    
    results_summary = []
    
    for test in test_cases:
        query = test["query"]
        query_vec = model.encode([query])[0]
        
        # Calculate metrics
        distances = np.linalg.norm(embeddings - query_vec, axis=1)
        cos_sims = cosine_similarity([query_vec], embeddings)[0]
        
        print(f"\nQuery: '{query}'")
        print(f"Expected: {test['expected_type']}")
        print("-" * 60)
        
        # Analyze each category
        categories = list(test["indices"].keys())
        cat_stats = {}
        
        for cat in categories:
            indices = test["indices"][cat]
            
            # Distance stats
            cat_dists = [distances[i] for i in indices]
            dist_mean = np.mean(cat_dists)
            dist_std = np.std(cat_dists)
            
            # Cosine stats
            cat_cos = [cos_sims[i] for i in indices]
            cos_mean = np.mean(cat_cos)
            cos_std = np.std(cat_cos)
            
            cat_stats[cat] = {
                "dist_mean": dist_mean,
                "dist_std": dist_std,
                "cos_mean": cos_mean,
                "cos_std": cos_std
            }
            
            print(f"\n{cat.upper()}:")
            print(f"  Distance: {dist_mean:.3f} (±{dist_std:.3f})")
            print(f"  Cosine:   {cos_mean:.3f} (±{cos_std:.3f})")
        
        # Calculate separation
        if len(categories) == 2:
            cat1, cat2 = categories
            
            # Distance separation
            dist_gap = abs(cat_stats[cat1]["dist_mean"] - cat_stats[cat2]["dist_mean"])
            dist_overlap = (cat_stats[cat1]["dist_std"] + cat_stats[cat2]["dist_std"]) / 2
            dist_separation = dist_gap / dist_overlap if dist_overlap > 0 else float('inf')
            
            # Cosine separation
            cos_gap = abs(cat_stats[cat1]["cos_mean"] - cat_stats[cat2]["cos_mean"])
            cos_overlap = (cat_stats[cat1]["cos_std"] + cat_stats[cat2]["cos_std"]) / 2
            cos_separation = cos_gap / cos_overlap if cos_overlap > 0 else float('inf')
            
            print(f"\nSEPARATION QUALITY:")
            print(f"  Distance: {dist_separation:.2f}x (gap/overlap)")
            print(f"  Cosine:   {cos_separation:.2f}x (gap/overlap)")
            print(f"  Winner:   {'Distance' if dist_separation > cos_separation else 'Cosine'}")
            
            results_summary.append({
                "query": query,
                "dist_sep": dist_separation,
                "cos_sep": cos_separation,
                "categories": categories,
                "stats": cat_stats
            })
        
        # Show top results
        print(f"\nTop 5 results by distance:")
        dist_order = np.argsort(distances)[:5]
        for i, idx in enumerate(dist_order):
            category = "?"
            for cat, indices in test["indices"].items():
                if idx in indices:
                    category = cat
                    break
            print(f"  {i+1}. [{distances[idx]:.3f}] ({category}) {corpus[idx][:40]}...")
    
    # Summary visualization
    if results_summary:
        plt.figure(figsize=(10, 6))
        
        queries = [r["query"].split()[0] for r in results_summary]  # First word
        dist_seps = [r["dist_sep"] for r in results_summary]
        cos_seps = [r["cos_sep"] for r in results_summary]
        
        x = np.arange(len(queries))
        width = 0.35
        
        plt.bar(x - width/2, dist_seps, width, label='Distance', alpha=0.8)
        plt.bar(x + width/2, cos_seps, width, label='Cosine', alpha=0.8)
        
        plt.xlabel('Query')
        plt.ylabel('Separation Quality (higher = better)')
        plt.title('Semantic Boundary Clarity: Distance vs Cosine')
        plt.xticks(x, queries, rotation=45)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        # Add reference line at 1.0
        plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Equal overlap')
        
        plt.tight_layout()
        plt.savefig('ambiguous_queries_comparison.png', dpi=150)
        print(f"\n\nSaved comparison chart to 'ambiguous_queries_comparison.png'")
    
    # Find optimal thresholds
    print("\n" + "=" * 80)
    print("OPTIMAL THRESHOLD ANALYSIS")
    print("=" * 80)
    
    for result in results_summary:
        if len(result["categories"]) != 2:
            continue
            
        cat1, cat2 = result["categories"]
        stats = result["stats"]
        
        # Find midpoint between categories
        dist_midpoint = (stats[cat1]["dist_mean"] + stats[cat2]["dist_mean"]) / 2
        cos_midpoint = (stats[cat1]["cos_mean"] + stats[cat2]["cos_mean"]) / 2
        
        print(f"\n{result['query']}:")
        print(f"  Distance threshold: {dist_midpoint:.3f}")
        print(f"  Cosine threshold:   {cos_midpoint:.3f}")


if __name__ == "__main__":
    analyze_ambiguous_queries()