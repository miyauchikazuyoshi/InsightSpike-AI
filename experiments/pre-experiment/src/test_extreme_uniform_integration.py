#!/usr/bin/env python3
"""
Extreme Uniform Integration Test
================================

Test with more extreme cases where Q relevance varies greatly.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple


def create_extreme_integration(q_vec: np.ndarray,
                             item_vecs: Dict[str, np.ndarray],
                             include_q: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create three types of integration:
    1. Q-weighted (traditional)
    2. Uniform with Q
    3. Uniform without Q (pure knowledge)
    """
    
    # Calculate Q relevance
    q_relevance = {}
    for label, vec in item_vecs.items():
        q_relevance[label] = cosine_similarity([q_vec], [vec])[0][0]
    
    # 1. Q-weighted integration
    X_weighted = np.zeros_like(q_vec)
    total_weight = 0
    
    if include_q:
        X_weighted += q_vec  # Q has weight 1.0
        total_weight += 1.0
    
    for label, vec in item_vecs.items():
        weight = q_relevance[label]
        X_weighted += weight * vec
        total_weight += weight
    
    X_weighted = X_weighted / total_weight
    
    # 2. Uniform with Q
    all_vecs = [q_vec] + list(item_vecs.values()) if include_q else list(item_vecs.values())
    X_uniform_with_q = np.mean(all_vecs, axis=0)
    
    # 3. Uniform without Q (pure knowledge integration)
    X_uniform_no_q = np.mean(list(item_vecs.values()), axis=0)
    
    return X_weighted, X_uniform_with_q, X_uniform_no_q, q_relevance


def run_extreme_test():
    """Test with extreme relevance variations"""
    
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    print("=== 極端なケースでの均等重み統合テスト ===\n")
    
    # Create test cases with varying relevance
    test_cases = [
        {
            "name": "High Variance Relevance",
            "question": "What is quantum entanglement?",
            "items": {
                "A": "Quantum particles can be connected across vast distances",  # High relevance
                "B": "Einstein called it 'spooky action at a distance'",  # Medium
                "C": "Apples fall from trees due to gravity",  # Very low relevance
                "D": "Weather patterns are chaotic systems",  # Very low relevance
                "E": "Coffee contains caffeine"  # Extremely low relevance
            },
            "expected": "Quantum entanglement is a phenomenon where particles remain connected and instantly affect each other regardless of distance."
        },
        {
            "name": "All High Relevance",
            "question": "How does machine learning work?",
            "items": {
                "A": "Machine learning algorithms learn patterns from data",
                "B": "Neural networks are inspired by brain structure",
                "C": "Training data is used to optimize model parameters",
                "D": "Gradient descent minimizes the loss function",
                "E": "Validation sets prevent overfitting"
            },
            "expected": "Machine learning works by training algorithms on data to recognize patterns and make predictions without explicit programming."
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f"\n{'='*60}")
        print(f"Test: {test_case['name']}")
        print(f"Q: {test_case['question']}")
        print('='*60)
        
        # Encode
        q_vec = model.encode(test_case["question"])
        d_vec = model.encode(test_case["expected"])
        item_vecs = {k: model.encode(v) for k, v in test_case["items"].items()}
        
        # Create integrations
        X_weighted, X_uniform_q, X_uniform_no_q, q_relevance = create_extreme_integration(
            q_vec, item_vecs
        )
        
        # Show relevance distribution
        print("\n[Q関連度分布]")
        sorted_rel = sorted(q_relevance.items(), key=lambda x: x[1], reverse=True)
        for label, rel in sorted_rel:
            bar = "█" * int(rel * 20)
            print(f"  {label}: {rel:.3f} {bar}")
        
        # Calculate all similarities
        methods = {
            "Weighted (Q考慮)": X_weighted,
            "Uniform (Q含む)": X_uniform_q,
            "Uniform (Q除外)": X_uniform_no_q
        }
        
        print("\n[各手法の類似度]")
        print(f"{'Method':<20} | X↔Q  | X↔D  | X↔Items")
        print("-" * 50)
        
        for method_name, X in methods.items():
            x_to_q = cosine_similarity([X], [q_vec])[0][0]
            x_to_d = cosine_similarity([X], [d_vec])[0][0]
            x_to_items = np.mean([cosine_similarity([X], [v])[0][0] for v in item_vecs.values()])
            
            print(f"{method_name:<20} | {x_to_q:.3f} | {x_to_d:.3f} | {x_to_items:.3f}")
            
            results.append({
                "case": test_case["name"],
                "method": method_name,
                "x_to_q": x_to_q,
                "x_to_d": x_to_d,
                "x_to_items": x_to_items
            })
    
    # Visualization
    print("\n\n=== 可視化 ===")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Bar comparison
    ax1.set_title('X↔D Similarity by Integration Method')
    
    # Prepare data
    case_names = list(set(r["case"] for r in results))
    method_names = ["Weighted (Q考慮)", "Uniform (Q含む)", "Uniform (Q除外)"]
    
    x = np.arange(len(case_names))
    width = 0.25
    
    for i, method in enumerate(method_names):
        values = [r["x_to_d"] for r in results if r["method"] == method]
        ax1.bar(x + i*width, values, width, label=method, alpha=0.8)
    
    ax1.set_ylabel('X↔D Similarity')
    ax1.set_xlabel('Test Case')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(case_names, rotation=15)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Q-D space trajectories
    ax2.set_title('Integration Methods in Q-D Space')
    
    # Reference points
    ax2.scatter([1], [0], s=300, c='blue', marker='o', label='Q', 
               edgecolors='black', linewidth=2, zorder=5)
    ax2.scatter([0], [1], s=300, c='red', marker='s', label='D', 
               edgecolors='black', linewidth=2, zorder=5)
    
    # Plot each method's position
    colors = plt.cm.Set2(range(len(case_names)))
    markers = ['o', 's', '^']
    
    for i, case in enumerate(case_names):
        case_results = [r for r in results if r["case"] == case]
        
        for j, method in enumerate(method_names):
            method_result = next(r for r in case_results if r["method"] == method)
            ax2.scatter(method_result["x_to_q"], method_result["x_to_d"],
                       s=150, c=[colors[i]], marker=markers[j], 
                       alpha=0.8, edgecolors='black', linewidth=1)
        
        # Connect methods for same case
        x_coords = [r["x_to_q"] for r in case_results]
        y_coords = [r["x_to_d"] for r in case_results]
        ax2.plot(x_coords, y_coords, c=colors[i], alpha=0.3, linestyle='--')
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = []
    
    # Add method markers
    for j, (method, marker) in enumerate(zip(method_names, markers)):
        legend_elements.append(Line2D([0], [0], marker=marker, color='w', 
                                    markerfacecolor='gray', markersize=8, 
                                    label=method, markeredgecolor='black'))
    
    ax2.legend(handles=legend_elements, loc='lower left')
    
    ax2.plot([0, 1], [0, 1], 'gray', alpha=0.2, linestyle='--')
    ax2.set_xlabel('X↔Q Similarity')
    ax2.set_ylabel('X↔D Similarity')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.1, 1.1)
    ax2.set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    plt.savefig('extreme_uniform_integration.png', dpi=150, bbox_inches='tight')
    print("結果を 'extreme_uniform_integration.png' に保存")
    
    # Analysis
    print("\n\n=== 分析 ===")
    
    # Find best method for each case
    for case in case_names:
        print(f"\n{case}:")
        case_results = [r for r in results if r["case"] == case]
        best = max(case_results, key=lambda r: r["x_to_d"])
        print(f"  最高 X↔D: {best['method']} ({best['x_to_d']:.3f})")
        
        # Compare improvements
        weighted = next(r for r in case_results if "Weighted" in r["method"])
        uniform_no_q = next(r for r in case_results if "Q除外" in r["method"])
        improvement = uniform_no_q["x_to_d"] - weighted["x_to_d"]
        print(f"  Q除外の効果: {improvement:+.3f}")
    
    print("\n結論:")
    print("- 関連性の低い項目が多い場合、Q除外が効果的")
    print("- すべて高関連の場合、Qを含めた方が良い")
    print("- 状況に応じて統合戦略を変える必要がある")


if __name__ == "__main__":
    run_extreme_test()