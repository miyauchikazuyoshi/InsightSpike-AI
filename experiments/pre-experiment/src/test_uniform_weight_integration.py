#!/usr/bin/env python3
"""
Uniform Weight Integration Experiment
====================================

Test the effect of using uniform weights (instead of Q-relevance weights)
when creating the integrated vector X.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple


def message_passing_with_uniform_integration(q_vec: np.ndarray,
                                           item_vecs: Dict[str, np.ndarray],
                                           alpha: float = 0.3,
                                           iterations: int = 3) -> Tuple[np.ndarray, Dict]:
    """
    Message passing with UNIFORM weights for final integration
    """
    
    vectors = {"Q": q_vec}
    vectors.update(item_vecs)
    
    h = {k: v.copy() for k, v in vectors.items()}
    
    # Calculate Q relevance (still used during message passing)
    q_relevance = {}
    for node, vec in vectors.items():
        if node == "Q":
            q_relevance[node] = 1.0
        else:
            q_relevance[node] = cosine_similarity([q_vec], [vec])[0][0]
    
    # Message passing (still uses Q relevance)
    for t in range(iterations):
        h_new = {}
        
        for node in vectors:
            messages = []
            weights = []
            
            for other in vectors:
                sim = cosine_similarity([h[node]], [h[other]])[0][0]
                # Still use Q relevance during message passing
                weight = sim * (1 + alpha * q_relevance[other])
                
                messages.append(h[other])
                weights.append(weight)
            
            weights = np.array(weights)
            weights = weights / weights.sum()
            h_new[node] = np.average(messages, axis=0, weights=weights)
        
        h = h_new
    
    # Create X with UNIFORM weights (key difference!)
    X_uniform = np.mean(list(h.values()), axis=0)
    
    # Also create traditional weighted version for comparison
    X_weighted = np.zeros_like(q_vec)
    total_weight = 0
    for node, vec in h.items():
        weight = q_relevance[node]
        X_weighted += weight * vec
        total_weight += weight
    X_weighted = X_weighted / total_weight
    
    return X_uniform, X_weighted, h, q_relevance


def run_comparison_experiment():
    """Compare uniform vs weighted integration"""
    
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    print("=== 均等重み vs 類似度重み統合の比較実験 ===\n")
    
    # Test cases
    test_cases = [
        {
            "name": "Scientific Discovery",
            "question": "How do everyday observations lead to scientific breakthroughs?",
            "items": {
                "A": "Scientific breakthroughs often emerge from careful observation",
                "B": "Major discoveries come from noticing patterns in common phenomena",
                "C": "Scientists transform familiar observations into insights",
                "D": "Pattern recognition in everyday events leads to theories",
                "E": "Questioning the mundane yields extraordinary answers"
            },
            "expected": "Scientific breakthroughs occur when prepared minds transform routine observations into profound insights by recognizing hidden patterns."
        },
        {
            "name": "Problem Solving",
            "question": "What role does creativity play in problem solving?",
            "items": {
                "A": "Creative thinking allows us to see problems from new angles",
                "B": "Innovation comes from combining existing ideas in novel ways",
                "C": "Breaking conventional thinking patterns leads to solutions",
                "D": "Lateral thinking opens unexpected solution paths",
                "E": "Imagination helps visualize beyond constraints"
            },
            "expected": "Creativity enables problem solving by generating novel perspectives and unconventional approaches that transcend traditional constraints."
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f"\nTest Case: {test_case['name']}")
        print(f"Question: {test_case['question']}")
        print("-" * 80)
        
        # Encode
        q_vec = model.encode(test_case["question"])
        d_vec = model.encode(test_case["expected"])
        item_vecs = {k: model.encode(v) for k, v in test_case["items"].items()}
        
        # Run message passing
        X_uniform, X_weighted, h, q_relevance = message_passing_with_uniform_integration(
            q_vec, item_vecs
        )
        
        # Calculate similarities
        uniform_to_q = cosine_similarity([X_uniform], [q_vec])[0][0]
        uniform_to_d = cosine_similarity([X_uniform], [d_vec])[0][0]
        
        weighted_to_q = cosine_similarity([X_weighted], [q_vec])[0][0]
        weighted_to_d = cosine_similarity([X_weighted], [d_vec])[0][0]
        
        # Item similarities
        uniform_to_items = np.mean([cosine_similarity([X_uniform], [v])[0][0] 
                                   for v in item_vecs.values()])
        weighted_to_items = np.mean([cosine_similarity([X_weighted], [v])[0][0] 
                                    for v in item_vecs.values()])
        
        print("\n[重み付け統合（従来）]")
        print(f"X_weighted ↔ Q: {weighted_to_q:.3f}")
        print(f"X_weighted ↔ D: {weighted_to_d:.3f}")
        print(f"X_weighted ↔ items: {weighted_to_items:.3f}")
        
        print("\n[均等重み統合（新提案）]")
        print(f"X_uniform ↔ Q: {uniform_to_q:.3f}")
        print(f"X_uniform ↔ D: {uniform_to_d:.3f}")
        print(f"X_uniform ↔ items: {uniform_to_items:.3f}")
        
        print("\n[改善度]")
        improvement = uniform_to_d - weighted_to_d
        print(f"D への近さの改善: {improvement:+.3f}")
        print(f"Q からの距離変化: {uniform_to_q - weighted_to_q:+.3f}")
        
        # Store results
        results.append({
            "case": test_case["name"],
            "weighted_to_q": weighted_to_q,
            "weighted_to_d": weighted_to_d,
            "uniform_to_q": uniform_to_q,
            "uniform_to_d": uniform_to_d,
            "improvement": improvement
        })
        
        # Show Q relevance distribution
        print("\n[Q関連度の分布]")
        for node, relevance in sorted(q_relevance.items(), key=lambda x: x[1], reverse=True):
            if node != "Q":
                print(f"  {node}: {relevance:.3f}")
    
    # Visualization
    print("\n\n=== 結果の可視化 ===")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Comparison bar chart
    ax1.set_title('Integration Method Comparison')
    
    x = np.arange(len(results))
    width = 0.35
    
    weighted_d = [r["weighted_to_d"] for r in results]
    uniform_d = [r["uniform_to_d"] for r in results]
    
    bars1 = ax1.bar(x - width/2, weighted_d, width, label='Weighted', alpha=0.8)
    bars2 = ax1.bar(x + width/2, uniform_d, width, label='Uniform', alpha=0.8)
    
    # Add improvement labels
    for i, (w, u) in enumerate(zip(weighted_d, uniform_d)):
        if u > w:
            ax1.text(i, max(w, u) + 0.01, f'+{u-w:.3f}', 
                    ha='center', va='bottom', fontsize=9, color='green')
        else:
            ax1.text(i, max(w, u) + 0.01, f'{u-w:.3f}', 
                    ha='center', va='bottom', fontsize=9, color='red')
    
    ax1.set_ylabel('X ↔ D Similarity')
    ax1.set_xlabel('Test Case')
    ax1.set_xticks(x)
    ax1.set_xticklabels([r["case"] for r in results])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Q-D space scatter
    ax2.set_title('X Position in Q-D Space')
    
    # Reference points
    ax2.scatter([1], [0], s=200, c='blue', marker='o', label='Q', 
               edgecolors='black', linewidth=2)
    ax2.scatter([0], [1], s=200, c='red', marker='s', label='D', 
               edgecolors='black', linewidth=2)
    
    # Plot X positions
    colors = plt.cm.Set1(range(len(results)))
    
    for i, result in enumerate(results):
        # Weighted X
        ax2.scatter(result["weighted_to_q"], result["weighted_to_d"], 
                   s=150, c=[colors[i]], marker='o', alpha=0.7,
                   label=f'{result["case"]} (W)')
        
        # Uniform X
        ax2.scatter(result["uniform_to_q"], result["uniform_to_d"], 
                   s=150, c=[colors[i]], marker='s', alpha=0.7,
                   label=f'{result["case"]} (U)')
        
        # Connect them
        ax2.plot([result["weighted_to_q"], result["uniform_to_q"]],
                [result["weighted_to_d"], result["uniform_to_d"]],
                c=colors[i], linestyle='--', alpha=0.5)
        
        # Add arrow
        ax2.annotate('', 
                    xy=(result["uniform_to_q"], result["uniform_to_d"]),
                    xytext=(result["weighted_to_q"], result["weighted_to_d"]),
                    arrowprops=dict(arrowstyle='->', color=colors[i], alpha=0.7))
    
    ax2.plot([0, 1], [0, 1], 'gray', alpha=0.2, linestyle='--')
    ax2.set_xlabel('X ↔ Q Similarity')
    ax2.set_ylabel('X ↔ D Similarity')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0.4, 1.1)
    ax2.set_ylim(0.4, 1.1)
    
    # Simplified legend
    ax2.scatter([], [], c='gray', marker='o', label='Weighted')
    ax2.scatter([], [], c='gray', marker='s', label='Uniform')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('uniform_weight_comparison.png', dpi=150, bbox_inches='tight')
    print("結果を 'uniform_weight_comparison.png' に保存")
    
    # Summary
    print("\n\n=== まとめ ===")
    avg_improvement = np.mean([r["improvement"] for r in results])
    print(f"平均改善度: {avg_improvement:+.3f}")
    
    if avg_improvement > 0:
        print("\n結論: 均等重み統合により、Dへの近接性が向上！")
        print("理由: Qの過度な影響を排除し、知識の純粋な統合を実現")
    else:
        print("\n結論: 重み付け統合の方が効果的")
        print("理由: 質問文脈の考慮が重要")
    
    # Prompt generation example
    print("\n\n=== プロンプト生成の例 ===")
    print("\n均等重み統合後、類似度を計算してプロンプトに含める：")
    
    # Example for first test case
    X = X_uniform  # Use uniform integration
    
    prompt = f"""Question: "{test_cases[0]['question']}"

After knowledge integration, the unified representation shows:
- Relevance to question: {cosine_similarity([X], [q_vec])[0][0]:.3f}
- Integration with knowledge items: {uniform_to_items:.3f}

Based on this integrated understanding, the key insight is:"""
    
    print(prompt)


if __name__ == "__main__":
    run_comparison_experiment()