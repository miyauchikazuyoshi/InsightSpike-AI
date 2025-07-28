#!/usr/bin/env python3
"""
Many Items Message Passing Experiment
====================================

Test message passing with more related items (7+) to see if 
richer context improves convergence towards D.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List


def message_passing_many_items(q_vec: np.ndarray,
                              item_vecs: Dict[str, np.ndarray],
                              q_weight: float = 0.5,
                              alpha: float = 0.3,
                              iterations: int = 5) -> tuple:
    """Message passing with many items"""
    
    vectors = {"Q": q_vec}
    vectors.update(item_vecs)
    
    h = {k: v.copy() for k, v in vectors.items()}
    
    # Calculate Q relevance
    q_relevance = {}
    for node, vec in vectors.items():
        if node == "Q":
            q_relevance[node] = q_weight
        else:
            q_relevance[node] = cosine_similarity([q_vec], [vec])[0][0]
    
    # Track evolution
    evolution = {"X": [], "avg_similarity": []}
    
    # Message passing
    for t in range(iterations):
        h_new = {}
        
        for node in vectors:
            messages = []
            weights = []
            
            for other in vectors:
                sim = cosine_similarity([h[node]], [h[other]])[0][0]
                weight = sim * (1 + alpha * q_relevance[other])
                
                messages.append(h[other])
                weights.append(weight)
            
            weights = np.array(weights)
            weights = weights / weights.sum()
            h_new[node] = np.average(messages, axis=0, weights=weights)
        
        h = h_new
        
        # Create X at this iteration
        X = np.zeros_like(q_vec)
        total_weight = 0
        
        for node, vec in h.items():
            weight = q_relevance[node]
            X += weight * vec
            total_weight += weight
        
        X = X / total_weight
        evolution["X"].append(X)
        
        # Track average pairwise similarity
        sims = []
        nodes = list(h.keys())
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                sims.append(cosine_similarity([h[nodes[i]]], [h[nodes[j]]])[0][0])
        evolution["avg_similarity"].append(np.mean(sims))
    
    return X, h, q_relevance, evolution


def run_many_items_experiment():
    """Test with different numbers of items"""
    
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    print("=== 多項目メッセージパッシング実験 ===\n")
    
    # Rich test case with many related items
    test_cases = [
        {
            "name": "Scientific Discovery",
            "question": "How do everyday observations lead to scientific breakthroughs?",
            "items_3": {
                "A": "Scientific breakthroughs emerge from careful observation",
                "B": "Major discoveries come from noticing patterns",
                "C": "Scientists transform observations into insights"
            },
            "items_7": {
                "A": "Scientific breakthroughs emerge from careful observation of everyday phenomena",
                "B": "Major discoveries come from noticing unusual patterns in common occurrences",
                "C": "Scientists transform familiar observations into profound insights",
                "D": "The history of science is filled with accidental discoveries from daily life",
                "E": "Curiosity about ordinary things drives scientific innovation",
                "F": "Systematic observation methods reveal hidden truths in nature",
                "G": "Pattern recognition in everyday events leads to theories"
            },
            "items_10": {
                "A": "Scientific breakthroughs emerge from careful observation of everyday phenomena",
                "B": "Major discoveries come from noticing unusual patterns in common occurrences",
                "C": "Scientists transform familiar observations into profound insights",
                "D": "The history of science is filled with accidental discoveries from daily life",
                "E": "Curiosity about ordinary things drives scientific innovation",
                "F": "Systematic observation methods reveal hidden truths in nature",
                "G": "Pattern recognition in everyday events leads to theories",
                "H": "Serendipity plays a crucial role in scientific advancement",
                "I": "Questioning the mundane often yields extraordinary answers",
                "J": "Scientific method transforms observations into knowledge"
            },
            "expected": "Scientific breakthroughs occur when prepared minds transform routine observations into profound insights by recognizing hidden patterns and fundamental principles in everyday phenomena."
        }
    ]
    
    for test_case in test_cases:
        print(f"\nTest: {test_case['name']}")
        print(f"Question: {test_case['question']}")
        print(f"Expected: {test_case['expected'][:80]}...")
        print("-" * 80)
        
        # Encode
        q_vec = model.encode(test_case["question"])
        d_vec = model.encode(test_case["expected"])
        
        results = []
        
        # Test different numbers of items
        for items_key in ["items_3", "items_7", "items_10"]:
            items = test_case[items_key]
            n_items = len(items)
            
            print(f"\n[{n_items} items experiment]")
            
            # Encode items
            item_vecs = {k: model.encode(v) for k, v in items.items()}
            
            # Check similarities
            avg_q_sim = np.mean([cosine_similarity([q_vec], [v])[0][0] for v in item_vecs.values()])
            print(f"Average Q↔items similarity: {avg_q_sim:.3f}")
            
            # Run message passing
            X, h, q_rel, evolution = message_passing_many_items(q_vec, item_vecs, iterations=5)
            
            # Calculate metrics
            x_to_q = cosine_similarity([X], [q_vec])[0][0]
            x_to_d = cosine_similarity([X], [d_vec])[0][0]
            
            x_to_items = []
            for vec in item_vecs.values():
                x_to_items.append(cosine_similarity([X], [vec])[0][0])
            x_to_items_avg = np.mean(x_to_items)
            
            # Item diversity (how different are the items)
            item_diversity = []
            item_keys = list(item_vecs.keys())
            for i in range(len(item_keys)):
                for j in range(i+1, len(item_keys)):
                    sim = cosine_similarity([item_vecs[item_keys[i]]], 
                                          [item_vecs[item_keys[j]]])[0][0]
                    item_diversity.append(1 - sim)
            avg_diversity = np.mean(item_diversity) if item_diversity else 0
            
            result = {
                "n_items": n_items,
                "x_to_q": x_to_q,
                "x_to_d": x_to_d,
                "x_to_items": x_to_items_avg,
                "avg_q_sim": avg_q_sim,
                "diversity": avg_diversity,
                "evolution": evolution,
                "final_X": X
            }
            results.append(result)
            
            print(f"X↔Q: {x_to_q:.3f}")
            print(f"X↔D: {x_to_d:.3f}")
            print(f"X↔items avg: {x_to_items_avg:.3f}")
            print(f"Item diversity: {avg_diversity:.3f}")
        
        # Visualization
        print("\n\n=== 可視化 ===")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: X↔D vs number of items
        ax1 = axes[0, 0]
        n_items_list = [r["n_items"] for r in results]
        x_to_d_list = [r["x_to_d"] for r in results]
        x_to_q_list = [r["x_to_q"] for r in results]
        
        ax1.plot(n_items_list, x_to_d_list, 'ro-', linewidth=2, markersize=10, label='X↔D')
        ax1.plot(n_items_list, x_to_q_list, 'bo-', linewidth=2, markersize=10, label='X↔Q')
        
        ax1.set_xlabel('Number of Items')
        ax1.set_ylabel('Similarity')
        ax1.set_title('X Similarity vs Number of Items')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(n_items_list)
        
        # Plot 2: Evolution of X↔D over iterations
        ax2 = axes[0, 1]
        
        for result in results:
            evolution_x_to_d = []
            for X_iter in result["evolution"]["X"]:
                sim = cosine_similarity([X_iter], [d_vec])[0][0]
                evolution_x_to_d.append(sim)
            
            iterations = list(range(1, len(evolution_x_to_d) + 1))
            ax2.plot(iterations, evolution_x_to_d, 'o-', 
                    label=f'{result["n_items"]} items', linewidth=2)
        
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('X↔D Similarity')
        ax2.set_title('Convergence of X towards D')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Item diversity vs X↔D
        ax3 = axes[1, 0]
        
        diversity_list = [r["diversity"] for r in results]
        
        ax3.scatter(diversity_list, x_to_d_list, s=200, alpha=0.7)
        for i, n in enumerate(n_items_list):
            ax3.annotate(f'{n} items', (diversity_list[i], x_to_d_list[i]),
                        xytext=(5, 5), textcoords='offset points')
        
        ax3.set_xlabel('Average Item Diversity')
        ax3.set_ylabel('X↔D Similarity')
        ax3.set_title('Item Diversity vs Final X↔D')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: 2D projection of vectors
        ax4 = axes[1, 1]
        
        from sklearn.decomposition import PCA
        
        # Collect all vectors for largest item set
        largest_result = results[-1]  # 10 items
        all_vecs = [q_vec, d_vec, largest_result["final_X"]]
        all_labels = ["Q", "D", "X"]
        
        # Add some item vectors
        for i, (k, v) in enumerate(list(item_vecs.items())[:5]):
            all_vecs.append(v)
            all_labels.append(f"Item_{k}")
        
        # PCA projection
        pca = PCA(n_components=2, random_state=42)
        coords_2d = pca.fit_transform(all_vecs)
        
        # Plot
        ax4.scatter(coords_2d[0, 0], coords_2d[0, 1], s=300, c='blue', 
                   marker='o', label='Q', edgecolors='black', linewidth=2)
        ax4.scatter(coords_2d[1, 0], coords_2d[1, 1], s=300, c='red', 
                   marker='s', label='D', edgecolors='black', linewidth=2)
        ax4.scatter(coords_2d[2, 0], coords_2d[2, 1], s=300, c='green', 
                   marker='*', label='X', edgecolors='black', linewidth=2)
        
        # Plot items
        for i in range(3, len(coords_2d)):
            ax4.scatter(coords_2d[i, 0], coords_2d[i, 1], s=150, 
                       c='orange', marker='^', alpha=0.6)
            ax4.annotate(all_labels[i], (coords_2d[i, 0], coords_2d[i, 1]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax4.set_title('Vector Space Visualization (10 items)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('many_items_message_passing.png', dpi=150, bbox_inches='tight')
        print("結果を 'many_items_message_passing.png' に保存")
        
        # Summary
        print("\n\n=== まとめ ===")
        print("-" * 60)
        
        # Find best configuration
        best_result = max(results, key=lambda r: r["x_to_d"])
        print(f"最高 X↔D: {best_result['n_items']} items ({best_result['x_to_d']:.3f})")
        
        # Compare improvements
        baseline_d = results[0]["x_to_d"]  # 3 items
        for result in results[1:]:
            improvement = result["x_to_d"] - baseline_d
            print(f"{result['n_items']} items vs 3 items: {improvement:+.3f} improvement")
        
        print("\n発見:")
        print("- 項目数を増やすとX↔Dが向上する可能性")
        print("- より多様な視点が統合されることで、Dに近づく")
        print("- ただし、項目の質と多様性が重要")


if __name__ == "__main__":
    run_many_items_experiment()