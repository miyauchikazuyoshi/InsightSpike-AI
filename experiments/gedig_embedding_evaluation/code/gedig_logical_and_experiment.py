#!/usr/bin/env python3
"""
geDIG Logical AND Experiment
============================

This experiment tests using logical AND operations instead of linear combinations
for intrinsic reward thresholds in geDIG embeddings.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Set random seeds
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

class LogicalGeDIG:
    """geDIG with logical AND operations for intrinsic rewards"""
    
    def __init__(self, ig_threshold=0.3, ged_threshold=-0.1, operation="and", name="geDIG"):
        self.ig_threshold = ig_threshold
        self.ged_threshold = ged_threshold
        self.operation = operation
        self.name = name
        
    def compute_similarity(self, query, document):
        """Compute geDIG similarity with logical operations"""
        # Simple IG: word overlap ratio
        q_words = set(query.lower().split())
        d_words = set(document.lower().split())
        overlap = len(q_words & d_words)
        total = len(q_words | d_words)
        ig_score = overlap / (total + 1e-8)
        
        # Simple GED: structural difference (negative is better)
        length_diff = abs(len(q_words) - len(d_words))
        ged_score = -length_diff / (len(q_words) + len(d_words) + 1e-8)
        
        # Check threshold conditions
        ig_pass = ig_score > self.ig_threshold
        ged_pass = ged_score > self.ged_threshold
        
        if self.operation == "and":
            # Logical AND: both conditions must be met
            if ig_pass and ged_pass:
                return ig_score + (-ged_score)  # Return combined score
            else:
                return 0.0  # No match if either condition fails
                
        elif self.operation == "or":
            # Logical OR: at least one condition must be met
            if ig_pass or ged_pass:
                return ig_score + (-ged_score)
            else:
                return 0.0
                
        elif self.operation == "weighted_and":
            # Weighted AND: score proportional to how many conditions are met
            score = ig_score + (-ged_score)
            if ig_pass and ged_pass:
                return score * 1.0  # Full score
            elif ig_pass or ged_pass:
                return score * 0.5  # Half score
            else:
                return score * 0.1  # Small non-zero score
                
        elif self.operation == "multiplicative":
            # Multiplicative: score is product of threshold activations
            ig_activation = 1.0 if ig_pass else 0.1
            ged_activation = 1.0 if ged_pass else 0.1
            base_score = ig_score + (-ged_score)
            return base_score * ig_activation * ged_activation
            
        else:  # linear (original)
            # Linear combination with threshold boost
            ig_activated = 1.0 if ig_pass else 0.5
            ged_activated = 1.0 if ged_pass else 0.5
            base_score = 0.7 * ig_score + 0.3 * (-ged_score)
            intrinsic_boost = ig_activated * ged_activated
            return base_score * intrinsic_boost

def create_test_dataset(n_questions=100):
    """Create a more diverse test dataset"""
    questions = []
    documents = []
    
    # Different question types
    question_templates = [
        ("What is {}?", "{} is a concept in computer science."),
        ("How does {} work?", "{} works by processing data efficiently."),
        ("Define {}.", "The definition of {} involves several key aspects."),
        ("Explain {} briefly.", "{} can be explained as a method for solving problems."),
        ("What are the benefits of {}?", "The benefits of {} include improved performance."),
    ]
    
    topics = ["Python", "machine learning", "data science", "neural networks", 
              "algorithms", "databases", "cloud computing", "cybersecurity"]
    
    for i in range(n_questions):
        template_idx = i % len(question_templates)
        topic_idx = i % len(topics)
        topic = topics[topic_idx]
        
        q_template, d_template = question_templates[template_idx]
        questions.append(q_template.format(topic))
        
        # Add relevant document
        documents.append(d_template.format(topic))
        
        # Add semi-relevant documents (same template, different topic)
        for j in range(2):
            other_topic = topics[(topic_idx + j + 1) % len(topics)]
            documents.append(d_template.format(other_topic))
        
        # Add irrelevant documents (different template and topic)
        for j in range(2):
            other_template_idx = (template_idx + j + 1) % len(question_templates)
            other_topic = topics[(topic_idx + j + 3) % len(topics)]
            _, other_d_template = question_templates[other_template_idx]
            documents.append(other_d_template.format(other_topic))
    
    return questions, documents

def evaluate_logical_operations():
    """Evaluate different logical operations for geDIG"""
    
    print("Loading data and models...")
    questions, documents = create_test_dataset(100)
    
    # Initialize methods with different logical operations
    methods = {
        "TF-IDF": TfidfVectorizer(max_features=1000),
        "Sentence-BERT": SentenceTransformer('all-MiniLM-L6-v2'),
        
        # Different logical operations with medium thresholds
        "geDIG-Linear": LogicalGeDIG(0.3, -0.1, "linear", "geDIG-Linear"),
        "geDIG-AND": LogicalGeDIG(0.3, -0.1, "and", "geDIG-AND"),
        "geDIG-OR": LogicalGeDIG(0.3, -0.1, "or", "geDIG-OR"),
        "geDIG-WeightedAND": LogicalGeDIG(0.3, -0.1, "weighted_and", "geDIG-WeightedAND"),
        "geDIG-Multiplicative": LogicalGeDIG(0.3, -0.1, "multiplicative", "geDIG-Multiplicative"),
        
        # AND operation with different thresholds
        "geDIG-AND-Low": LogicalGeDIG(0.2, -0.15, "and", "geDIG-AND-Low"),
        "geDIG-AND-High": LogicalGeDIG(0.4, -0.05, "and", "geDIG-AND-High"),
    }
    
    results = defaultdict(dict)
    
    # Fit TF-IDF
    all_texts = questions + documents
    methods["TF-IDF"].fit(all_texts)
    tfidf_docs = methods["TF-IDF"].transform(documents)
    
    # Encode with Sentence-BERT
    print("Encoding with Sentence-BERT...")
    sbert_docs = methods["Sentence-BERT"].encode(documents, show_progress_bar=False)
    
    print("\nEvaluating methods...")
    
    for method_name, method in methods.items():
        print(f"\nEvaluating {method_name}...")
        
        recalls_at_k = {1: [], 5: [], 10: []}
        precisions_at_k = {1: [], 5: [], 10: []}
        latencies = []
        threshold_hits = []
        
        for i, query in enumerate(questions):
            start_time = time.time()
            
            if method_name == "TF-IDF":
                query_vec = method.transform([query])
                similarities = cosine_similarity(query_vec, tfidf_docs).flatten()
                
            elif method_name == "Sentence-BERT":
                query_emb = method.encode([query])
                similarities = cosine_similarity(query_emb, sbert_docs).flatten()
                
            else:  # geDIG variants
                similarities = []
                hits = 0
                for doc in documents:
                    sim = method.compute_similarity(query, doc)
                    similarities.append(sim)
                    if sim > 0.1:  # Count non-zero scores for AND operations
                        hits += 1
                similarities = np.array(similarities)
                threshold_hits.append(hits)
            
            latency = (time.time() - start_time) * 1000  # ms
            latencies.append(latency)
            
            # Calculate metrics
            # Relevant document is at position i*5
            relevant_doc = i * 5
            rankings = np.argsort(similarities)[::-1]
            
            # Find position of relevant document
            relevant_pos = np.where(rankings == relevant_doc)[0]
            if len(relevant_pos) > 0:
                rank = relevant_pos[0] + 1
            else:
                rank = len(documents) + 1
            
            # Calculate recall and precision at different k values
            for k in [1, 5, 10]:
                top_k = rankings[:k]
                recall = 1 if relevant_doc in top_k else 0
                precision = 1/k if relevant_doc in top_k else 0
                
                recalls_at_k[k].append(recall)
                precisions_at_k[k].append(precision)
        
        # Store results
        for k in [1, 5, 10]:
            results[method_name][f"recall@{k}"] = np.mean(recalls_at_k[k])
            results[method_name][f"precision@{k}"] = np.mean(precisions_at_k[k])
        
        results[method_name]["latency_ms"] = np.mean(latencies)
        
        if threshold_hits:  # For geDIG methods
            results[method_name]["avg_hits"] = np.mean(threshold_hits)
            results[method_name]["hit_rate"] = np.mean(threshold_hits) / len(documents)
        
        print(f"  Recall@5: {results[method_name]['recall@5']:.3f}")
        print(f"  Precision@5: {results[method_name]['precision@5']:.3f}")
        print(f"  Latency: {results[method_name]['latency_ms']:.1f}ms")
        if "avg_hits" in results[method_name]:
            print(f"  Avg threshold hits: {results[method_name]['avg_hits']:.1f}")
    
    return results

def visualize_logical_results(results):
    """Create visualizations for logical operation results"""
    
    output_dir = Path("results_logical_operations")
    output_dir.mkdir(exist_ok=True)
    
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Filter geDIG methods for operation comparison
    operation_methods = ["geDIG-Linear", "geDIG-AND", "geDIG-OR", 
                        "geDIG-WeightedAND", "geDIG-Multiplicative"]
    threshold_methods = ["geDIG-AND-Low", "geDIG-AND", "geDIG-AND-High"]
    
    # 1. Recall@5 comparison by operation
    ax = axes[0, 0]
    recalls = [results[m]["recall@5"] for m in operation_methods]
    colors = ['gray', 'red', 'blue', 'green', 'purple']
    bars = ax.bar(range(len(operation_methods)), recalls, color=colors)
    ax.set_xticks(range(len(operation_methods)))
    ax.set_xticklabels([m.split('-')[1] for m in operation_methods], rotation=45)
    ax.set_ylabel('Recall@5')
    ax.set_title('Performance by Logical Operation')
    
    # Add value labels on bars
    for bar, value in zip(bars, recalls):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # 2. Precision-Recall trade-off
    ax = axes[0, 1]
    for method, color in zip(operation_methods, colors):
        recalls = [results[method][f"recall@{k}"] for k in [1, 5, 10]]
        precisions = [results[method][f"precision@{k}"] for k in [1, 5, 10]]
        ax.plot(recalls, precisions, 'o-', label=method.split('-')[1], color=color)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Trade-off')
    ax.legend()
    
    # 3. Threshold sensitivity for AND operation
    ax = axes[0, 2]
    recalls = [results[m]["recall@5"] for m in threshold_methods]
    ax.plot(['Low\n(0.2,-0.15)', 'Medium\n(0.3,-0.1)', 'High\n(0.4,-0.05)'], 
            recalls, 'ro-', linewidth=2, markersize=10)
    ax.set_xlabel('Threshold Settings')
    ax.set_ylabel('Recall@5')
    ax.set_title('AND Operation: Threshold Sensitivity')
    ax.grid(True, alpha=0.3)
    
    # 4. Hit rate comparison
    ax = axes[1, 0]
    hit_methods = [m for m in operation_methods if "avg_hits" in results[m]]
    if hit_methods:
        hit_rates = [results[m]["hit_rate"] * 100 for m in hit_methods]
        bars = ax.bar(range(len(hit_methods)), hit_rates, color=colors[1:])
        ax.set_xticks(range(len(hit_methods)))
        ax.set_xticklabels([m.split('-')[1] for m in hit_methods], rotation=45)
        ax.set_ylabel('Hit Rate (%)')
        ax.set_title('Percentage of Documents Passing Thresholds')
    
    # 5. Performance vs Latency
    ax = axes[1, 1]
    all_methods = list(results.keys())
    for method in all_methods:
        perf = results[method]["recall@5"]
        latency = results[method]["latency_ms"]
        if method == "TF-IDF":
            ax.scatter(latency, perf, s=100, color='blue', marker='s', label=method)
        elif method == "Sentence-BERT":
            ax.scatter(latency, perf, s=100, color='green', marker='^', label=method)
        elif "AND" in method:
            ax.scatter(latency, perf, s=100, color='red', marker='o', label=method if "geDIG-AND" == method else "")
        else:
            ax.scatter(latency, perf, s=50, color='gray', alpha=0.5)
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Recall@5')
    ax.set_title('Performance vs Speed')
    ax.set_xscale('log')
    ax.legend(loc='best')
    
    # 6. Comparison summary
    ax = axes[1, 2]
    # Create a heatmap of different metrics
    methods_subset = ["TF-IDF", "Sentence-BERT", "geDIG-Linear", "geDIG-AND", "geDIG-OR"]
    metrics = ["recall@1", "recall@5", "recall@10"]
    data = []
    for method in methods_subset:
        if method in results:
            row = [results[method][metric] for metric in metrics]
            data.append(row)
    
    im = ax.imshow(data, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics)
    ax.set_yticks(range(len(methods_subset)))
    ax.set_yticklabels(methods_subset)
    ax.set_title('Performance Heatmap')
    
    # Add text annotations
    for i in range(len(methods_subset)):
        for j in range(len(metrics)):
            text = ax.text(j, i, f'{data[i][j]:.2f}', ha="center", va="center", color="black")
    
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'logical_operations_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results
    with open(output_dir / 'logical_operations_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create summary report
    create_logical_summary_report(results, output_dir)
    
    print(f"\nResults saved to {output_dir}")

def create_logical_summary_report(results, output_dir):
    """Create a markdown summary report for logical operations"""
    
    report = ["# geDIG Logical Operations Analysis\n"]
    report.append(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"**Random Seed**: {RANDOM_SEED}\n")
    
    report.append("\n## Key Findings\n")
    
    # Find best logical operation
    operation_methods = ["geDIG-Linear", "geDIG-AND", "geDIG-OR", 
                        "geDIG-WeightedAND", "geDIG-Multiplicative"]
    operation_perfs = [(m, results[m]["recall@5"]) for m in operation_methods if m in results]
    best_op = max(operation_perfs, key=lambda x: x[1])
    
    report.append(f"1. **Best Logical Operation**: {best_op[0].split('-')[1]} (Recall@5={best_op[1]:.3f})")
    
    # Compare to baselines
    baseline_perf = results.get("Sentence-BERT", {}).get("recall@5", 0)
    report.append(f"2. **vs Sentence-BERT**: {best_op[1]/baseline_perf*100:.1f}% of SBERT performance")
    
    # AND operation analysis
    and_methods = [m for m in results if "AND" in m and m != "geDIG-WeightedAND"]
    if and_methods:
        and_perfs = [(m, results[m]["recall@5"]) for m in and_methods]
        best_and = max(and_perfs, key=lambda x: x[1])
        report.append(f"3. **Best AND Configuration**: {best_and[0]} (Recall@5={best_and[1]:.3f})")
    
    report.append("\n## Logical Operations Comparison\n")
    report.append("| Operation | Recall@1 | Recall@5 | Recall@10 | Avg Hits | Description |")
    report.append("|-----------|----------|----------|-----------|----------|-------------|")
    
    descriptions = {
        "Linear": "Weighted sum with threshold boost",
        "AND": "Both IG and GED must pass thresholds",
        "OR": "Either IG or GED must pass thresholds",
        "WeightedAND": "Graduated scoring based on threshold passing",
        "Multiplicative": "Score multiplied by threshold activations"
    }
    
    for method in operation_methods:
        if method in results:
            r = results[method]
            op = method.split('-')[1]
            avg_hits = r.get('avg_hits', 'N/A')
            if avg_hits != 'N/A':
                avg_hits = f"{avg_hits:.1f}"
            report.append(f"| {op} | {r['recall@1']:.3f} | {r['recall@5']:.3f} | "
                         f"{r['recall@10']:.3f} | {avg_hits} | {descriptions.get(op, '')} |")
    
    report.append("\n## Threshold Analysis (AND Operation)\n")
    
    threshold_methods = ["geDIG-AND-Low", "geDIG-AND", "geDIG-AND-High"]
    report.append("| Threshold | IG | GED | Recall@5 | Hit Rate |")
    report.append("|-----------|-----|-----|----------|----------|")
    
    threshold_configs = {
        "geDIG-AND-Low": ("Low", "0.2", "-0.15"),
        "geDIG-AND": ("Medium", "0.3", "-0.1"),
        "geDIG-AND-High": ("High", "0.4", "-0.05")
    }
    
    for method in threshold_methods:
        if method in results:
            r = results[method]
            config = threshold_configs[method]
            hit_rate = r.get('hit_rate', 0) * 100
            report.append(f"| {config[0]} | {config[1]} | {config[2]} | "
                         f"{r['recall@5']:.3f} | {hit_rate:.1f}% |")
    
    report.append("\n## Insights\n")
    
    # Analyze which operation works best
    if best_op[0] == "geDIG-AND":
        report.append("- **AND operation** provides the best performance, enforcing strict criteria")
    elif best_op[0] == "geDIG-OR":
        report.append("- **OR operation** works best, allowing flexibility in matching")
    elif best_op[0] == "geDIG-WeightedAND":
        report.append("- **Weighted AND** balances strictness with flexibility")
    
    # Hit rate analysis
    and_hit_rate = results.get("geDIG-AND", {}).get("hit_rate", 0) * 100
    if and_hit_rate < 10:
        report.append(f"- AND operation is too restrictive ({and_hit_rate:.1f}% hit rate)")
    elif and_hit_rate > 50:
        report.append(f"- AND operation might be too permissive ({and_hit_rate:.1f}% hit rate)")
    
    with open(output_dir / 'LOGICAL_OPERATIONS_SUMMARY.md', 'w') as f:
        f.write('\n'.join(report))

def main():
    """Run the logical operations experiment"""
    
    print("="*60)
    print("geDIG LOGICAL OPERATIONS EXPERIMENT")
    print("="*60)
    
    # Run evaluation
    results = evaluate_logical_operations()
    
    # Create visualizations
    print("\nCreating visualizations...")
    visualize_logical_results(results)
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()