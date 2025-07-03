#!/usr/bin/env python3
"""
geDIG Intrinsic Reward Threshold & Memory Efficiency Experiment
==============================================================

This experiment focuses on:
1. Testing different intrinsic reward thresholds for geDIG
2. Evaluating memory compression ratios for RAG
3. Finding optimal threshold settings for performance
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

class SimpleGeDIG:
    """Simplified geDIG with configurable intrinsic reward thresholds"""
    
    def __init__(self, ig_threshold=0.3, ged_threshold=-0.1, name="geDIG"):
        self.ig_threshold = ig_threshold
        self.ged_threshold = ged_threshold
        self.name = name
        
    def compute_similarity(self, query, document):
        """Compute geDIG similarity with threshold activation"""
        # Simple IG: word overlap ratio
        q_words = set(query.lower().split())
        d_words = set(document.lower().split())
        overlap = len(q_words & d_words)
        total = len(q_words | d_words)
        ig_score = overlap / (total + 1e-8)
        
        # Simple GED: structural difference (negative is better)
        length_diff = abs(len(q_words) - len(d_words))
        ged_score = -length_diff / (len(q_words) + len(d_words) + 1e-8)
        
        # Apply thresholds
        ig_activated = 1.0 if ig_score > self.ig_threshold else 0.5
        ged_activated = 1.0 if ged_score > self.ged_threshold else 0.5
        
        # Combine with intrinsic boost
        base_score = 0.7 * ig_score + 0.3 * (-ged_score)
        intrinsic_boost = ig_activated * ged_activated
        
        return base_score * intrinsic_boost

def create_test_dataset(n_questions=100):
    """Create a simple test dataset"""
    questions = []
    documents = []
    
    topics = ["Python programming", "Machine learning", "Data science", 
              "Neural networks", "Natural language processing"]
    
    for i in range(n_questions):
        topic = topics[i % len(topics)]
        questions.append(f"What is {topic}?")
        documents.append(f"{topic} is a field of study that involves various techniques and methods.")
        
        # Add some noise documents
        for j in range(4):
            noise_topic = topics[(i + j + 1) % len(topics)]
            documents.append(f"{noise_topic} is different from {topic} in many ways.")
    
    return questions, documents

def evaluate_threshold_settings():
    """Evaluate different threshold settings"""
    
    print("Loading data and models...")
    questions, documents = create_test_dataset(100)
    
    # Initialize methods
    methods = {
        "TF-IDF": TfidfVectorizer(max_features=1000),
        "Sentence-BERT": SentenceTransformer('all-MiniLM-L6-v2'),
        "geDIG-Low": SimpleGeDIG(ig_threshold=0.1, ged_threshold=-0.2, name="geDIG-Low"),
        "geDIG-Medium": SimpleGeDIG(ig_threshold=0.3, ged_threshold=-0.1, name="geDIG-Medium"),
        "geDIG-High": SimpleGeDIG(ig_threshold=0.5, ged_threshold=-0.05, name="geDIG-High"),
        "geDIG-VeryHigh": SimpleGeDIG(ig_threshold=0.7, ged_threshold=-0.02, name="geDIG-VeryHigh"),
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
        
        recalls = []
        latencies = []
        memory_usage = 0
        
        for i, query in enumerate(questions):
            start_time = time.time()
            
            if method_name == "TF-IDF":
                query_vec = method.transform([query])
                similarities = cosine_similarity(query_vec, tfidf_docs).flatten()
                memory_usage = tfidf_docs.data.nbytes / 1024  # KB
                
            elif method_name == "Sentence-BERT":
                query_emb = method.encode([query])
                similarities = cosine_similarity(query_emb, sbert_docs).flatten()
                memory_usage = sbert_docs.nbytes / 1024  # KB
                
            else:  # geDIG variants
                similarities = []
                for doc in documents:
                    sim = method.compute_similarity(query, doc)
                    similarities.append(sim)
                similarities = np.array(similarities)
                memory_usage = len(str(documents)) / 1024  # Simple text size
            
            latency = (time.time() - start_time) * 1000  # ms
            latencies.append(latency)
            
            # Calculate recall@5 (document i*5 is relevant to question i)
            relevant_doc = i * 5
            top_5 = np.argsort(similarities)[::-1][:5]
            recall = 1 if relevant_doc in top_5 else 0
            recalls.append(recall)
        
        # Store results
        results[method_name]["recall@5"] = np.mean(recalls)
        results[method_name]["latency_ms"] = np.mean(latencies)
        results[method_name]["memory_kb"] = memory_usage
        results[method_name]["compression_ratio"] = len(documents) * 100 / memory_usage
        results[method_name]["efficiency_score"] = results[method_name]["recall@5"] * results[method_name]["compression_ratio"]
        
        print(f"  Recall@5: {results[method_name]['recall@5']:.3f}")
        print(f"  Latency: {results[method_name]['latency_ms']:.1f}ms")
        print(f"  Memory: {results[method_name]['memory_kb']:.1f}KB")
        print(f"  Compression: {results[method_name]['compression_ratio']:.1f} docs/KB")
    
    return results

def visualize_results(results):
    """Create visualizations for the results"""
    
    output_dir = Path("results_intrinsic_threshold")
    output_dir.mkdir(exist_ok=True)
    
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    methods = list(results.keys())
    
    # 1. Performance comparison
    ax = axes[0, 0]
    recalls = [results[m]["recall@5"] for m in methods]
    colors = ['blue', 'green', 'orange', 'orange', 'orange', 'orange']
    ax.bar(methods, recalls, color=colors)
    ax.set_ylabel('Recall@5')
    ax.set_title('Retrieval Performance')
    ax.set_xticklabels(methods, rotation=45, ha='right')
    
    # 2. Latency comparison
    ax = axes[0, 1]
    latencies = [results[m]["latency_ms"] for m in methods]
    ax.bar(methods, latencies, color=colors)
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Query Processing Speed')
    ax.set_yscale('log')
    ax.set_xticklabels(methods, rotation=45, ha='right')
    
    # 3. Memory efficiency
    ax = axes[0, 2]
    compressions = [results[m]["compression_ratio"] for m in methods]
    ax.bar(methods, compressions, color=colors)
    ax.set_ylabel('Compression Ratio (docs/KB)')
    ax.set_title('Memory Efficiency')
    ax.set_yscale('log')
    ax.set_xticklabels(methods, rotation=45, ha='right')
    
    # 4. Performance vs Speed trade-off
    ax = axes[1, 0]
    for i, method in enumerate(methods):
        perf = results[method]["recall@5"]
        speed = 1000 / results[method]["latency_ms"]
        ax.scatter(speed, perf, s=100, color=colors[i], label=method)
    ax.set_xlabel('Speed (queries/second)')
    ax.set_ylabel('Performance (Recall@5)')
    ax.set_title('Performance vs Speed Trade-off')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xscale('log')
    
    # 5. Memory-Performance trade-off
    ax = axes[1, 1]
    for i, method in enumerate(methods):
        perf = results[method]["recall@5"]
        mem_eff = results[method]["efficiency_score"]
        ax.scatter(mem_eff, perf, s=100, color=colors[i], label=method)
    ax.set_xlabel('Memory Efficiency Score')
    ax.set_ylabel('Performance (Recall@5)')
    ax.set_title('Memory-Performance Trade-off')
    
    # 6. Threshold sensitivity (geDIG only)
    ax = axes[1, 2]
    gedig_methods = [m for m in methods if m.startswith("geDIG")]
    gedig_perfs = [results[m]["recall@5"] for m in gedig_methods]
    threshold_labels = ["Low\n(0.1,-0.2)", "Medium\n(0.3,-0.1)", "High\n(0.5,-0.05)", "VeryHigh\n(0.7,-0.02)"]
    ax.plot(threshold_labels, gedig_perfs, 'o-', color='orange', linewidth=2, markersize=8)
    ax.set_xlabel('Threshold Settings (IG, GED)')
    ax.set_ylabel('Recall@5')
    ax.set_title('geDIG Performance vs Threshold Settings')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'intrinsic_threshold_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results
    with open(output_dir / 'intrinsic_threshold_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create summary report
    create_summary_report(results, output_dir)
    
    print(f"\nResults saved to {output_dir}")

def create_summary_report(results, output_dir):
    """Create a markdown summary report"""
    
    report = ["# geDIG Intrinsic Threshold & Memory Efficiency Analysis\n"]
    report.append(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"**Random Seed**: {RANDOM_SEED}\n")
    
    report.append("\n## Key Findings\n")
    
    # Find best geDIG threshold
    gedig_methods = [(m, results[m]["recall@5"]) for m in results if m.startswith("geDIG")]
    best_gedig = max(gedig_methods, key=lambda x: x[1])
    report.append(f"1. **Best geDIG Threshold**: {best_gedig[0]} (Recall@5={best_gedig[1]:.3f})")
    
    # Memory efficiency champion
    best_compression = max(results.items(), key=lambda x: x[1]["compression_ratio"])
    report.append(f"2. **Best Memory Efficiency**: {best_compression[0]} ({best_compression[1]['compression_ratio']:.1f} docs/KB)")
    
    # Overall winner
    best_overall = max(results.items(), key=lambda x: x[1]["efficiency_score"])
    report.append(f"3. **Best Overall (Performance×Compression)**: {best_overall[0]} (Score={best_overall[1]['efficiency_score']:.1f})")
    
    report.append("\n## Performance Summary\n")
    report.append("| Method | Recall@5 | Latency (ms) | Memory (KB) | Compression | Efficiency |")
    report.append("|--------|----------|--------------|-------------|-------------|------------|")
    
    for method in results:
        r = results[method]
        report.append(f"| {method} | {r['recall@5']:.3f} | {r['latency_ms']:.1f} | "
                     f"{r['memory_kb']:.1f} | {r['compression_ratio']:.1f} | {r['efficiency_score']:.1f} |")
    
    report.append("\n## Threshold Analysis\n")
    report.append("The experiment tested various intrinsic reward thresholds:")
    report.append("- **Low** (IG=0.1, GED=-0.2): More permissive, accepts more matches")
    report.append("- **Medium** (IG=0.3, GED=-0.1): Balanced approach")
    report.append("- **High** (IG=0.5, GED=-0.05): More selective")
    report.append("- **VeryHigh** (IG=0.7, GED=-0.02): Very selective, only high-quality matches")
    
    # Threshold trend
    gedig_perfs = [results[m]["recall@5"] for m in ["geDIG-Low", "geDIG-Medium", "geDIG-High", "geDIG-VeryHigh"]]
    if gedig_perfs[0] > gedig_perfs[-1]:
        report.append("\n**Finding**: Lower thresholds improve recall by being more inclusive.")
    else:
        report.append("\n**Finding**: Higher thresholds improve precision but may hurt recall.")
    
    report.append("\n## Memory Efficiency Insights\n")
    report.append(f"- TF-IDF: Sparse representation achieves {results['TF-IDF']['compression_ratio']:.1f} docs/KB")
    report.append(f"- Sentence-BERT: Dense embeddings use {results['Sentence-BERT']['memory_kb']:.1f}KB")
    report.append(f"- geDIG variants: Text-based storage achieves {results['geDIG-Medium']['compression_ratio']:.1f} docs/KB")
    
    with open(output_dir / 'THRESHOLD_ANALYSIS_SUMMARY.md', 'w') as f:
        f.write('\n'.join(report))

def main():
    """Run the intrinsic threshold experiment"""
    
    print("="*60)
    print("geDIG INTRINSIC THRESHOLD & MEMORY EFFICIENCY EXPERIMENT")
    print("="*60)
    
    # Run evaluation
    results = evaluate_threshold_settings()
    
    # Create visualizations
    print("\nCreating visualizations...")
    visualize_results(results)
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()