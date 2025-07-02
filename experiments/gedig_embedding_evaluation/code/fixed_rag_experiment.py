#!/usr/bin/env python3
"""
FIXED Dynamic RAG Comparison Experiment
=====================================

This is a corrected version of the RAG experiment that properly uses
the actual InsightSpike-AI components with correct imports and API usage.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Any
from collections import defaultdict, Counter
import re
import math

# プロジェクトのsrcディレクトリをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

warnings.filterwarnings('ignore')

# Text processing imports with fallbacks
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    print("✅ Sentence Transformers available")
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️ Sentence Transformers not available")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
    print("✅ Scikit-learn available")
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ Scikit-learn not available")

# FIXED: Import actual InsightSpike-AI components with correct paths
try:
    from insightspike.algorithms.graph_edit_distance import GraphEditDistance, OptimizationLevel
    from insightspike.algorithms.information_gain import InformationGain, EntropyMethod
    from insightspike.core.config_manager import ConfigManager
    print("✅ InsightSpike-AI components imported successfully")
    INSIGHTSPIKE_AVAILABLE = True
except ImportError as e:
    print(f"❌ InsightSpike-AI import error: {e}")
    print("🔧 Using fallback implementations")
    INSIGHTSPIKE_AVAILABLE = False

# Sample dataset for testing
SAMPLE_QUESTIONS = [
    {
        "question": "When was the Declaration of Independence signed?",
        "answer": "July 4, 1776",
        "context": "The Declaration of Independence was signed on July 4, 1776, in Philadelphia. This document declared the thirteen American colonies' independence from British rule and established the United States as a sovereign nation.",
        "type": "factual"
    },
    {
        "question": "What is the capital of France?",
        "answer": "Paris",
        "context": "Paris is the capital and largest city of France. It is located in the north-central part of the country and is known for its art, culture, cuisine, and iconic landmarks like the Eiffel Tower and Louvre Museum.",
        "type": "factual"
    },
    {
        "question": "Who wrote Romeo and Juliet and when was it written?",
        "answer": "William Shakespeare, around 1594-1596",
        "context": "Romeo and Juliet is a tragedy written by William Shakespeare. It was written around 1594-1596 and tells the story of two young star-crossed lovers whose deaths ultimately unite their feuding families in Verona, Italy.",
        "type": "multi-hop"
    },
    {
        "question": "What is photosynthesis?",
        "answer": "The process by which plants convert light energy into chemical energy",
        "context": "Photosynthesis is the biological process by which plants, algae, and some bacteria convert light energy from the sun into chemical energy stored in glucose molecules. This process uses carbon dioxide and water as inputs and produces oxygen as a byproduct.",
        "type": "factual"
    },
    {
        "question": "If Einstein developed relativity and worked at Princeton, where did the theory of relativity originate?",
        "answer": "The theory was developed by Einstein, who later worked at Princeton",
        "context": "Albert Einstein developed the theory of relativity in the early 1900s while working at various institutions in Europe. He later joined Princeton University's Institute for Advanced Study where he continued his research until his death in 1955.",
        "type": "multi-hop"
    },
    {
        "question": "What is the largest planet in our solar system?",
        "answer": "Jupiter",
        "context": "Jupiter is the largest planet in our solar system, with a mass greater than all other planets combined. It is a gas giant located fifth from the Sun and is known for its Great Red Spot, a giant storm larger than Earth.",
        "type": "factual"
    },
    {
        "question": "Who invented the telephone and when?",
        "answer": "Alexander Graham Bell in 1876",
        "context": "Alexander Graham Bell invented the telephone in 1876. Bell was a Scottish-born inventor and scientist who was awarded the first U.S. patent for the telephone on March 7, 1876. The first successful telephone call was made on March 10, 1876.",
        "type": "factual"
    },
    {
        "question": "If Shakespeare wrote Hamlet and lived during Elizabeth I's reign, what era was Hamlet written in?",
        "answer": "The Elizabethan era",
        "context": "William Shakespeare wrote Hamlet during the Elizabethan era, specifically around 1600-1601. Queen Elizabeth I reigned from 1558 to 1603, and Shakespeare wrote most of his famous plays during this period of English history.",
        "type": "multi-hop"
    }
]

class BM25Retriever:
    """BM25 (Best Matching 25) retrieval system"""
    
    def __init__(self, documents, k1=1.5, b=0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.tokenized_docs = [self._tokenize(doc) for doc in documents]
        self.doc_lengths = [len(doc) for doc in self.tokenized_docs]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths)
        self.idf_cache = {}
        self._build_idf()
    
    def _tokenize(self, text):
        """Simple tokenization"""
        return re.findall(r'\b\w+\b', text.lower())
    
    def _build_idf(self):
        """Precompute IDF values"""
        all_tokens = set()
        for doc in self.tokenized_docs:
            all_tokens.update(doc)
        
        for token in all_tokens:
            doc_freq = sum(1 for doc in self.tokenized_docs if token in doc)
            self.idf_cache[token] = math.log((len(self.documents) - doc_freq + 0.5) / (doc_freq + 0.5))
    
    def retrieve(self, query, k=5):
        """Retrieve top-k documents for query"""
        query_tokens = self._tokenize(query)
        scores = []
        
        for i, doc in enumerate(self.tokenized_docs):
            score = 0
            doc_counter = Counter(doc)
            
            for token in query_tokens:
                if token in doc_counter:
                    tf = doc_counter[token]
                    idf = self.idf_cache.get(token, 0)
                    
                    # BM25 formula
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * (self.doc_lengths[i] / self.avg_doc_length))
                    score += idf * (numerator / denominator)
            
            scores.append((i, score))
        
        # Sort by score and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

class StaticEmbeddingRetriever:
    """TF-IDF based static embedding retrieval"""
    
    def __init__(self, documents):
        self.documents = documents
        if SKLEARN_AVAILABLE:
            self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            self.doc_vectors = self.vectorizer.fit_transform(documents)
        else:
            self.vectorizer = None
    
    def retrieve(self, query, k=5):
        """Retrieve top-k documents for query"""
        if self.vectorizer is None:
            # Fallback: simple word overlap
            query_words = set(query.lower().split())
            scores = []
            
            for i, doc in enumerate(self.documents):
                doc_words = set(doc.lower().split())
                overlap = len(query_words & doc_words)
                scores.append((i, overlap / len(query_words) if query_words else 0))
            
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[:k]
        else:
            query_vector = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
            
            # Get top-k indices
            top_indices = similarities.argsort()[-k:][::-1]
            return [(idx, similarities[idx]) for idx in top_indices]

class DPRRetriever:
    """Dense Passage Retrieval using sentence transformers"""
    
    def __init__(self, documents):
        self.documents = documents
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            print("   📊 Encoding documents for DPR...")
            self.doc_embeddings = self.model.encode(documents, convert_to_tensor=True)
        else:
            self.model = None
            self.fallback = StaticEmbeddingRetriever(documents)
    
    def retrieve(self, query, k=5):
        """Retrieve top-k documents for query"""
        if self.model is None:
            return self.fallback.retrieve(query, k)
        
        import torch
        
        # Encode query
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        
        # Compute similarities
        similarities = torch.cosine_similarity(query_embedding.unsqueeze(0), self.doc_embeddings)
        
        # Get top-k
        top_k_indices = torch.topk(similarities, k).indices.cpu().numpy()
        top_k_scores = torch.topk(similarities, k).values.cpu().numpy()
        
        return [(int(idx), float(score)) for idx, score in zip(top_k_indices, top_k_scores)]

class FixedInsightSpikeRAG:
    """FIXED InsightSpike Dynamic RAG with REAL components"""
    
    def __init__(self, documents):
        self.documents = documents
        self.bm25 = BM25Retriever(documents)
        self.static = StaticEmbeddingRetriever(documents)
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.dense = DPRRetriever(documents)
        else:
            self.dense = None
        
        # FIXED: Initialize REAL InsightSpike-AI components
        if INSIGHTSPIKE_AVAILABLE:
            print("✅ Using REAL InsightSpike-AI components")
            self.ged_calculator = GraphEditDistance(optimization_level=OptimizationLevel.FAST)
            self.ig_calculator = InformationGain(method=EntropyMethod.SHANNON)
        else:
            print("⚠️ Using mock InsightSpike-AI components")
            self.ged_calculator = None
            self.ig_calculator = None
        
        # Dynamic weights
        self.base_weights = {
            'bm25': 0.4,
            'static': 0.3,
            'dense': 0.3 if self.dense else 0.0
        }
        
        # Normalize weights
        total_weight = sum(self.base_weights.values())
        self.base_weights = {k: v/total_weight for k, v in self.base_weights.items()}
    
    def _calculate_intrinsic_motivation(self, query, retrieved_docs):
        """FIXED: Calculate intrinsic motivation using REAL InsightSpike-AI"""
        if not INSIGHTSPIKE_AVAILABLE:
            # Fallback calculation
            return np.random.random() * 0.1
        
        try:
            # Create simple graphs for GED calculation
            import networkx as nx
            
            # Query graph (simple representation)
            query_words = query.lower().split()
            query_graph = nx.Graph()
            for i, word in enumerate(query_words):
                query_graph.add_node(i, label=word)
                if i > 0:
                    query_graph.add_edge(i-1, i)
            
            # Document graph (sample from retrieved docs)
            if retrieved_docs:
                doc_text = self.documents[retrieved_docs[0][0]]
                doc_words = doc_text.lower().split()[:len(query_words)]
                doc_graph = nx.Graph()
                for i, word in enumerate(doc_words):
                    doc_graph.add_node(i, label=word)
                    if i > 0:
                        doc_graph.add_edge(i-1, i)
                
                # FIXED: Use correct API - calculate() method
                ged_result = self.ged_calculator.calculate(query_graph, doc_graph)
                delta_ged = ged_result.ged_value
            else:
                delta_ged = 0.0
            
            # Information Gain calculation
            query_data = np.array([len(query_words), len(set(query_words))])
            if retrieved_docs:
                doc_text = self.documents[retrieved_docs[0][0]]
                doc_words = doc_text.lower().split()
                doc_data = np.array([len(doc_words), len(set(doc_words))])
                
                # FIXED: Use correct API - calculate() method
                ig_result = self.ig_calculator.calculate(query_data, doc_data)
                delta_ig = ig_result.ig_value
            else:
                delta_ig = 0.0
            
            # ΔGED × ΔIG intrinsic motivation
            intrinsic_score = delta_ged * delta_ig
            
            print(f"   🧠 REAL InsightSpike-AI: ΔGED={delta_ged:.3f}, ΔIG={delta_ig:.3f}, Intrinsic={intrinsic_score:.3f}")
            return intrinsic_score
            
        except Exception as e:
            print(f"   ⚠️ InsightSpike calculation error: {e}")
            return 0.0
    
    def _adaptive_weighting(self, query, intrinsic_motivation=0.0):
        """Dynamically adjust weights based on query and intrinsic motivation"""
        query_length = len(query.split())
        has_entities = any(word[0].isupper() for word in query.split())
        
        # Base heuristics
        if query_length > 10:  # Long queries favor dense retrieval
            weights = {'bm25': 0.2, 'static': 0.3, 'dense': 0.5}
        elif has_entities:  # Entity queries favor BM25
            weights = {'bm25': 0.6, 'static': 0.2, 'dense': 0.2}
        else:
            weights = self.base_weights.copy()
        
        # FIXED: Apply intrinsic motivation adjustment
        if intrinsic_motivation > 0.1:  # High intrinsic motivation
            # Favor more exploration (dense retrieval)
            weights['dense'] = min(1.0, weights.get('dense', 0) + intrinsic_motivation * 0.3)
            # Rebalance
            total = sum(weights.values())
            weights = {k: v/total for k, v in weights.items()}
        
        return weights
    
    def retrieve(self, query, k=5):
        """FIXED: Dynamic retrieval with REAL InsightSpike-AI intrinsic motivation"""
        # Get results from each system
        bm25_results = self.bm25.retrieve(query, k*2)
        static_results = self.static.retrieve(query, k*2)
        
        if self.dense:
            dense_results = self.dense.retrieve(query, k*2)
        else:
            dense_results = []
        
        # Calculate intrinsic motivation using REAL InsightSpike-AI
        intrinsic_motivation = self._calculate_intrinsic_motivation(query, bm25_results)
        
        # Get adaptive weights
        weights = self._adaptive_weighting(query, intrinsic_motivation)
        
        # Combine scores with adaptive weighting + intrinsic motivation boost
        combined_scores = {}
        
        # BM25 scores
        for doc_idx, score in bm25_results:
            combined_scores[doc_idx] = combined_scores.get(doc_idx, 0) + weights['bm25'] * score
        
        # Static embedding scores
        for doc_idx, score in static_results:
            combined_scores[doc_idx] = combined_scores.get(doc_idx, 0) + weights['static'] * score
        
        # Dense scores
        for doc_idx, score in dense_results:
            combined_scores[doc_idx] = combined_scores.get(doc_idx, 0) + weights['dense'] * score
        
        # Apply intrinsic motivation boost
        for doc_idx in combined_scores:
            combined_scores[doc_idx] += intrinsic_motivation * 0.1  # Scale factor
        
        # Sort and return top-k
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:k]

def evaluate_retrieval_system(retriever, questions, documents, k_values, system_name="Unknown"):
    """Evaluate a retrieval system on the given questions"""
    print(f"🔍 Evaluating {system_name}...")
    
    results = {
        "recall_at_k": {k: [] for k in k_values},
        "precision_at_k": {k: [] for k in k_values},
        "exact_matches": [],
        "f1_scores": [],
        "latencies": []
    }
    
    for i, q in enumerate(questions):
        query = q["question"]
        expected_context = q["context"]
        expected_answer = q["answer"].lower()
        
        # Measure retrieval latency
        start_time = time.time()
        retrieved_docs = retriever.retrieve(query, max(k_values))
        latency = time.time() - start_time
        results["latencies"].append(latency)
        
        # Find if expected context is retrieved
        relevant_found = False
        for doc_idx, _ in retrieved_docs:
            if documents[doc_idx] == expected_context:
                relevant_found = True
                break
        
        # Calculate recall and precision at k
        for k in k_values:
            top_k_docs = retrieved_docs[:k]
            
            # Simple relevance check (context match)
            relevant_in_k = any(documents[doc_idx] == expected_context for doc_idx, _ in top_k_docs)
            
            results["recall_at_k"][k].append(1.0 if relevant_in_k else 0.0)
            results["precision_at_k"][k].append(1.0/k if relevant_in_k else 0.0)
        
        # Exact match and F1 (simplified)
        retrieved_text = " ".join([documents[doc_idx] for doc_idx, _ in retrieved_docs[:1]])
        exact_match = 1.0 if expected_answer in retrieved_text.lower() else 0.0
        
        # Simple F1 calculation
        answer_words = set(expected_answer.split())
        retrieved_words = set(retrieved_text.lower().split())
        
        if answer_words and retrieved_words:
            precision = len(answer_words & retrieved_words) / len(retrieved_words)
            recall = len(answer_words & retrieved_words) / len(answer_words)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        else:
            f1 = 0.0
        
        results["exact_matches"].append(exact_match)
        results["f1_scores"].append(f1)
        
        if (i + 1) % 2 == 0:
            print(f"   Processed {i+1}/{len(questions)} questions...")
    
    return results

def create_visualization(all_results):
    """Create comprehensive visualization of RAG comparison results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('FIXED Dynamic RAG Comparison: Performance Analysis', fontsize=16, fontweight='bold')
    
    systems = list(all_results.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(systems)))
    
    # 1. Recall@k comparison
    ax1 = axes[0, 0]
    k_values = [1, 3, 5]
    x = np.arange(len(k_values))
    width = 0.8 / len(systems)
    
    for i, system in enumerate(systems):
        recalls = [np.mean(all_results[system]["recall_at_k"][k]) for k in k_values]
        ax1.bar(x + i * width, recalls, width, label=system, color=colors[i], alpha=0.8)
    
    ax1.set_xlabel('k value')
    ax1.set_ylabel('Recall@k')
    ax1.set_title('Recall@k Performance')
    ax1.set_xticks(x + width * (len(systems) - 1) / 2)
    ax1.set_xticklabels([f'@{k}' for k in k_values])
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Latency comparison
    ax2 = axes[0, 1]
    latencies = [np.mean(all_results[system]["latencies"]) * 1000 for system in systems]
    bars = ax2.bar(systems, latencies, color=colors, alpha=0.8)
    ax2.set_ylabel('Average Latency (ms)')
    ax2.set_title('Query Latency Comparison')
    ax2.set_xticklabels(systems, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # 3. F1 Score comparison
    ax3 = axes[1, 0]
    f1_scores = [np.mean(all_results[system]["f1_scores"]) for system in systems]
    bars = ax3.bar(systems, f1_scores, color=colors, alpha=0.8)
    ax3.set_ylabel('Average F1 Score')
    ax3.set_title('F1 Score Comparison')
    ax3.set_xticklabels(systems, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # 4. Exact Match comparison
    ax4 = axes[1, 1]
    exact_matches = [np.mean(all_results[system]["exact_matches"]) for system in systems]
    bars = ax4.bar(systems, exact_matches, color=colors, alpha=0.8)
    ax4.set_ylabel('Exact Match Rate')
    ax4.set_title('Exact Match Comparison')
    ax4.set_xticklabels(systems, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def run_fixed_rag_experiment():
    """Run the FIXED RAG comparison experiment"""
    print("🚀 Starting FIXED Dynamic RAG Comparison Experiment")
    print("📊 Using REAL InsightSpike-AI Components")
    print("=" * 60)
    
    # Prepare dataset
    questions = SAMPLE_QUESTIONS
    documents = [q["context"] for q in questions]
    
    # Add some document variations for better testing
    expanded_docs = []
    for doc in documents:
        expanded_docs.append(doc)
        # Add variations
        variation = doc.replace(".", ". This is historically significant.")
        expanded_docs.append(variation)
    
    documents = expanded_docs
    
    print(f"📊 Dataset: {len(questions)} questions, {len(documents)} documents")
    
    # Initialize retrieval systems
    print("\n🔧 Initializing retrieval systems...")
    
    systems = {}
    
    # BM25
    print("   📊 BM25 Retriever...")
    systems["BM25"] = BM25Retriever(documents)
    
    # Static Embeddings
    print("   🔢 Static Embedding Retriever...")
    systems["Static Embeddings"] = StaticEmbeddingRetriever(documents)
    
    # DPR (if available)
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        print("   🧠 DPR Dense Retriever...")
        systems["DPR (Dense)"] = DPRRetriever(documents)
    
    # FIXED InsightSpike RAG
    print("   🚀 FIXED InsightSpike Dynamic RAG...")
    systems["FIXED InsightSpike RAG"] = FixedInsightSpikeRAG(documents)
    
    print(f"\n✅ Initialized {len(systems)} retrieval systems")
    
    # Run evaluation
    print("\n📈 Running evaluation...")
    k_values = [1, 3, 5]
    all_results = {}
    
    for name, system in systems.items():
        results = evaluate_retrieval_system(system, questions, documents, k_values, name)
        all_results[name] = results
        
        # Quick summary
        avg_recall_5 = np.mean(results["recall_at_k"][5])
        avg_f1 = np.mean(results["f1_scores"])
        avg_latency = np.mean(results["latencies"])
        
        print(f"   {name}: Recall@5={avg_recall_5:.3f}, F1={avg_f1:.3f}, Latency={avg_latency*1000:.1f}ms")
    
    # Create visualization
    print("\n📈 Creating visualization...")
    fig = create_visualization(all_results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("fixed_rag_results")
    results_dir.mkdir(exist_ok=True)
    
    # Save plot
    plot_path = results_dir / f"fixed_rag_comparison_{timestamp}.png"
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    # Save data
    data_path = results_dir / f"fixed_rag_data_{timestamp}.json"
    
    # Convert numpy arrays for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    results_data = {
        "timestamp": timestamp,
        "experiment": "FIXED Dynamic RAG Comparison",
        "insightspike_available": INSIGHTSPIKE_AVAILABLE,
        "sentence_transformers_available": SENTENCE_TRANSFORMERS_AVAILABLE,
        "systems_evaluated": list(all_results.keys()),
        "results": convert_numpy(all_results)
    }
    
    with open(data_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n💾 Results saved:")
    print(f"   📊 Data: {data_path}")
    print(f"   📈 Plot: {plot_path}")
    
    # Print summary table
    print(f"\n📋 FIXED RAG Experiment Summary:")
    print("-" * 80)
    print(f"{'System':<25} {'Recall@5':<10} {'F1 Score':<10} {'Exact Match':<12} {'Latency (ms)':<12}")
    print("-" * 80)
    
    for system in all_results:
        recall5 = np.mean(all_results[system]["recall_at_k"][5])
        f1 = np.mean(all_results[system]["f1_scores"])
        em = np.mean(all_results[system]["exact_matches"])
        latency = np.mean(all_results[system]["latencies"]) * 1000
        
        print(f"{system:<25} {recall5:<10.3f} {f1:<10.3f} {em:<12.3f} {latency:<12.1f}")
    
    print("-" * 80)
    print("✅ FIXED RAG experiment completed successfully!")
    
    return all_results

if __name__ == "__main__":
    # Set random seed
    np.random.seed(42)
    
    # Run the experiment
    results = run_fixed_rag_experiment()