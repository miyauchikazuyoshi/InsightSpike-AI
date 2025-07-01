#!/usr/bin/env python3
"""
Dynamic RAG Comparison Experiment
================================

This experiment compares InsightSpike-AI's dynamic RAG construction capabilities
against existing methods using standard QA benchmarks.

Experimental Design:
- Datasets: NaturalQuestions, HotpotQA  
- Baselines: BM25, static embeddings, DPR
- Metrics: Recall@k, EM/F1, inference latency
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import re

warnings.filterwarnings('ignore')

# Text processing and embedding imports
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Dataset simulation (since we can't download full datasets in Colab easily)
SAMPLE_NATURAL_QUESTIONS = [
    {
        "question": "When was the Declaration of Independence signed?",
        "answer": "July 4, 1776",
        "context": "The Declaration of Independence was signed on July 4, 1776, in Philadelphia. This document declared the thirteen American colonies' independence from British rule.",
        "type": "factual"
    },
    {
        "question": "Who wrote Romeo and Juliet?",
        "answer": "William Shakespeare",
        "context": "Romeo and Juliet is a tragedy written by William Shakespeare early in his career about two young Italian star-crossed lovers whose deaths ultimately unite their feuding families.",
        "type": "factual"
    },
    {
        "question": "What is the capital of France?",
        "answer": "Paris",
        "context": "Paris is the capital and most populous city of France. Located in the north-central part of the country, Paris is a major European city and a global center for art, fashion, gastronomy and culture.",
        "type": "factual"
    },
    {
        "question": "How many continents are there?",
        "answer": "Seven",
        "context": "There are seven continents on Earth: Asia, Africa, North America, South America, Antarctica, Europe, and Australia/Oceania. This is the most widely accepted model.",
        "type": "factual"
    },
    {
        "question": "What year did World War II end?",
        "answer": "1945",
        "context": "World War II ended in 1945. The war in Europe ended on May 8, 1945 (VE Day), and the war in the Pacific ended on September 2, 1945 (VJ Day) after the atomic bombings of Hiroshima and Nagasaki.",
        "type": "factual"
    }
]

SAMPLE_HOTPOT_QA = [
    {
        "question": "Which film won the Academy Award for Best Picture in the year that the first iPhone was released?",
        "answer": "No Country for Old Men",
        "context": "The first iPhone was released in 2007. The Academy Award for Best Picture in 2007 (for films from 2007) was won by 'No Country for Old Men', directed by the Coen Brothers.",
        "type": "multi-hop",
        "supporting_facts": ["iPhone was released in 2007", "No Country for Old Men won Best Picture for 2007"]
    },
    {
        "question": "What is the population of the country where the Eiffel Tower is located?",
        "answer": "Approximately 67 million",
        "context": "The Eiffel Tower is located in Paris, France. France has a population of approximately 67 million people as of recent estimates.",
        "type": "multi-hop",
        "supporting_facts": ["Eiffel Tower is in France", "France has 67 million people"]
    },
    {
        "question": "Who directed the movie that features the song 'My Heart Will Go On'?",
        "answer": "James Cameron",
        "context": "The song 'My Heart Will Go On' is the main theme song from the movie Titanic. Titanic was directed by James Cameron and released in 1997.",
        "type": "multi-hop",
        "supporting_facts": ["My Heart Will Go On is from Titanic", "Titanic was directed by James Cameron"]
    }
]

class BM25Retriever:
    """BM25 baseline retriever"""
    
    def __init__(self, documents: List[str], k1=1.5, b=0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.avgdl = 0
        
        self._build_index()
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        return re.findall(r'\b\w+\b', text.lower())
    
    def _build_index(self):
        """Build BM25 index"""
        # Tokenize documents
        tokenized_docs = [self._tokenize(doc) for doc in self.documents]
        
        # Calculate document lengths and average length
        self.doc_len = [len(doc) for doc in tokenized_docs]
        self.avgdl = sum(self.doc_len) / len(self.doc_len)
        
        # Calculate document frequencies
        df = defaultdict(int)
        for doc in tokenized_docs:
            for word in set(doc):
                df[word] += 1
        
        # Calculate IDF
        N = len(self.documents)
        for word, freq in df.items():
            self.idf[word] = np.log((N - freq + 0.5) / (freq + 0.5))
        
        # Calculate term frequencies for each document
        for doc in tokenized_docs:
            freqs = defaultdict(int)
            for word in doc:
                freqs[word] += 1
            self.doc_freqs.append(freqs)
    
    def retrieve(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
        """Retrieve top-k documents for query"""
        query_tokens = self._tokenize(query)
        scores = []
        
        for i, doc_freq in enumerate(self.doc_freqs):
            score = 0
            for token in query_tokens:
                if token in doc_freq:
                    tf = doc_freq[token]
                    idf = self.idf.get(token, 0)
                    
                    # BM25 formula
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                    score += idf * numerator / denominator
            
            scores.append((i, score))
        
        # Sort by score and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

class StaticEmbeddingRetriever:
    """Static embedding baseline using pre-computed embeddings"""
    
    def __init__(self, documents: List[str]):
        self.documents = documents
        
        if SKLEARN_AVAILABLE:
            # Use TF-IDF as a simple embedding baseline
            self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            self.doc_embeddings = self.vectorizer.fit_transform(documents)
        else:
            # Fallback to simple word counting
            self.doc_embeddings = None
            print("⚠️ Sklearn not available, using simplified embeddings")
    
    def retrieve(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
        """Retrieve top-k documents for query"""
        if self.doc_embeddings is not None and SKLEARN_AVAILABLE:
            query_embedding = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_embedding, self.doc_embeddings).flatten()
            
            # Get top-k indices
            top_indices = np.argsort(similarities)[::-1][:k]
            return [(idx, similarities[idx]) for idx in top_indices]
        else:
            # Fallback: simple word overlap
            query_words = set(query.lower().split())
            scores = []
            
            for i, doc in enumerate(self.documents):
                doc_words = set(doc.lower().split())
                overlap = len(query_words.intersection(doc_words))
                scores.append((i, overlap / max(len(query_words), 1)))
            
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[:k]

class DPRRetriever:
    """Dense Passage Retrieval simulation"""
    
    def __init__(self, documents: List[str]):
        self.documents = documents
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            # Use sentence transformers as DPR simulation
            # Safe loading with device specification to avoid meta tensor issues
            device = 'cpu'  # Force CPU to avoid GPU meta tensor issues in Colab
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
                print("📦 Loading DPR-style embeddings...")
                self.doc_embeddings = self.model.encode(documents, convert_to_numpy=True)
            except Exception as e:
                print(f"⚠️ SentenceTransformer loading failed: {e}")
                print("📦 Falling back to TF-IDF...")
                if SKLEARN_AVAILABLE:
                    self.vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
                    self.doc_embeddings = self.vectorizer.fit_transform(documents).toarray()
                    self.model = None
                else:
                    self.doc_embeddings = None
                    self.model = None
        else:
            print("⚠️ Sentence transformers not available, using TF-IDF fallback")
            if SKLEARN_AVAILABLE:
                self.vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
                self.doc_embeddings = self.vectorizer.fit_transform(documents).toarray()
            else:
                self.doc_embeddings = None
    
    def retrieve(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
        """Retrieve top-k documents for query"""
        if self.doc_embeddings is not None:
            if SENTENCE_TRANSFORMERS_AVAILABLE and self.model is not None:
                try:
                    query_embedding = self.model.encode([query], convert_to_numpy=True)
                    similarities = cosine_similarity(query_embedding, self.doc_embeddings).flatten()
                except Exception as e:
                    print(f"⚠️ SentenceTransformer encoding failed: {e}")
                    # Fallback to simple string matching
                    similarities = np.array([0.1 * len(set(query.lower().split()).intersection(set(doc.lower().split()))) 
                                           for doc in self.documents])
            else:
                query_embedding = self.vectorizer.transform([query]).toarray()
                similarities = cosine_similarity(query_embedding, self.doc_embeddings).flatten()
            
            top_indices = np.argsort(similarities)[::-1][:k]
            return [(idx, similarities[idx]) for idx in top_indices]
        else:
            # Fallback
            return [(i, 0.1) for i in range(min(k, len(self.documents)))]

class InsightSpikeRAG:
    """InsightSpike-AI Dynamic RAG System (Simplified)"""
    
    def __init__(self, documents: List[str]):
        self.documents = documents
        self.retrieval_history = []
        self.dynamic_weights = defaultdict(float)
        self.context_memory = []
        
        # Initialize with multiple retrieval strategies
        self.bm25 = BM25Retriever(documents)
        self.static_emb = StaticEmbeddingRetriever(documents)
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.dense = DPRRetriever(documents)
        else:
            self.dense = None
    
    def _calculate_dynamic_weights(self, query: str) -> Dict[str, float]:
        """Calculate dynamic weights based on query characteristics and history"""
        weights = {"bm25": 0.4, "static": 0.3, "dense": 0.3}
        
        # Adjust based on query length (longer queries might benefit from semantic search)
        query_len = len(query.split())
        if query_len > 10:
            weights["dense"] += 0.1
            weights["bm25"] -= 0.1
        
        # Adjust based on question type detection
        if any(word in query.lower() for word in ["what", "when", "where", "who"]):
            weights["bm25"] += 0.1  # Factual questions benefit from BM25
            weights["dense"] -= 0.1
        elif any(word in query.lower() for word in ["how", "why", "explain"]):
            weights["dense"] += 0.1  # Complex questions benefit from dense retrieval
            weights["bm25"] -= 0.1
        
        # Ensure weights sum to 1
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        
        return weights
    
    def _calculate_ged_ig_score(self, retrieved_docs: List[int], query: str) -> float:
        """Calculate intrinsic motivation score (ΔGED × ΔIG) for retrieved documents"""
        # Simplified implementation
        if not self.retrieval_history:
            return 1.0  # High score for first retrieval
        
        # Calculate novelty (ΔIG simulation)
        previous_docs = set()
        for prev_retrieval in self.retrieval_history[-5:]:  # Last 5 retrievals
            previous_docs.update(prev_retrieval)
        
        current_docs = set(retrieved_docs)
        novelty = len(current_docs - previous_docs) / max(len(current_docs), 1)
        
        # Calculate diversity (ΔGED simulation)
        diversity = len(set(retrieved_docs)) / max(len(retrieved_docs), 1)
        
        # Combine for intrinsic motivation score
        im_score = novelty * diversity
        return im_score
    
    def retrieve(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
        """Dynamic retrieval with intrinsic motivation"""
        start_time = time.time()
        
        # Calculate dynamic weights
        weights = self._calculate_dynamic_weights(query)
        
        # Get results from each retriever
        bm25_results = self.bm25.retrieve(query, k*2)  # Get more candidates
        static_results = self.static_emb.retrieve(query, k*2)
        
        if self.dense:
            dense_results = self.dense.retrieve(query, k*2)
        else:
            dense_results = []
        
        # Combine results with dynamic weighting
        combined_scores = defaultdict(float)
        
        for idx, score in bm25_results:
            combined_scores[idx] += weights["bm25"] * score
        
        for idx, score in static_results:
            combined_scores[idx] += weights["static"] * score
        
        for idx, score in dense_results:
            combined_scores[idx] += weights["dense"] * score
        
        # Apply intrinsic motivation boost
        top_candidates = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:k*2]
        candidate_docs = [idx for idx, _ in top_candidates]
        
        im_score = self._calculate_ged_ig_score(candidate_docs, query)
        
        # Boost scores with intrinsic motivation
        for idx in candidate_docs:
            combined_scores[idx] *= (1 + 0.1 * im_score)  # 10% boost weighted by IM
        
        # Final ranking
        final_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        # Update history
        retrieved_docs = [idx for idx, _ in final_results]
        self.retrieval_history.append(retrieved_docs)
        
        # Update context memory
        self.context_memory.append({
            "query": query,
            "retrieved_docs": retrieved_docs,
            "weights": weights,
            "im_score": im_score,
            "latency": time.time() - start_time
        })
        
        return final_results

def evaluate_retrieval_system(retriever, questions: List[Dict], documents: List[str], 
                            k_values: List[int] = [1, 3, 5]) -> Dict[str, Any]:
    """Evaluate a retrieval system on a set of questions"""
    
    results = {
        "recall_at_k": {k: [] for k in k_values},
        "precision_at_k": {k: [] for k in k_values},
        "latencies": [],
        "exact_matches": [],
        "f1_scores": []
    }
    
    for question_data in questions:
        question = question_data["question"]
        correct_answer = question_data["answer"].lower()
        
        # Measure retrieval latency
        start_time = time.time()
        retrieved = retriever.retrieve(question, k=max(k_values))
        latency = time.time() - start_time
        results["latencies"].append(latency)
        
        # Get retrieved documents
        retrieved_docs = [documents[idx] for idx, _ in retrieved]
        
        # Calculate metrics for different k values
        for k in k_values:
            top_k_docs = retrieved_docs[:k]
            
            # Recall@k: Did we retrieve the relevant document?
            relevant_found = any(correct_answer in doc.lower() for doc in top_k_docs)
            results["recall_at_k"][k].append(1.0 if relevant_found else 0.0)
            
            # Precision@k: What fraction of retrieved docs are relevant?
            relevant_count = sum(1 for doc in top_k_docs if correct_answer in doc.lower())
            precision = relevant_count / k if k > 0 else 0.0
            results["precision_at_k"][k].append(precision)
        
        # Exact Match (EM): Does the top document contain the exact answer?
        if retrieved_docs:
            top_doc = retrieved_docs[0]
            em = 1.0 if correct_answer in top_doc.lower() else 0.0
            results["exact_matches"].append(em)
        else:
            results["exact_matches"].append(0.0)
        
        # F1 Score: Token-level F1 between answer and top retrieved document
        if retrieved_docs:
            f1 = calculate_f1_score(correct_answer, retrieved_docs[0])
            results["f1_scores"].append(f1)
        else:
            results["f1_scores"].append(0.0)
    
    return results

def calculate_f1_score(answer: str, text: str) -> float:
    """Calculate F1 score between answer and text at token level"""
    answer_tokens = set(answer.lower().split())
    text_tokens = set(text.lower().split())
    
    if not answer_tokens:
        return 0.0
    
    intersection = answer_tokens.intersection(text_tokens)
    
    if not intersection:
        return 0.0
    
    precision = len(intersection) / len(text_tokens) if text_tokens else 0.0
    recall = len(intersection) / len(answer_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def create_expanded_dataset():
    """Create an expanded dataset for more robust evaluation"""
    
    # Combine and expand the sample datasets
    all_questions = SAMPLE_NATURAL_QUESTIONS + SAMPLE_HOTPOT_QA
    
    # Create a larger document corpus
    all_contexts = [q["context"] for q in all_questions]
    
    # Add additional relevant documents to make retrieval more challenging
    additional_docs = [
        "The United States Constitution was written in 1787 during the Constitutional Convention in Philadelphia.",
        "Shakespeare was born in 1564 in Stratford-upon-Avon and wrote approximately 37 plays during his career.",
        "London is the capital city of England and the United Kingdom, located on the Thames River.",
        "The Earth has seven continents and five oceans, covering a surface area of about 510 million square kilometers.",
        "World War I lasted from 1914 to 1918 and involved many of the world's great powers.",
        "The iPhone revolutionized the smartphone industry when Apple released it in 2007.",
        "France is located in Western Europe and is known for its rich culture, art, and cuisine.",
        "The Academy Awards, also known as the Oscars, are presented annually by the Academy of Motion Picture Arts and Sciences.",
        "Titanic was one of the highest-grossing films of all time when it was released in 1997.",
        "The Eiffel Tower was completed in 1889 and stands 324 meters tall in Paris.",
        # Add some distractors
        "The Great Wall of China was built over many centuries to protect against invasions.",
        "Albert Einstein developed the theory of relativity in the early 20th century.",
        "The Amazon rainforest is the largest tropical rainforest in the world.",
        "Basketball was invented in 1891 by James Naismith in Springfield, Massachusetts.",
        "The human brain contains approximately 86 billion neurons.",
    ]
    
    all_documents = all_contexts + additional_docs
    
    return all_questions, all_documents

def run_rag_comparison_experiment():
    """Run the complete RAG comparison experiment"""
    
    print("🚀 Starting Dynamic RAG Comparison Experiment")
    print("=" * 60)
    
    # Create expanded dataset
    questions, documents = create_expanded_dataset()
    print(f"📊 Dataset: {len(questions)} questions, {len(documents)} documents")
    
    # Initialize retrievers
    print("\n🔧 Initializing retrieval systems...")
    
    retrievers = {
        "BM25": BM25Retriever(documents),
        "Static Embeddings": StaticEmbeddingRetriever(documents),
        "InsightSpike Dynamic RAG": InsightSpikeRAG(documents)
    }
    
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        retrievers["DPR (Dense)"] = DPRRetriever(documents)
        print("✅ All retrievers initialized (including DPR)")
    else:
        print("⚠️ DPR not available, using fallback methods")
    
    # Run evaluation
    k_values = [1, 3, 5]
    all_results = {}
    
    print(f"\n📈 Running evaluations with k={k_values}")
    
    for name, retriever in retrievers.items():
        print(f"\n🔍 Evaluating {name}...")
        results = evaluate_retrieval_system(retriever, questions, documents, k_values)
        all_results[name] = results
        
        # Print quick summary
        avg_recall_5 = np.mean(results["recall_at_k"][5])
        avg_latency = np.mean(results["latencies"])
        avg_em = np.mean(results["exact_matches"])
        
        print(f"  Recall@5: {avg_recall_5:.3f}")
        print(f"  Exact Match: {avg_em:.3f}")
        print(f"  Avg Latency: {avg_latency*1000:.1f}ms")
    
    return all_results, questions, documents

def create_rag_visualization(results: Dict[str, Any], questions: List[Dict]):
    """Create comprehensive visualization of RAG comparison results"""
    
    # Set up plotting
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Dynamic RAG Comparison: InsightSpike-AI vs Baselines', fontsize=16)
    
    systems = list(results.keys())
    colors = sns.color_palette("husl", len(systems))
    
    # 1. Recall@k Comparison
    ax1 = axes[0, 0]
    k_values = [1, 3, 5]
    x = np.arange(len(k_values))
    width = 0.8 / len(systems)
    
    for i, system in enumerate(systems):
        recall_means = [np.mean(results[system]["recall_at_k"][k]) for k in k_values]
        ax1.bar(x + i * width, recall_means, width, label=system, color=colors[i], alpha=0.8)
    
    ax1.set_xlabel('k value')
    ax1.set_ylabel('Recall@k')
    ax1.set_title('Recall@k Performance')
    ax1.set_xticks(x + width * (len(systems) - 1) / 2)
    ax1.set_xticklabels([f'k={k}' for k in k_values])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Precision@k Comparison
    ax2 = axes[0, 1]
    for i, system in enumerate(systems):
        precision_means = [np.mean(results[system]["precision_at_k"][k]) for k in k_values]
        ax2.bar(x + i * width, precision_means, width, label=system, color=colors[i], alpha=0.8)
    
    ax2.set_xlabel('k value')
    ax2.set_ylabel('Precision@k')
    ax2.set_title('Precision@k Performance')
    ax2.set_xticks(x + width * (len(systems) - 1) / 2)
    ax2.set_xticklabels([f'k={k}' for k in k_values])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Latency Comparison
    ax3 = axes[0, 2]
    latency_data = []
    system_labels = []
    
    for system in systems:
        latencies = np.array(results[system]["latencies"]) * 1000  # Convert to ms
        latency_data.extend(latencies)
        system_labels.extend([system] * len(latencies))
    
    latency_df = pd.DataFrame({"System": system_labels, "Latency (ms)": latency_data})
    sns.boxplot(data=latency_df, x="System", y="Latency (ms)", ax=ax3)
    ax3.set_title('Inference Latency Comparison')
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
    
    # 4. Exact Match and F1 Scores
    ax4 = axes[1, 0]
    em_scores = [np.mean(results[system]["exact_matches"]) for system in systems]
    f1_scores = [np.mean(results[system]["f1_scores"]) for system in systems]
    
    x_pos = np.arange(len(systems))
    width = 0.35
    
    ax4.bar(x_pos - width/2, em_scores, width, label='Exact Match', alpha=0.8)
    ax4.bar(x_pos + width/2, f1_scores, width, label='F1 Score', alpha=0.8)
    
    ax4.set_xlabel('System')
    ax4.set_ylabel('Score')
    ax4.set_title('Exact Match vs F1 Score')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(systems, rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Performance by Question Type
    ax5 = axes[1, 1]
    
    # Separate factual vs multi-hop questions
    factual_questions = [q for q in questions if q.get("type") == "factual"]
    multihop_questions = [q for q in questions if q.get("type") == "multi-hop"]
    
    factual_performance = {}
    multihop_performance = {}
    
    for system in systems:
        # Calculate performance for factual questions
        factual_em = []
        multihop_em = []
        
        for i, q in enumerate(questions):
            em_score = results[system]["exact_matches"][i]
            if q.get("type") == "factual":
                factual_em.append(em_score)
            elif q.get("type") == "multi-hop":
                multihop_em.append(em_score)
        
        factual_performance[system] = np.mean(factual_em) if factual_em else 0
        multihop_performance[system] = np.mean(multihop_em) if multihop_em else 0
    
    x_pos = np.arange(len(systems))
    ax5.bar(x_pos - width/2, list(factual_performance.values()), width, 
            label='Factual Questions', alpha=0.8)
    ax5.bar(x_pos + width/2, list(multihop_performance.values()), width, 
            label='Multi-hop Questions', alpha=0.8)
    
    ax5.set_xlabel('System')
    ax5.set_ylabel('Exact Match Score')
    ax5.set_title('Performance by Question Type')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(systems, rotation=45)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. System Comparison Matrix
    ax6 = axes[1, 2]
    
    # Create a summary matrix
    metrics = ['Recall@5', 'Precision@5', 'Exact Match', 'F1 Score', 'Latency (inv)']
    comparison_matrix = []
    
    for system in systems:
        row = [
            np.mean(results[system]["recall_at_k"][5]),
            np.mean(results[system]["precision_at_k"][5]),
            np.mean(results[system]["exact_matches"]),
            np.mean(results[system]["f1_scores"]),
            1.0 / (np.mean(results[system]["latencies"]) + 0.001)  # Inverse latency (higher is better)
        ]
        comparison_matrix.append(row)
    
    # Normalize each metric to 0-1 scale
    comparison_matrix = np.array(comparison_matrix)
    for i in range(comparison_matrix.shape[1]):
        col = comparison_matrix[:, i]
        comparison_matrix[:, i] = (col - col.min()) / (col.max() - col.min() + 1e-8)
    
    im = ax6.imshow(comparison_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax6.set_title('Normalized Performance Matrix')
    ax6.set_xticks(range(len(metrics)))
    ax6.set_yticks(range(len(systems)))
    ax6.set_xticklabels(metrics, rotation=45)
    ax6.set_yticklabels(systems)
    
    # Add text annotations
    for i in range(len(systems)):
        for j in range(len(metrics)):
            text = ax6.text(j, i, f'{comparison_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontweight='bold')
    
    plt.colorbar(im, ax=ax6, label='Normalized Score')
    
    plt.tight_layout()
    return fig

def run_dynamic_rag_experiment():
    """Run the complete dynamic RAG comparison experiment"""
    
    print("🚀 Starting Dynamic RAG Comparison Experiment")
    print("=" * 70)
    
    # Setup experiment directories and data management
    print("📁 Setting up experiment directories...")
    experiment_dirs = setup_experiment_directories()
    
    # Run the main experiment with persistent data management
    results, questions, documents = run_rag_comparison_experiment_with_storage(experiment_dirs)
    
    # Save comprehensive results using the new data management system
    print("\n💾 Saving comprehensive experimental results...")
    results_file = save_experiment_results(results, experiment_dirs, "dynamic_rag_comparison")
    
    # Create visualizations
    print("\n📈 Creating comprehensive visualizations...")
    fig = create_rag_visualization(results, questions)
    
    # Save visualization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = experiment_dirs["results"] / f"rag_comparison_visualization_{timestamp}.png"
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Visualization saved: {plot_path}")
    
    # Prepare comprehensive results data
    results_data = {
        "timestamp": timestamp,
        "experiment_type": "dynamic_rag_comparison",
        "dataset_info": {
            "num_questions": len(questions),
            "num_documents": len(documents),
            "question_types": list(set(q.get("type", "unknown") for q in questions))
        },
        "systems_evaluated": list(results.keys()),
        "metrics": {
            "recall_at_k": [1, 3, 5],
            "precision_at_k": [1, 3, 5],
            "exact_match": True,
            "f1_score": True,
            "latency": True
        },
        "results": results
    }
    
    # Convert numpy arrays to lists for JSON serialization
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
    
    # Save JSON data
    data_path = experiment_dirs["results"] / f"rag_experimental_data_{timestamp}.json"
    with open(data_path, 'w') as f:
        json.dump(convert_numpy(results_data), f, indent=2)
    
    print(f"💾 Experimental data saved: {data_path}")
    
    # Print comprehensive summary
    print("\n📋 Experimental Summary:")
    print("-" * 70)
    print(f"{'System':<25} {'Recall@5':<10} {'EM':<8} {'F1':<8} {'Latency(ms)':<12}")
    print("-" * 70)
    
    for system in results.keys():
        recall_5 = np.mean(results[system]["recall_at_k"][5])
        em = np.mean(results[system]["exact_matches"])
        f1 = np.mean(results[system]["f1_scores"])
        latency = np.mean(results[system]["latencies"]) * 1000
        
        print(f"{system:<25} {recall_5:<10.3f} {em:<8.3f} {f1:<8.3f} {latency:<12.1f}")
    
    print("-" * 70)
    print("\n✅ Dynamic RAG Comparison Experiment Complete!")
    
    return {
        "results": results_data,
        "visualization": fig,
        "paths": {
            "plot": plot_path,
            "data": data_path
        }
    }

def setup_experiment_directories(base_dir="data/rag_experiments"):
    """Setup directory structure for RAG comparison experiments"""
    from pathlib import Path
    
    # Create experiment directories
    dirs = {
        "base": Path(base_dir),
        "baselines": Path(base_dir) / "baselines",
        "insightspike": Path(base_dir) / "insightspike",
        "results": Path(base_dir) / "results",
        "embeddings": Path(base_dir) / "embeddings",
        "indices": Path(base_dir) / "indices",
        "datasets": Path(base_dir) / "datasets",
        "models": Path(base_dir) / "models"
    }
    
    for name, dir_path in dirs.items():
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {dir_path}")
    
    # Create subdirectories for baseline methods
    baseline_methods = ["bm25", "static_embeddings", "dpr", "tfidf"]
    for method in baseline_methods:
        method_dir = dirs["baselines"] / method
        method_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for each method
        for subdir in ["indices", "embeddings", "results"]:
            (method_dir / subdir).mkdir(exist_ok=True)
    
    return dirs

def save_baseline_data(baseline_name: str, data: Dict, experiment_dirs: Dict):
    """Save baseline method data (embeddings, indices, results)"""
    import pickle
    import json
    from datetime import datetime
    
    baseline_dir = experiment_dirs["baselines"] / baseline_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save embeddings if available
    if "embeddings" in data:
        embedding_file = baseline_dir / "embeddings" / f"embeddings_{timestamp}.pkl"
        with open(embedding_file, "wb") as f:
            pickle.dump(data["embeddings"], f)
        print(f"💾 Saved {baseline_name} embeddings to {embedding_file}")
    
    # Save indices/models
    if "model" in data:
        model_file = baseline_dir / "indices" / f"model_{timestamp}.pkl"
        with open(model_file, "wb") as f:
            pickle.dump(data["model"], f)
        print(f"💾 Saved {baseline_name} model to {model_file}")
    
    # Save configuration and metadata
    config = {
        "baseline_name": baseline_name,
        "timestamp": timestamp,
        "document_count": data.get("document_count", 0),
        "parameters": data.get("parameters", {}),
        "preprocessing": data.get("preprocessing", {})
    }
    
    config_file = baseline_dir / f"config_{timestamp}.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)
    print(f"💾 Saved {baseline_name} config to {config_file}")

def save_experiment_results(results: Dict, experiment_dirs: Dict, experiment_name: str = "rag_comparison"):
    """Save comprehensive experiment results with environment and library version tracking"""
    import json
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("📊 Collecting environment information...")
    env_info = collect_environment_info()
    
    # Create comprehensive results structure
    comprehensive_results = {
        "experiment_metadata": {
            "name": experiment_name,
            "timestamp": timestamp,
            "total_questions": len(results.get("questions", [])),
            "systems_compared": list(results.keys()),
            "datasets_used": ["NaturalQuestions_sample", "HotpotQA_sample"],
            "experiment_id": f"{experiment_name}_{timestamp}",
            "reproducibility_info": {
                "random_seed": "Not explicitly set - recommend setting for reproducibility",
                "deterministic_operations": "Not explicitly configured"
            }
        },
        "environment_info": env_info,
        "system_results": results,
        "statistical_analysis": {},
        "performance_summary": {}
    }
    
    # Calculate statistical summaries
    if results:
        systems = [k for k in results.keys() if k != "questions"]
        
        for system in systems:
            if system in results:
                sys_results = results[system]
                
                # Calculate summary statistics
                comprehensive_results["performance_summary"][system] = {
                    "recall_at_5": {
                        "mean": np.mean(sys_results.get("recall_at_k", {}).get(5, [])),
                        "std": np.std(sys_results.get("recall_at_k", {}).get(5, [])),
                        "median": np.median(sys_results.get("recall_at_k", {}).get(5, []))
                    },
                    "exact_match": {
                        "mean": np.mean(sys_results.get("exact_matches", [])),
                        "std": np.std(sys_results.get("exact_matches", []))
                    },
                    "f1_score": {
                        "mean": np.mean(sys_results.get("f1_scores", [])),
                        "std": np.std(sys_results.get("f1_scores", []))
                    },
                    "average_latency_ms": np.mean(sys_results.get("latencies", [])) * 1000
                }
    
    # Save main results file
    results_file = experiment_dirs["results"] / f"{experiment_name}_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(comprehensive_results, f, indent=2, default=str)
    print(f"💾 Saved comprehensive results to {results_file}")
    
    # Save CSV for easy analysis
    csv_file = experiment_dirs["results"] / f"{experiment_name}_{timestamp}.csv"
    save_results_as_csv(comprehensive_results, csv_file)
    
    # Save individual system results
    for system in systems:
        if system in results:
            system_file = experiment_dirs["results"] / f"{system}_detailed_{timestamp}.json"
            with open(system_file, "w") as f:
                json.dump({
                    "system": system,
                    "timestamp": timestamp,
                    "results": results[system]
                }, f, indent=2, default=str)
    
    return results_file

def save_results_as_csv(results: Dict, csv_file: Path):
    """Convert results to CSV format for easy analysis"""
    if "system_results" not in results:
        return
    
    # Create a flat structure for CSV
    csv_data = []
    systems = [k for k in results["system_results"].keys() if k != "questions"]
    
    for system in systems:
        sys_results = results["system_results"][system]
        summary = results["performance_summary"].get(system, {})
        
        row = {
            "system": system,
            "recall_at_5_mean": summary.get("recall_at_5", {}).get("mean", 0),
            "recall_at_5_std": summary.get("recall_at_5", {}).get("std", 0),
            "exact_match_mean": summary.get("exact_match", {}).get("mean", 0),
            "f1_score_mean": summary.get("f1_score", {}).get("mean", 0),
            "average_latency_ms": summary.get("average_latency_ms", 0),
            "total_questions": len(sys_results.get("exact_matches", []))
        }
        csv_data.append(row)
    
    # Save to CSV
    import pandas as pd
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_file, index=False)
    print(f"📊 Saved CSV summary to {csv_file}")

def load_or_create_baseline_data(baseline_name: str, documents: List[str], experiment_dirs: Dict) -> Any:
    """Load existing baseline data or create new if not exists"""
    import pickle
    import glob
    from pathlib import Path
    
    baseline_dir = experiment_dirs["baselines"] / baseline_name
    
    # Look for existing model files
    model_files = list(baseline_dir.glob("indices/model_*.pkl"))
    
    if model_files:
        # Load the most recent model
        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
        print(f"📂 Loading existing {baseline_name} model from {latest_model}")
        
        try:
            with open(latest_model, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"⚠️  Failed to load {baseline_name} model: {e}")
            print(f"🔄 Creating new {baseline_name} model...")
    
    # Create new baseline model
    print(f"🔨 Creating new {baseline_name} baseline...")
    
    if baseline_name == "bm25":
        model = BM25Retriever(documents)
    elif baseline_name == "static_embeddings":
        model = StaticEmbeddingRetriever(documents)
    elif baseline_name == "dpr":
        model = DPRRetriever(documents)
    else:
        raise ValueError(f"Unknown baseline: {baseline_name}")
    
    # Save the newly created model
    model_data = {
        "model": model,
        "document_count": len(documents),
        "parameters": getattr(model, 'parameters', {}),
        "baseline_name": baseline_name
    }
    
    save_baseline_data(baseline_name, model_data, experiment_dirs)
    return model

def run_rag_comparison_experiment_with_storage(experiment_dirs: Dict):
    """Run RAG comparison experiment with persistent storage management"""
    
    # Prepare datasets
    print("📊 Preparing experimental datasets...")
    questions = SAMPLE_NATURAL_QUESTIONS + SAMPLE_HOTPOT_QA
    documents = [q["context"] for q in questions]
    
    # Save dataset for reproducibility
    dataset_file = experiment_dirs["datasets"] / "experiment_dataset.json"
    with open(dataset_file, "w") as f:
        json.dump({
            "questions": questions,
            "documents": documents,
            "metadata": {
                "total_questions": len(questions),
                "total_documents": len(documents),
                "question_types": list(set(q.get("type", "unknown") for q in questions))
            }
        }, f, indent=2)
    print(f"💾 Dataset saved to {dataset_file}")
    
    # Create retrievers with persistent storage
    print("🔧 Initializing retrieval systems...")
    systems = {
        "BM25": load_or_create_baseline_data("bm25", documents, experiment_dirs),
        "Static Embeddings": load_or_create_baseline_data("static_embeddings", documents, experiment_dirs),
        "InsightSpike RAG": InsightSpikeRAG(documents)
    }
    
    # Add DPR if available
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        systems["DPR"] = load_or_create_baseline_data("dpr", documents, experiment_dirs)
        print("✅ All systems initialized (including DPR)")
    else:
        print("⚠️  DPR not available, using fallback systems only")
    
    # Run evaluation
    print("\n🔍 Running retrieval evaluation...")
    results = {}
    
    for system_name, system in systems.items():
        print(f"\n📋 Evaluating {system_name}...")
        
        # Initialize metrics
        results[system_name] = {
            "recall_at_k": {k: [] for k in [1, 3, 5]},
            "precision_at_k": {k: [] for k in [1, 3, 5]},
            "exact_matches": [],
            "f1_scores": [],
            "latencies": []
        }
        
        # Evaluate each question
        for i, question_data in enumerate(questions):
            question = question_data["question"]
            expected_answer = question_data["answer"]
            context = question_data["context"]
            
            # Measure retrieval latency
            start_time = time.time()
            
            if system_name == "InsightSpike RAG":
                retrieved_docs, scores = system.retrieve(question, k=5)
            else:
                retrieved_results = system.retrieve(question, k=5)
                retrieved_docs = [idx for idx, _ in retrieved_results]
                scores = [score for _, score in retrieved_results]
            
            end_time = time.time()
            latency = end_time - start_time
            results[system_name]["latencies"].append(latency)
            
            # Calculate metrics
            for k in [1, 3, 5]:
                top_k_docs = retrieved_docs[:k]
                
                # Recall@k (simplified: check if correct document is in top-k)
                correct_doc_idx = i  # In this setup, each question's context is its own document
                recall = 1.0 if correct_doc_idx in top_k_docs else 0.0
                results[system_name]["recall_at_k"][k].append(recall)
                
                # Precision@k (simplified)
                precision = recall / k if k > 0 else 0.0
                results[system_name]["precision_at_k"][k].append(precision)
            
            # Exact Match and F1 (simplified)
            if retrieved_docs:
                # Get the top retrieved document
                top_doc_idx = retrieved_docs[0]
                if top_doc_idx < len(documents):
                    retrieved_context = documents[top_doc_idx]
                    
                    # Simple exact match check
                    em = 1.0 if expected_answer.lower() in retrieved_context.lower() else 0.0
                    results[system_name]["exact_matches"].append(em)
                    
                    # Simple F1 calculation (token overlap)
                    f1 = calculate_f1_score(expected_answer, retrieved_context)
                    results[system_name]["f1_scores"].append(f1)
                else:
                    results[system_name]["exact_matches"].append(0.0)
                    results[system_name]["f1_scores"].append(0.0)
            else:
                results[system_name]["exact_matches"].append(0.0)
                results[system_name]["f1_scores"].append(0.0)
            
            if (i + 1) % 2 == 0:
                print(f"  Processed {i + 1}/{len(questions)} questions")
        
        # Save individual system results
        system_results = {
            "system_name": system_name,
            "results": results[system_name],
            "summary": {
                "avg_recall_at_5": np.mean(results[system_name]["recall_at_k"][5]),
                "avg_precision_at_5": np.mean(results[system_name]["precision_at_k"][5]),
                "avg_exact_match": np.mean(results[system_name]["exact_matches"]),
                "avg_f1": np.mean(results[system_name]["f1_scores"]),
                "avg_latency_ms": np.mean(results[system_name]["latencies"]) * 1000
            }
        }
        
        # Save system-specific results
        save_baseline_data(system_name.lower().replace(" ", "_"), 
                          {"results": system_results}, experiment_dirs)
        
        print(f"✅ {system_name} evaluation complete")
        print(f"   Recall@5: {system_results['summary']['avg_recall_at_5']:.3f}")
        print(f"   F1 Score: {system_results['summary']['avg_f1']:.3f}")
        print(f"   Latency: {system_results['summary']['avg_latency_ms']:.1f}ms")
    
    return results, questions, documents

def collect_environment_info():
    """Collect comprehensive environment and library version information"""
    import sys
    import platform
    import subprocess
    import pkg_resources
    import os
    
    env_info = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "architecture": platform.architecture(),
            "platform_details": platform.platform()
        },
        "environment_variables": {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "HOME": os.environ.get("HOME", ""),
            "PWD": os.environ.get("PWD", "")
        },
        "libraries": {},
        "hardware_info": {}
    }
    
    # Collect library versions
    important_libraries = [
        'numpy', 'pandas', 'matplotlib', 'seaborn', 'scikit-learn',
        'sentence-transformers', 'torch', 'transformers', 'faiss-cpu',
        'plotly', 'networkx', 'jupyter', 'ipython'
    ]
    
    for lib in important_libraries:
        try:
            version = pkg_resources.get_distribution(lib).version
            env_info["libraries"][lib] = version
        except pkg_resources.DistributionNotFound:
            env_info["libraries"][lib] = "not_installed"
        except Exception as e:
            env_info["libraries"][lib] = f"error: {str(e)}"
    
    # Additional PyTorch specific info
    try:
        import torch
        env_info["pytorch_info"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
            "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
        }
        
        if torch.cuda.is_available():
            try:
                env_info["pytorch_info"]["device_names"] = [
                    torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
                ]
                env_info["pytorch_info"]["device_capabilities"] = [
                    torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())
                ]
            except Exception:
                pass
    except ImportError:
        env_info["pytorch_info"] = {"error": "PyTorch not available"}
    
    # Memory information
    try:
        import psutil
        env_info["hardware_info"]["memory"] = {
            "total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
            "percent_used": psutil.virtual_memory().percent
        }
        env_info["hardware_info"]["cpu"] = {
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
        }
    except ImportError:
        env_info["hardware_info"]["note"] = "psutil not available for detailed hardware info"
    
    # Git information if available
    try:
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], 
                                         stderr=subprocess.DEVNULL).decode('ascii').strip()
        git_branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], 
                                           stderr=subprocess.DEVNULL).decode('ascii').strip()
        env_info["git_info"] = {
            "commit_hash": git_hash,
            "branch": git_branch
        }
        
        # Check if there are uncommitted changes
        try:
            subprocess.check_output(['git', 'diff', '--quiet'])
            env_info["git_info"]["clean_working_directory"] = True
        except subprocess.CalledProcessError:
            env_info["git_info"]["clean_working_directory"] = False
            env_info["git_info"]["warning"] = "Uncommitted changes detected"
    except (subprocess.CalledProcessError, FileNotFoundError):
        env_info["git_info"] = {"error": "Git not available or not in git repository"}
    
    # Execution environment detection
    env_info["execution_environment"] = "unknown"
    if "google.colab" in sys.modules:
        env_info["execution_environment"] = "google_colab"
    elif "ipykernel" in sys.modules:
        env_info["execution_environment"] = "jupyter"
    elif hasattr(sys, 'ps1'):
        env_info["execution_environment"] = "interactive_python"
    else:
        env_info["execution_environment"] = "script"
    
    return env_info
