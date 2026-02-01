
"""
Demo: Structure-Guided RAG Reranking (The 'Banana Test')
========================================================

Verifies if Flash-geDIG can distinguish between:
1. Coherent, logical text (High Structure)
2. Incoherent, chaotic text (Low Structure)

Hypothesis: 
Transformer attention should form a 'Small-World Network' (High SP, Low Entropy) 
for logical text, but a 'Random Graph' for chaotic text.
"""

import sys
import os
import re
import torch

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from insightspike.rag.reranker import StructureReranker

def _tokenize(text: str):
    return re.findall(r"[a-z0-9]+", text.lower())

def compute_base_scores(query: str, documents):
    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return [0.0 for _ in documents]
    scores = []
    for doc in documents:
        doc_tokens = set(_tokenize(doc))
        overlap = query_tokens.intersection(doc_tokens)
        scores.append(len(overlap) / len(query_tokens))
    return scores

def run_banana_test():
    print("=== Structure-Guided RAG: The 'Banana Test' ===")
    
    # 1. Setup
    model_name = "bert-base-uncased"
    reranker = StructureReranker(model_name=model_name)
    
    query = "What is the definition of Artificial Intelligence?"
    
    documents = [
        # Doc A: Coherent, high definition quality
        "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by animals including humans.",
        
        # Doc B: Semi-coherent but irrelevant
        "The banana is an edible fruit – botanically a berry – produced by several kinds of large herbaceous flowering plants in the genus Musa.",
        
        # Doc C: Completely incoherent (Word Salad)
        "Blue sky banana hamburger justice river cloud dance computer yesterday tomorrow purple monkey dishwasher."
    ]
    
    mix_weight = 0.2
    gate_min_norm = 0.1
    gate_penalty = 1.0

    print(f"\nQuery: {query}\n")
    print(f"Settings: mix_weight={mix_weight}, gate_min_norm={gate_min_norm}, gate_penalty={gate_penalty}\n")
    
    # 2. Rerank
    base_scores = compute_base_scores(query, documents)
    results = reranker.rerank(
        query,
        documents,
        base_scores=base_scores,
        mix_weight=mix_weight,
        gate_min_norm=gate_min_norm,
        gate_penalty=gate_penalty,
    )
    
    # 3. Display Results
    print(f"{'Rank':<5} | {'Combined':<9} | {'Struct':<8} | {'Base':<6} | {'Gated':<5} | {'SP':<8} | {'Ent':<8} | {'EPC':<8} | {'Clust':<8} | {'Content (First 30 chars)'}")
    print("-" * 120)
    
    for i, res in enumerate(results):
        m = res["metrics"]
        combined = res.get("combined_score", res["rank_score"])
        base = res.get("base_score", 0.0)
        gated = "Y" if res.get("gated") else "N"
        print(f"{i+1:<5} | {combined:.4f}   | {res['rank_score']:.4f}  | {base:.2f}  | {gated:<5} | {m['sp']:.4f}   | {m['entropy']:.4f}   | {m['epc']:.4f}   | {m.get('clustering', 0.0):.4f}   | {res['doc'][:30]}...")

    # 4. Verification Logic
    # We expect Rank 1 to be Doc A (AI definition) or maybe Doc B (Grammatically correct).
    # We expect Last Rank to be Doc C (Word Salad).
    
    top_doc = results[0]["doc"]
    bottom_doc = results[-1]["doc"]
    
    if "Blue sky" in bottom_doc:
        print("\nSUCCESS: Logic prevailed over Chaos (Word Salad is last).")
    else:
        print("\nFAILURE: Chaos reigns supreme.")

if __name__ == "__main__":
    run_banana_test()
