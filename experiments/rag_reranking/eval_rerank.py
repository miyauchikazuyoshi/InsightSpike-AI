"""
Small Evaluation: Structure-Guided RAG Reranking
================================================

Compares three ranking strategies on a tiny toy set:
1) Base lexical overlap
2) Structure-only (geDIG)
3) Mixed (base + structure)
"""

import argparse
import os
import re
import sys

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


def doc_index(docs, doc):
    for idx, candidate in enumerate(docs):
        if candidate == doc:
            return idx
    return -1


def build_cases():
    return [
        {
            "query": "What is artificial intelligence?",
            "docs": [
                "Artificial intelligence (AI) is intelligence demonstrated by machines.",
                "The banana is an edible fruit produced by several kinds of flowering plants.",
                "Blue sky banana hamburger justice river cloud dance computer.",
            ],
            "answer_idx": 0,
        },
        {
            "query": "What is a banana?",
            "docs": [
                "Artificial intelligence (AI) is intelligence demonstrated by machines.",
                "The banana is an edible fruit produced by several kinds of flowering plants.",
                "Purple monkey dishwasher cloud apple.",
            ],
            "answer_idx": 1,
        },
        {
            "query": "Where is Mount Everest located?",
            "docs": [
                "Mount Everest is located in the Himalayas on the border of Nepal and China.",
                "The Amazon River flows through South America and is very long.",
                "Time river blue banana justice.",
            ],
            "answer_idx": 0,
        },
        {
            "query": "What do mitochondria do?",
            "docs": [
                "Mitochondria are organelles that generate energy in the form of ATP.",
                "Photosynthesis occurs in chloroplasts and produces glucose.",
                "Cloud river banana quiet idea.",
            ],
            "answer_idx": 0,
        },
        {
            "query": "What is photosynthesis?",
            "docs": [
                "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen.",
                "Cellular respiration breaks down glucose to produce ATP.",
                "Banana river cloud engine.",
            ],
            "answer_idx": 0,
        },
        {
            "query": "Who wrote Hamlet?",
            "docs": [
                "Hamlet was written by William Shakespeare.",
                "Charles Dickens wrote Oliver Twist.",
                "Purple monkey newspaper.",
            ],
            "answer_idx": 0,
        },
        {
            "query": "What is the capital of France?",
            "docs": [
                "Paris is the capital of France.",
                "Berlin is the capital of Germany.",
                "Sky banana river.",
            ],
            "answer_idx": 0,
        },
        {
            "query": "What is the boiling point of water?",
            "docs": [
                "Water boils at 100 degrees Celsius at sea level.",
                "Water freezes at 0 degrees Celsius.",
                "Cloud banana hammer.",
            ],
            "answer_idx": 0,
        },
        {
            "query": "Who painted the Mona Lisa?",
            "docs": [
                "The Mona Lisa was painted by Leonardo da Vinci.",
                "Pablo Picasso painted Guernica.",
                "River blue monkey.",
            ],
            "answer_idx": 0,
        },
        {
            "query": "What is the largest planet?",
            "docs": [
                "Jupiter is the largest planet in the Solar System.",
                "Mars is known as the Red Planet.",
                "Banana sky dust.",
            ],
            "answer_idx": 0,
        },
    ]


def main():
    parser = argparse.ArgumentParser(description="Tiny eval for Structure-Guided RAG reranking")
    parser.add_argument("--model", type=str, default="bert-base-uncased")
    parser.add_argument("--mix_weight", type=float, default=0.7)
    parser.add_argument("--gate_percentile", type=float, default=None)
    parser.add_argument("--gate_min_norm", type=float, default=None)
    parser.add_argument("--gate_penalty", type=float, default=1.0)
    parser.add_argument("--max_cases", type=int, default=0, help="Limit number of cases (0 = all)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cases = build_cases()
    if args.max_cases > 0:
        cases = cases[: args.max_cases]

    reranker = StructureReranker(model_name=args.model, mix_weight=args.mix_weight)

    base_correct = 0
    struct_correct = 0
    mixed_correct = 0

    for idx, case in enumerate(cases, start=1):
        query = case["query"]
        docs = case["docs"]
        answer_idx = case["answer_idx"]

        base_scores = compute_base_scores(query, docs)
        results = reranker.rerank(
            query,
            docs,
            base_scores=base_scores,
            mix_weight=args.mix_weight,
            gate_percentile=args.gate_percentile,
            gate_min_norm=args.gate_min_norm,
            gate_penalty=args.gate_penalty,
        )
        struct_results = sorted(results, key=lambda x: x["rank_score"], reverse=True)

        base_top_idx = max(range(len(docs)), key=lambda i: base_scores[i])
        struct_top_idx = doc_index(docs, struct_results[0]["doc"])
        mixed_top_idx = doc_index(docs, results[0]["doc"])

        if base_top_idx == answer_idx:
            base_correct += 1
        if struct_top_idx == answer_idx:
            struct_correct += 1
        if mixed_top_idx == answer_idx:
            mixed_correct += 1

        if args.verbose:
            print(f"\nCase {idx}")
            print(f"Query: {query}")
            print(f"Answer idx: {answer_idx}")
            print(f"Base top idx: {base_top_idx}")
            print(f"Struct top idx: {struct_top_idx}")
            print(f"Mixed top idx: {mixed_top_idx}")
            print(f"Top (Mixed): {results[0]['doc'][:80]}...")

    total = len(cases)
    if total == 0:
        print("No cases to evaluate.")
        return

    print("\nResults")
    print(f"- total_cases: {total}")
    print(f"- base_top1_acc: {base_correct / total:.3f}")
    print(f"- struct_top1_acc: {struct_correct / total:.3f}")
    print(f"- mixed_top1_acc: {mixed_correct / total:.3f}")


if __name__ == "__main__":
    main()
