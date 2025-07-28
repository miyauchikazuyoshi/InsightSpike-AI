#!/usr/bin/env python3
"""
Vector Similarity Test for Association Prompt
============================================

Test the effectiveness of association prompts by comparing:
1. Input documents (A, B, C) vectors
2. Expected answer vector
3. AI-generated answer vector (first sentence only)
"""

import os
import numpy as np
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic
from sklearn.metrics.pairwise import cosine_similarity
import re


def extract_first_sentence(text):
    """Extract the first sentence from a text"""
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Find the first sentence (ends with . ! or ?)
    sentences = re.split(r'[.!?]+', text)
    if sentences:
        return sentences[0].strip() + '.'
    return text


def test_association_vector_similarity():
    """Test vector similarity between expected and generated answers"""
    
    # Initialize
    api_key = "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"
    client = Anthropic(api_key=api_key)
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Test cases
    test_cases = [
        {
            "question": "What is the relationship between apples and discovery?",
            "documents": {
                "A": "Newton discovered gravity when an apple fell from a tree",
                "B": "The apple has been a symbol of knowledge since ancient times",
                "C": "Many scientific breakthroughs come from observing everyday objects"
            },
            "relevances": {"A": 0.92, "B": 0.85, "C": 0.73},
            "expected_answer": "Apples represent moments of scientific discovery and enlightenment."
        },
        {
            "question": "How do fruits relate to technology companies?",
            "documents": {
                "A": "Apple Inc. is one of the world's largest technology companies",
                "B": "BlackBerry was a popular smartphone brand",
                "C": "Many tech companies use natural objects as brand names"
            },
            "relevances": {"A": 0.89, "B": 0.87, "C": 0.81},
            "expected_answer": "Technology companies often adopt fruit names to appear approachable and memorable."
        },
        {
            "question": "What connects mathematics to nature?",
            "documents": {
                "A": "The Fibonacci sequence appears in flower petals and seashells",
                "B": "Fractals can be found in coastlines and mountain ranges",
                "C": "Mathematical patterns govern the growth of plants"
            },
            "relevances": {"A": 0.91, "B": 0.88, "C": 0.86},
            "expected_answer": "Mathematical patterns fundamentally structure natural forms and processes."
        }
    ]
    
    print("=== Association Prompt Vector Similarity Test ===\n")
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test Case {i}: {test_case['question']}")
        print("-" * 60)
        
        # Build association prompt
        prompt = f"{test_case['question']}\n\nAccording to my research, the answer to this question has:"
        for label, doc in test_case['documents'].items():
            relevance = test_case['relevances'][label]
            prompt += f"\n- Relevance {relevance:.2f} with document {label}: \"{doc}\""
        prompt += "\n\nWhat statement can be inferred from these associations?\nPlease first express the answer in one concise statement, then explain your reasoning."
        
        # Get AI response
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=500,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}]
        )
        
        ai_response = response.content[0].text
        ai_first_sentence = extract_first_sentence(ai_response)
        
        print(f"Expected: {test_case['expected_answer']}")
        print(f"AI Generated: {ai_first_sentence}")
        
        # Compute vectors
        doc_vectors = {}
        for label, doc in test_case['documents'].items():
            doc_vectors[label] = model.encode(doc)
        
        expected_vector = model.encode(test_case['expected_answer'])
        ai_vector = model.encode(ai_first_sentence)
        
        # Calculate similarities
        # 1. Similarity between AI response and expected answer
        ai_expected_sim = cosine_similarity([ai_vector], [expected_vector])[0][0]
        
        # 2. Similarity between AI response and each document
        ai_doc_sims = {}
        for label, vec in doc_vectors.items():
            ai_doc_sims[label] = cosine_similarity([ai_vector], [vec])[0][0]
        
        # 3. Average similarity between AI response and all documents
        avg_doc_sim = np.mean(list(ai_doc_sims.values()))
        
        # 4. Weighted similarity (using relevance scores)
        weighted_sim = sum(ai_doc_sims[label] * test_case['relevances'][label] 
                          for label in doc_vectors.keys()) / sum(test_case['relevances'].values())
        
        print(f"\nSimilarity Metrics:")
        print(f"  AI ↔ Expected: {ai_expected_sim:.3f}")
        print(f"  AI ↔ Doc A: {ai_doc_sims['A']:.3f}")
        print(f"  AI ↔ Doc B: {ai_doc_sims['B']:.3f}")
        print(f"  AI ↔ Doc C: {ai_doc_sims['C']:.3f}")
        print(f"  Average Doc Similarity: {avg_doc_sim:.3f}")
        print(f"  Weighted Doc Similarity: {weighted_sim:.3f}")
        
        # Store results
        results.append({
            'case': i,
            'ai_expected_sim': ai_expected_sim,
            'avg_doc_sim': avg_doc_sim,
            'weighted_sim': weighted_sim,
            'ai_response': ai_first_sentence
        })
        
        print("\n")
    
    # Summary statistics
    print("=== Summary ===")
    print(f"Average AI ↔ Expected similarity: {np.mean([r['ai_expected_sim'] for r in results]):.3f}")
    print(f"Average document similarity: {np.mean([r['avg_doc_sim'] for r in results]):.3f}")
    print(f"Average weighted similarity: {np.mean([r['weighted_sim'] for r in results]):.3f}")
    
    # Test different prompt styles
    print("\n\n=== Comparing Prompt Styles ===")
    
    # Standard prompt (no association game)
    standard_results = []
    for test_case in test_cases[:1]:  # Just test first case
        # Build standard prompt
        prompt = f"Context:\n"
        for doc in test_case['documents'].values():
            prompt += f"{doc}\n"
        prompt += f"\nQuestion: {test_case['question']}\n\nAnswer:"
        
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=500,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}]
        )
        
        ai_response = response.content[0].text
        ai_first_sentence = extract_first_sentence(ai_response)
        ai_vector = model.encode(ai_first_sentence)
        expected_vector = model.encode(test_case['expected_answer'])
        
        similarity = cosine_similarity([ai_vector], [expected_vector])[0][0]
        
        print(f"Standard Prompt:")
        print(f"  Response: {ai_first_sentence}")
        print(f"  Similarity to expected: {similarity:.3f}")
        
        standard_results.append(similarity)
    
    # Compare with association results
    association_sim = results[0]['ai_expected_sim']
    print(f"\nAssociation Prompt similarity: {association_sim:.3f}")
    print(f"Improvement: {(association_sim - standard_results[0]) / standard_results[0] * 100:.1f}%")


if __name__ == "__main__":
    test_association_vector_similarity()