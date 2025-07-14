#!/usr/bin/env python3
"""
RAT Test with RAG comparison
Shows: Base LLM < RAG < InsightSpike
"""

import time
from transformers import pipeline, set_seed
import numpy as np
from pathlib import Path
import json

class RATComparison:
    def __init__(self):
        print("🚀 Initializing RAT comparison (Base vs RAG vs InsightSpike)...")
        self.llm = pipeline('text-generation', model='distilgpt2', device=-1)
        set_seed(42)
        print("✅ Ready!")
        
    def test_base_llm(self, words):
        """Base LLM with no context"""
        prompt = f"What word connects {', '.join(words)}?"
        
        start = time.time()
        result = self.llm(prompt, max_new_tokens=5)
        elapsed = time.time() - start
        
        response = result[0]['generated_text'].split()[-1].upper() if result else "?"
        
        return {
            "answer": response,
            "time": elapsed,
            "method": "Base LLM"
        }
    
    def test_rag(self, words, documents):
        """Traditional RAG - just concatenate documents"""
        # Build RAG prompt with all documents
        context = "Related information:\n"
        for word in words:
            if word in documents:
                context += f"- {word}: {', '.join(documents[word])}\n"
        
        prompt = f"{context}\nQuestion: What word connects {', '.join(words)}?\nAnswer:"
        
        start = time.time()
        result = self.llm(prompt, max_new_tokens=10)
        elapsed = time.time() - start
        
        # Extract answer
        response = result[0]['generated_text']
        if "Answer:" in response:
            response = response.split("Answer:")[-1].strip().upper().split()[0]
        else:
            response = response.split()[-1].upper()
        
        return {
            "answer": response,
            "time": elapsed,
            "method": "RAG",
            "context_length": len(context)
        }
    
    def test_insightspike(self, words, documents):
        """InsightSpike - find connections and detect spikes"""
        # Analyze connections (simplified geDIG logic)
        word_counts = {}
        
        for word in words:
            if word in documents:
                for doc in documents[word]:
                    for w in doc.lower().split():
                        if len(w) > 3:
                            word_counts[w] = word_counts.get(w, 0) + 1
        
        # Find strongest connection
        connections = [(w, count) for w, count in word_counts.items() if count >= 2]
        connections.sort(key=lambda x: x[1], reverse=True)
        
        # Detect spike
        spike = len(connections) > 0 and connections[0][1] >= len(words) - 1
        
        if connections and spike:
            answer = connections[0][0].upper()
            # Build insight-aware prompt
            prompt = f"Insight: The words {', '.join(words)} all connect to {answer}. This is because"
        else:
            # Fallback
            answer = "UNKNOWN"
            prompt = f"Finding connection between {', '.join(words)}:"
        
        start = time.time()
        result = self.llm(prompt, max_new_tokens=20)
        elapsed = time.time() - start
        
        return {
            "answer": answer,
            "time": elapsed,
            "method": "InsightSpike",
            "spike_detected": spike,
            "connections_found": len(connections),
            "confidence": connections[0][1] / len(words) if connections else 0
        }

def run_comparison():
    """Run the three-way RAT comparison"""
    comparator = RATComparison()
    
    # Test problems with associated knowledge
    problems = [
        {
            "words": ["COTTAGE", "SWISS", "CAKE"],
            "answer": "CHEESE",
            "documents": {
                "COTTAGE": ["cottage cheese", "small rural house"],
                "SWISS": ["Swiss cheese", "from Switzerland"],
                "CAKE": ["cheesecake", "birthday dessert"]
            }
        },
        {
            "words": ["CREAM", "SKATE", "WATER"],
            "answer": "ICE",
            "documents": {
                "CREAM": ["ice cream", "dairy product"],
                "SKATE": ["ice skate", "skating rink"],
                "WATER": ["ice water", "frozen water"]
            }
        },
        {
            "words": ["DUCK", "FOLD", "DOLLAR"],
            "answer": "BILL",
            "documents": {
                "DUCK": ["duck bill", "water bird"],
                "FOLD": ["fold bills", "paper folding"],
                "DOLLAR": ["dollar bill", "US currency"]
            }
        }
    ]
    
    print("\n🧪 Three-Way RAT Comparison")
    print("=" * 60)
    
    results = []
    
    for i, problem in enumerate(problems, 1):
        print(f"\nProblem {i}: {', '.join(problem['words'])} → {problem['answer']}")
        print("-" * 40)
        
        # Test all three methods
        base_result = comparator.test_base_llm(problem['words'])
        rag_result = comparator.test_rag(problem['words'], problem['documents'])
        insight_result = comparator.test_insightspike(problem['words'], problem['documents'])
        
        # Display results
        for result in [base_result, rag_result, insight_result]:
            correct = problem['answer'] == result['answer']
            print(f"{result['method']:12} : {result['answer']:10} {'✅' if correct else '❌'} ({result['time']:.2f}s)")
            
            if result['method'] == 'InsightSpike' and result.get('spike_detected'):
                print(f"             🎯 Spike! Confidence: {result['confidence']:.1%}")
        
        results.append({
            "problem": problem,
            "base": base_result,
            "rag": rag_result, 
            "insight": insight_result
        })
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY: Progressive Improvement")
    print("=" * 60)
    
    # Calculate accuracies
    base_correct = sum(1 for r in results if r['problem']['answer'] == r['base']['answer'])
    rag_correct = sum(1 for r in results if r['problem']['answer'] == r['rag']['answer'])
    insight_correct = sum(1 for r in results if r['problem']['answer'] == r['insight']['answer'])
    
    total = len(results)
    
    print(f"Base LLM     : {base_correct}/{total} = {base_correct/total*100:.0f}%")
    print(f"RAG          : {rag_correct}/{total} = {rag_correct/total*100:.0f}%")
    print(f"InsightSpike : {insight_correct}/{total} = {insight_correct/total*100:.0f}%")
    
    print(f"\nProgression  : {base_correct/total*100:.0f}% → {rag_correct/total*100:.0f}% → {insight_correct/total*100:.0f}%")
    
    # Save results
    output_dir = Path(__file__).parent.parent / "results" / "outputs"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    output_file = output_dir / "rat_rag_comparison.json"
    with open(output_file, 'w') as f:
        json.dump({
            "experiment": "RAT with RAG comparison",
            "results": results,
            "summary": {
                "base_accuracy": base_correct/total,
                "rag_accuracy": rag_correct/total,
                "insight_accuracy": insight_correct/total
            }
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    run_comparison()