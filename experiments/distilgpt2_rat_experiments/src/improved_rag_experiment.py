#!/usr/bin/env python3
"""
Improved RAG experiment with better knowledge base
Tests if better context improves RAG performance on RAT
"""

import json
from pathlib import Path
from datetime import datetime
from transformers import pipeline, set_seed
from tqdm import tqdm

class ImprovedRAGExperiment:
    def __init__(self):
        print("🚀 Initializing Improved RAG Experiment...")
        self.llm = pipeline('text-generation', model='distilgpt2', device=-1)
        set_seed(42)
        
        # Load RAT dataset
        data_path = Path(__file__).parent.parent / "data" / "input" / "rat_100_problems.json"
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        self.problems = self.data['problems'][:20]  # Test on subset
        print(f"✅ Testing on {len(self.problems)} RAT problems")
        
    def generate_word_context(self, word):
        """Generate rich context for a word using LLM"""
        prompts = [
            f"Define {word} and list its meanings:",
            f"List compound words containing {word}:",
            f"What things are associated with {word}?"
        ]
        
        context_parts = []
        for prompt in prompts:
            result = self.llm(prompt, max_new_tokens=30, temperature=0.7)
            text = result[0]['generated_text']
            # Extract the generated part
            if ':' in text:
                generated = text.split(':', 1)[1].strip()
                if generated and len(generated) > 5:
                    context_parts.append(generated)
        
        return ' '.join(context_parts) if context_parts else f"{word} is a common word"
    
    def test_simple_rag(self, problem):
        """Original simple RAG approach"""
        context = f"Words to connect: {', '.join(problem['words'])}. Think of compound words."
        prompt = f"{context}\nAnswer:"
        
        result = self.llm(prompt, max_new_tokens=5)
        text = result[0]['generated_text']
        words = text.split()
        return words[-1].upper() if words else "UNKNOWN"
    
    def test_improved_rag(self, problem):
        """Improved RAG with rich word contexts"""
        # Generate context for each word
        contexts = []
        for word in problem['words']:
            word_context = self.generate_word_context(word)
            contexts.append(f"{word}: {word_context}")
        
        # Build comprehensive prompt
        full_context = "\n".join(contexts)
        prompt = f"""Context about the words:
{full_context}

Task: Find a single word that connects all three words: {', '.join(problem['words'])}
Think about compound words, phrases, or common associations.
Answer:"""
        
        result = self.llm(prompt, max_new_tokens=10, temperature=0.5)
        text = result[0]['generated_text']
        
        # Extract answer
        if "Answer:" in text:
            answer_part = text.split("Answer:")[-1].strip().upper().split()
            return answer_part[0] if answer_part else "UNKNOWN"
        else:
            words = text.split()
            return words[-1].upper() if words else "UNKNOWN"
    
    def test_structured_rag(self, problem):
        """RAG with structured compound word hints"""
        # Create structured hints
        hints = []
        for word in problem['words']:
            hints.append(f"- {word} can form compounds like: {word.lower()}-X or X-{word.lower()}")
        
        prompt = f"""RAT Puzzle: Find the connecting word for {', '.join(problem['words'])}

Hints:
{chr(10).join(hints)}

Examples of compound patterns:
- COTTAGE cheese, SWISS cheese, cheese CAKE
- ICE cream, ICE skate, water ICE

The connecting word is:"""
        
        result = self.llm(prompt, max_new_tokens=5, temperature=0.3)
        answer = result[0]['generated_text'].split()[-1].upper()
        return answer if answer else "UNKNOWN"
    
    def run_experiment(self):
        """Compare different RAG approaches"""
        print("\n🧪 Running Improved RAG Comparison")
        print("=" * 60)
        
        results = {
            'simple': {'correct': 0, 'total': 0},
            'improved': {'correct': 0, 'total': 0},
            'structured': {'correct': 0, 'total': 0}
        }
        
        detailed_results = []
        
        for problem in tqdm(self.problems, desc="Testing"):
            # Test all approaches
            simple_answer = self.test_simple_rag(problem)
            improved_answer = self.test_improved_rag(problem)
            structured_answer = self.test_structured_rag(problem)
            
            correct = problem['answer']
            
            # Check correctness
            simple_correct = simple_answer == correct
            improved_correct = improved_answer == correct
            structured_correct = structured_answer == correct
            
            # Update counters
            results['simple']['total'] += 1
            results['improved']['total'] += 1
            results['structured']['total'] += 1
            
            if simple_correct: results['simple']['correct'] += 1
            if improved_correct: results['improved']['correct'] += 1
            if structured_correct: results['structured']['correct'] += 1
            
            # Store detailed results
            detailed_results.append({
                'problem': problem,
                'answers': {
                    'simple': simple_answer,
                    'improved': improved_answer,
                    'structured': structured_answer
                },
                'correct': {
                    'simple': simple_correct,
                    'improved': improved_correct,
                    'structured': structured_correct
                }
            })
            
            # Show progress for interesting cases
            if improved_correct and not simple_correct:
                print(f"\n✅ Improved RAG solved: {problem['words']} → {correct}")
        
        # Calculate accuracies
        accuracies = {}
        for method in results:
            total = results[method]['total']
            correct = results[method]['correct']
            accuracies[method] = (correct / total * 100) if total > 0 else 0
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 RESULTS")
        print("=" * 60)
        print(f"Simple RAG      : {results['simple']['correct']}/{results['simple']['total']} = {accuracies['simple']:.1f}%")
        print(f"Improved RAG    : {results['improved']['correct']}/{results['improved']['total']} = {accuracies['improved']:.1f}%")
        print(f"Structured RAG  : {results['structured']['correct']}/{results['structured']['total']} = {accuracies['structured']:.1f}%")
        
        # Save results
        output_dir = Path(__file__).parent.parent / "results" / "outputs"
        output_dir.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"improved_rag_results_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump({
                'metadata': {
                    'experiment': 'Improved RAG Comparison',
                    'num_problems': len(self.problems),
                    'timestamp': timestamp
                },
                'accuracies': accuracies,
                'results': results,
                'detailed_results': detailed_results
            }, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
        
        # Analysis
        print("\n💡 ANALYSIS:")
        improvement = accuracies['improved'] - accuracies['simple']
        if improvement > 0:
            print(f"✓ Improved RAG shows {improvement:.1f}% improvement over simple RAG")
            print("✓ Better context helps with creative problem solving")
        else:
            print("✗ Even with better context, RAG struggles with creative tasks")
            print("✗ RAT requires deeper conceptual understanding than retrieval")
        
        return results

if __name__ == "__main__":
    experiment = ImprovedRAGExperiment()
    experiment.run_experiment()