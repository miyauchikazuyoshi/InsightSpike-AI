"""
Test distance-based search with direct LLM API
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
import anthropic
from sentence_transformers import SentenceTransformer


class DistanceSearchExperiment:
    def __init__(self, api_key: str = None):
        """Initialize experiment with embedder and LLM."""
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        
        if self.api_key:
            self.client = anthropic.Anthropic(api_key=self.api_key)
        else:
            self.client = None
            print("Note: Using mock responses. Set ANTHROPIC_API_KEY for real API calls.")
    
    def distance_based_search(
        self, 
        query_vec: np.ndarray,
        item_vecs: List[Tuple[str, np.ndarray]],
        radius: float = 1.0
    ) -> List[Dict[str, Any]]:
        """
        Perform distance-based sphere search.
        
        Args:
            query_vec: Query embedding
            item_vecs: List of (text, embedding) tuples
            radius: Distance radius for search
            
        Returns:
            List of items within radius with distance metrics
        """
        results = []
        
        # Normalize vectors
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        
        for text, vec in item_vecs:
            vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
            
            # Calculate distance and cosine similarity
            distance = np.linalg.norm(query_norm - vec_norm)
            cosine_sim = np.dot(query_norm, vec_norm)
            
            # Include if within radius
            if distance <= radius:
                results.append({
                    "text": text,
                    "distance": distance,
                    "cosine": cosine_sim
                })
        
        # Sort by distance
        results.sort(key=lambda x: x["distance"])
        return results
    
    def create_prompt_with_distances(
        self,
        question: str,
        search_results: List[Dict[str, Any]],
        include_metrics: bool = True
    ) -> str:
        """Create prompt with distance metrics."""
        
        prompt = f"Question: {question}\n\n"
        prompt += "Context (ordered by relevance):\n"
        
        for i, result in enumerate(search_results, 1):
            if include_metrics:
                # Include both distance and cosine metrics
                prompt += f"{i}. {result['text']} (dist={result['distance']:.3f}, cos={result['cosine']:.3f})\n"
            else:
                # Just text
                prompt += f"{i}. {result['text']}\n"
        
        prompt += "\nBased on this context, please provide a comprehensive answer:"
        return prompt
    
    def generate_response(self, prompt: str) -> str:
        """Generate response using LLM."""
        if self.client is None:
            return "Mock response: Based on the provided context..."
        
        try:
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=300,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return message.content[0].text
        except Exception as e:
            return f"Error: {str(e)}"
    
    def evaluate_response_similarity(
        self,
        response: str,
        expected: str
    ) -> Dict[str, float]:
        """Evaluate how similar the response is to expected answer."""
        # Encode both
        response_vec = self.model.encode([response])[0]
        expected_vec = self.model.encode([expected])[0]
        
        # Normalize
        response_norm = response_vec / (np.linalg.norm(response_vec) + 1e-8)
        expected_norm = expected_vec / (np.linalg.norm(expected_vec) + 1e-8)
        
        # Calculate metrics
        distance = np.linalg.norm(response_norm - expected_norm)
        cosine_sim = np.dot(response_norm, expected_norm)
        
        return {
            "distance": distance,
            "cosine": cosine_sim
        }
    
    def run_comparison_experiment(
        self,
        test_case: Dict[str, Any],
        knowledge_level: str = "standard"
    ) -> Dict[str, Any]:
        """Run experiment comparing distance vs cosine search."""
        
        question = test_case["question"]
        expected = test_case["expected_answer"]
        knowledge_items = test_case["knowledge_items"][knowledge_level]
        
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print(f"Knowledge items: {len(knowledge_items)}")
        
        # Encode all
        query_vec = self.model.encode([question])[0]
        item_vecs = [(item, self.model.encode([item])[0]) for item in knowledge_items]
        
        # Test different radii
        radii = [0.8, 1.0, 1.2]
        results = {}
        
        for radius in radii:
            print(f"\n--- Testing radius: {radius} ---")
            
            # Distance-based search
            search_results = self.distance_based_search(query_vec, item_vecs, radius)
            print(f"Found {len(search_results)} items within radius")
            
            if search_results:
                # Create prompt and generate response
                prompt = self.create_prompt_with_distances(question, search_results)
                response = self.generate_response(prompt)
                
                # Evaluate
                similarity = self.evaluate_response_similarity(response, expected)
                
                print(f"Response preview: {response[:100]}...")
                print(f"Similarity to expected: dist={similarity['distance']:.3f}, cos={similarity['cosine']:.3f}")
                
                results[f"radius_{radius}"] = {
                    "items_found": len(search_results),
                    "response": response,
                    "similarity": similarity,
                    "search_results": search_results
                }
        
        # Also test traditional cosine similarity (top-k)
        print(f"\n--- Traditional cosine similarity (top-5) ---")
        
        # Calculate cosine similarities for all items
        cosine_results = []
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        
        for text, vec in item_vecs:
            vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
            cosine_sim = np.dot(query_norm, vec_norm)
            distance = np.linalg.norm(query_norm - vec_norm)
            
            cosine_results.append({
                "text": text,
                "distance": distance,
                "cosine": cosine_sim
            })
        
        # Sort by cosine similarity (descending)
        cosine_results.sort(key=lambda x: x["cosine"], reverse=True)
        top_k_results = cosine_results[:5]
        
        # Generate response
        prompt = self.create_prompt_with_distances(question, top_k_results, include_metrics=False)
        response = self.generate_response(prompt)
        similarity = self.evaluate_response_similarity(response, expected)
        
        print(f"Response preview: {response[:100]}...")
        print(f"Similarity to expected: dist={similarity['distance']:.3f}, cos={similarity['cosine']:.3f}")
        
        results["cosine_top5"] = {
            "items_found": len(top_k_results),
            "response": response,
            "similarity": similarity,
            "search_results": top_k_results
        }
        
        return results


def main():
    """Run experiments."""
    # Set API key
    os.environ["ANTHROPIC_API_KEY"] = "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"
    
    # Initialize experiment
    experiment = DistanceSearchExperiment()
    
    # Load test cases
    test_cases_path = Path(__file__).parent / "data" / "test_cases.json"
    with open(test_cases_path, 'r') as f:
        data = json.load(f)
    
    # Run experiment on first test case
    test_case = data["test_cases"][0]  # Scientific discovery
    results = experiment.run_comparison_experiment(test_case, "standard")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for method, result in results.items():
        sim = result["similarity"]
        print(f"\n{method}:")
        print(f"  Items used: {result['items_found']}")
        print(f"  Distance to expected: {sim['distance']:.3f}")
        print(f"  Cosine to expected: {sim['cosine']:.3f}")
    
    # Find best method
    best_method = min(results.items(), key=lambda x: x[1]["similarity"]["distance"])
    print(f"\nBest method: {best_method[0]} (distance={best_method[1]['similarity']['distance']:.3f})")
    
    # Save results (convert numpy types to Python types)
    def convert_to_json_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        return obj
    
    output = {
        "test_case": test_case,
        "results": convert_to_json_serializable(results),
        "best_method": best_method[0]
    }
    
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / "distance_search_direct_results.json", 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {results_dir / 'distance_search_direct_results.json'}")


if __name__ == "__main__":
    main()