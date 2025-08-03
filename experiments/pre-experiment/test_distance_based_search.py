"""
Test distance-based search with LLM prompt builder
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import sys
import os

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from insightspike import MainAgent
from insightspike.utils.similarity_converter import SimilarityConverter
from insightspike.utils.response_evaluator import ResponseEvaluator
# Import embedder later to avoid circular import
from insightspike.config.presets import ConfigPresets


def load_test_cases(file_path: str) -> Dict[str, Any]:
    """Load test cases from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def run_experiment(test_case: Dict[str, Any], knowledge_level: str = "standard") -> Dict[str, Any]:
    """
    Run a single experiment with a test case.
    
    Args:
        test_case: Test case dictionary
        knowledge_level: "minimal", "standard", or "rich"
        
    Returns:
        Experiment results
    """
    print(f"\n{'='*80}")
    print(f"Running test case: {test_case['id']} - {test_case['category']}")
    print(f"Knowledge level: {knowledge_level}")
    print(f"Question: {test_case['question']}")
    print(f"{'='*80}")
    
    # Initialize agent with distance-based search
    # Use dictionary config directly
    config = {
        "llm": {
            "provider": "anthropic",
            "model": "claude-3-opus-20240229"
        },
        "query": {
            "search_method": "sphere",
            "intuitive_radius": 0.5,
            "dimension_aware": True
        },
        "memory": {
            "max_retrieved_docs": 10
        },
        "graph": {
            "enable_graph_search": False,
            "similarity_threshold": 0.3
        },
        "processing": {
            "enable_insight_search": True,
            "max_insights_per_query": 5
        }
    }
    
    # Create agent
    agent = MainAgent(config)
    
    # Add knowledge items
    knowledge_items = test_case["knowledge_items"][knowledge_level]
    print(f"\nAdding {len(knowledge_items)} knowledge items...")
    
    for item in knowledge_items:
        agent.add_knowledge(item)
    
    # Process question
    print(f"\nProcessing question...")
    result = agent.process_question(test_case["question"])
    
    # Extract response
    if hasattr(result, 'response'):
        response = result.response
    else:
        response = result.get('response', '')
    
    print(f"\nGenerated response:")
    print(f"{response}")
    
    # Evaluate response
    evaluator = ResponseEvaluator()
    
    # Use sentence-transformers directly to avoid circular import
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Get embeddings
    response_vec = encoder.encode([response])[0]
    expected_vec = encoder.encode([test_case["expected_answer"]])[0]
    
    # Calculate metrics
    distance, cosine_sim = SimilarityConverter.get_both_metrics(
        response_vec, expected_vec
    )
    
    print(f"\nEvaluation against expected answer:")
    print(f"  {SimilarityConverter.format_metrics(distance, cosine_sim)}")
    print(f"  LLM format: {SimilarityConverter.format_for_llm(distance, cosine_sim)}")
    
    # Get detailed evaluation
    evaluation = evaluator.evaluate_response(
        query=test_case["question"],
        response=response,
        context_docs=None  # Could add retrieved docs here
    )
    
    # Check if response mentions key concepts
    key_concepts = extract_key_concepts(test_case["expected_answer"])
    mentioned_concepts = [c for c in key_concepts if c.lower() in response.lower()]
    concept_coverage = len(mentioned_concepts) / len(key_concepts) if key_concepts else 0
    
    print(f"\nKey concept coverage: {concept_coverage:.2%} ({len(mentioned_concepts)}/{len(key_concepts)})")
    print(f"Mentioned: {', '.join(mentioned_concepts)}")
    
    # Analyze graph metrics if available
    if hasattr(result, 'graph_analysis'):
        graph_analysis = result.graph_analysis
        print(f"\nGraph Analysis:")
        print(f"  Spike detected: {graph_analysis.get('spike_detected', False)}")
        print(f"  GED value: {graph_analysis.get('ged_value', 0):.3f}")
        print(f"  IG value: {graph_analysis.get('ig_value', 0):.3f}")
    
    return {
        "test_case_id": test_case["id"],
        "category": test_case["category"],
        "knowledge_level": knowledge_level,
        "question": test_case["question"],
        "response": response,
        "expected_answer": test_case["expected_answer"],
        "distance_to_expected": float(distance),
        "cosine_to_expected": float(cosine_sim),
        "concept_coverage": float(concept_coverage),
        "evaluation": {k: float(v) if isinstance(v, np.floating) else v for k, v in evaluation.items()},
        "graph_analysis": result.graph_analysis if hasattr(result, 'graph_analysis') else {}
    }


def extract_key_concepts(text: str) -> List[str]:
    """Extract key concepts from text (simple implementation)."""
    # Key words that are likely important
    key_words = [
        "scientific", "breakthrough", "observation", "pattern", "discovery",
        "creativity", "problem", "solving", "innovation", "thinking",
        "learning", "connection", "concept", "understanding", "knowledge",
        "insight", "phenomenon", "principle", "solution", "network"
    ]
    
    found_concepts = []
    for word in key_words:
        if word in text.lower():
            found_concepts.append(word)
    
    return found_concepts


def compare_search_methods():
    """Compare distance-based vs cosine-based search."""
    print("\n" + "="*80)
    print("COMPARING SEARCH METHODS")
    print("="*80)
    
    # Test with both search methods
    test_query = "How do everyday observations lead to scientific breakthroughs?"
    knowledge_items = [
        "Scientific breakthroughs often emerge from careful observation of everyday phenomena",
        "Major discoveries come from noticing unusual patterns in common occurrences",
        "Scientists transform familiar observations into profound insights",
        "The history of science is filled with accidental discoveries from daily life",
        "Curiosity about ordinary things drives scientific innovation"
    ]
    
    # Test 1: Distance-based search
    print("\n1. Distance-based search:")
    config1 = {
        "llm": {
            "provider": "anthropic",
            "model": "claude-3-opus-20240229"
        },
        "query": {
            "search_method": "sphere",
            "intuitive_radius": 0.5,
            "dimension_aware": True
        },
        "memory": {
            "max_retrieved_docs": 10
        }
    }
    agent1 = MainAgent(config1)
    
    for item in knowledge_items:
        agent1.add_knowledge(item)
    
    result1 = agent1.process_question(test_query)
    
    # Test 2: Cosine-based search
    print("\n2. Cosine-based search:")
    config2 = {
        "llm": {
            "provider": "anthropic",
            "model": "claude-3-opus-20240229"
        },
        "query": {
            "search_method": "knn",  # Traditional k-nearest
            "top_k": 5
        },
        "memory": {
            "max_retrieved_docs": 10
        }
    }
    agent2 = MainAgent(config2)
    
    for item in knowledge_items:
        agent2.add_knowledge(item)
    
    result2 = agent2.process_question(test_query)
    
    # Compare results
    print("\n" + "-"*60)
    print("COMPARISON RESULTS:")
    print("-"*60)
    
    response1 = result1.response if hasattr(result1, 'response') else result1.get('response', '')
    response2 = result2.response if hasattr(result2, 'response') else result2.get('response', '')
    
    print(f"\nDistance-based response length: {len(response1)} chars")
    print(f"Cosine-based response length: {len(response2)} chars")
    
    # Calculate similarity between responses
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    vec1 = encoder.encode([response1])[0]
    vec2 = encoder.encode([response2])[0]
    
    distance, cosine_sim = SimilarityConverter.get_both_metrics(vec1, vec2)
    print(f"\nSimilarity between responses:")
    print(f"  {SimilarityConverter.format_metrics(distance, cosine_sim)}")
    
    return {
        "distance_response": response1,
        "cosine_response": response2,
        "response_similarity": {"distance": distance, "cosine": cosine_sim}
    }


def main():
    """Run all experiments."""
    # Load test cases
    test_cases_path = Path(__file__).parent / "data" / "test_cases.json"
    data = load_test_cases(test_cases_path)
    test_cases = data["test_cases"]
    
    # Results storage
    all_results = []
    
    # Run experiments for each test case and knowledge level
    for test_case in test_cases[:1]:  # Start with first test case
        for knowledge_level in ["minimal", "standard"]:
            try:
                result = run_experiment(test_case, knowledge_level)
                all_results.append(result)
            except Exception as e:
                print(f"\nError in experiment: {e}")
                continue
    
    # Summary statistics
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    if all_results:
        avg_distance = np.mean([r["distance_to_expected"] for r in all_results])
        avg_cosine = np.mean([r["cosine_to_expected"] for r in all_results])
        avg_coverage = np.mean([r["concept_coverage"] for r in all_results])
    else:
        avg_distance = 0
        avg_cosine = 0
        avg_coverage = 0
    
    print(f"\nAverage distance to expected: {avg_distance:.3f}")
    print(f"Average cosine similarity: {avg_cosine:.3f}")
    print(f"Average concept coverage: {avg_coverage:.2%}")
    
    # Compare search methods
    comparison_results = compare_search_methods()
    
    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    output_path = results_dir / "distance_based_search_results.json"
    with open(output_path, 'w') as f:
        json.dump({
            "experiments": all_results,
            "summary": {
                "avg_distance": avg_distance,
                "avg_cosine": avg_cosine,
                "avg_concept_coverage": avg_coverage
            },
            "search_method_comparison": comparison_results
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()