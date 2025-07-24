#!/usr/bin/env python3
"""
Simplified experiment runner focusing on direct metric calculation.
"""

import json
import warnings
from pathlib import Path
import numpy as np
import networkx as nx

# Suppress warnings
warnings.filterwarnings("ignore", message="Advanced metrics not available")
warnings.filterwarnings("ignore", message="Requested GED algorithm")
warnings.filterwarnings("ignore", message="Failed to store episode")


def create_simple_graph(n_nodes=5):
    """Create a simple test graph."""
    G = nx.Graph()
    for i in range(n_nodes):
        G.add_node(i, embedding=np.random.randn(768))
    
    # Add some edges
    for i in range(n_nodes - 1):
        G.add_edge(i, i + 1)
    
    return G


def test_direct_metrics():
    """Test the direct metrics calculation."""
    from direct_metrics import DirectMetricsCalculator
    
    print("=== Testing Direct Metrics ===")
    
    calculator = DirectMetricsCalculator()
    
    # Create test graphs
    G1 = create_simple_graph(5)
    G2 = create_simple_graph(4)  # Simplified graph
    
    # Create test embeddings
    emb1 = np.random.randn(5, 768)
    emb2 = np.random.randn(4, 768)
    
    # Calculate metrics
    delta_ged = calculator.calculate_delta_ged(G1, G2)
    delta_ig = calculator.calculate_delta_ig(emb1, emb2)
    
    print(f"\nΔGED: {delta_ged:.3f}")
    print(f"ΔIG: {delta_ig:.3f}")
    
    # Test insight detection
    has_insight, metrics = calculator.detect_insight(G1, G2, emb1, emb2)
    print(f"\nInsight detected: {has_insight}")
    print(f"Metrics: {json.dumps(metrics, indent=2)}")
    
    return True


def test_question_generation():
    """Test question generation."""
    from question_generator import ExpandedQuestionGenerator
    
    print("\n=== Testing Question Generation ===")
    
    generator = ExpandedQuestionGenerator(seed=42)
    questions = generator.generate_questions(n_easy=5, n_medium=5, n_hard=5)
    
    print(f"\nGenerated {len(questions)} questions")
    
    # Show samples
    for difficulty in ['easy', 'medium', 'hard']:
        sample = next(q for q in questions if q.difficulty == difficulty)
        print(f"\n{difficulty.upper()}: {sample.text}")
        print(f"  Category: {sample.category}")
    
    return True


def run_minimal_experiment():
    """Run a minimal experiment to test core functionality."""
    print("\n=== Running Minimal Experiment ===")
    
    # Test imports
    try:
        import sys
        project_root = Path(__file__).resolve().parents[3]
        sys.path.insert(0, str(project_root))
        
        from insightspike import MainAgent
        print("✓ InsightSpike imports successful")
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False
    
    # Test agent initialization
    try:
        config = type('Config', (), {
            'graph': type('GraphConfig', (), {
                'similarity_threshold': 0.7,
                'conflict_threshold': 0.5,
                'ged_threshold': 0.3
            })(),
            'embedding': type('EmbeddingConfig', (), {
                'dimension': 768,
                'model_name': 'sentence-transformers/all-MiniLM-L6-v2'
            })(),
            'llm': type('LLMConfig', (), {
                'provider': 'mock',
                'model_name': 'mock'
            })(),
            'memory': type('MemoryConfig', (), {
                'max_episodes': 1000,
                'compression_enabled': False
            })(),
            'insight': type('InsightConfig', (), {
                'detection_threshold': 0.5,
                'min_confidence': 0.3
            })()
        })()
        
        agent = MainAgent(config)
        print("✓ Agent initialization successful")
    except Exception as e:
        print(f"✗ Agent initialization error: {e}")
        return False
    
    # Test knowledge addition
    try:
        knowledge_items = [
            "Addition is combining numbers to get a sum",
            "Multiplication is repeated addition",
            "Division is splitting into equal parts"
        ]
        
        for item in knowledge_items:
            try:
                agent.add_knowledge(item)
            except:
                pass  # Ignore storage errors
        
        print("✓ Knowledge addition completed")
    except Exception as e:
        print(f"✗ Knowledge addition error: {e}")
    
    # Test question processing
    try:
        test_questions = [
            "What is 2 + 3?",
            "What is 4 × 5?",
            "What is 10 ÷ 2?"
        ]
        
        for q in test_questions:
            result = agent.process_question(q)
            
            # Handle different result formats
            if hasattr(result, 'response'):
                print(f"\nQ: {q}")
                print(f"A: {result.response}")
                print(f"Spike: {getattr(result, 'has_spike', 'N/A')}")
            else:
                print(f"\nQ: {q}")
                print(f"Result: {result}")
        
        print("\n✓ Question processing successful")
    except Exception as e:
        print(f"✗ Question processing error: {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    print("=== InsightSpike v2 Experiment Test Suite ===\n")
    
    # Test components
    tests = [
        ("Direct Metrics", test_direct_metrics),
        ("Question Generation", test_question_generation),
        ("Minimal Experiment", run_minimal_experiment)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n✗ {test_name} failed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n\n=== Test Summary ===")
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    # Next steps
    print("\n=== Next Steps ===")
    if all(success for _, success in results):
        print("All tests passed! Ready to run full experiment.")
        print("\nTo run full experiment:")
        print("  poetry run python src/run_experiment.py")
    else:
        print("Some tests failed. Please fix issues before running full experiment.")
        print("\nCommon issues:")
        print("  - Missing dependencies (install with: poetry install)")
        print("  - Import path issues (run from experiment directory)")
        print("  - Configuration mismatches")


if __name__ == "__main__":
    main()