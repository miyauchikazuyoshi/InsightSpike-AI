#!/usr/bin/env python3
"""
Quick test with limited questions to verify TinyLlama integration.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from run_experiment import ComprehensiveGeDIGExperiment
from question_generator import ExpandedQuestionGenerator


def main():
    """Run quick test with 5 questions."""
    print("=== Quick TinyLlama Test ===")
    
    # Create experiment
    experiment = ComprehensiveGeDIGExperiment(seed=42)
    
    # Generate only 5 test questions
    print("\nGenerating 5 test questions...")
    generator = ExpandedQuestionGenerator(seed=42)
    questions = generator.generate_questions(n_easy=2, n_medium=2, n_hard=1)
    experiment.questions = questions[:5]  # Ensure only 5
    
    print(f"Generated {len(experiment.questions)} questions")
    
    # Initialize agent
    print("\nInitializing agent with TinyLlama...")
    agent = experiment.initialize_agent()
    
    # Run questions
    print("\nProcessing questions...")
    for i, question in enumerate(experiment.questions):
        print(f"\n[{i+1}/5] {question.text}")
        try:
            result = experiment.run_single_question(agent, question.__dict__)
            print(f"  ΔGED: {result['metrics']['delta_ged']:.3f}")
            print(f"  ΔIG: {result['metrics']['delta_ig']:.3f}")
            print(f"  Spike detected: {result['has_spike_detected']}")
            print(f"  Response preview: {result['response'][:100]}...")
            experiment.results['raw_results'].append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
    
    # Summary
    experiment._calculate_summary()
    summary = experiment.results.get('summary', {})
    if summary and 'overall_accuracy' in summary:
        print(f"\nOverall accuracy: {summary['overall_accuracy']:.2%}")
    
    print("\nQuick test completed!")


if __name__ == "__main__":
    main()