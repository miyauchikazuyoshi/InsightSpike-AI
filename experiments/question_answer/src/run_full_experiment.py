#!/usr/bin/env python3
"""
Run Full Minimal Solution Experiment
====================================

Uses sample_knowledge.json for a more realistic test.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from simple_minimal_solution import SimpleMinimalSolutionExperiment


def main():
    """Run full experiment with sample data"""
    print("="*60)
    print("Running Full Minimal Solution Experiment")
    print("="*60)
    
    # Setup paths
    knowledge_path = "data/input/knowledge_base/sample_knowledge_converted.json"
    questions_path = "data/input/questions/sample_questions.json"
    output_dir = "results/metrics"
    
    # Check if files exist
    if not Path(knowledge_path).exists():
        print(f"Error: Knowledge file not found: {knowledge_path}")
        return 1
        
    if not Path(questions_path).exists():
        print(f"Error: Questions file not found: {questions_path}")
        return 1
    
    # Create and run experiment
    experiment = SimpleMinimalSolutionExperiment(knowledge_path, questions_path)
    experiment.run_experiment()
    experiment.save_results(output_dir)
    
    print("\n" + "="*60)
    print("Experiment completed successfully!")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())