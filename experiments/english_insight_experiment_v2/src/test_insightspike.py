#!/usr/bin/env python3
"""Test InsightSpike integration independently"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from run_experiment import EnglishInsightExperiment

if __name__ == "__main__":
    # Get experiment directory
    exp_dir = Path(__file__).parent.parent
    
    # Create experiment
    experiment = EnglishInsightExperiment(exp_dir)
    
    # Prepare datastore
    print("Preparing datastore...")
    experiment.prepare_datastore()
    
    # Test InsightSpike with one question
    question = "What is entropy?"
    print(f"\nTesting InsightSpike with: {question}")
    
    try:
        result = experiment.run_insightspike(question)
        print(f"Success! Result: {result}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()