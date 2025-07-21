#!/usr/bin/env python3
"""Test a single InsightSpike run with detailed logging"""

import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

# Set detailed logging
logging.basicConfig(level=logging.DEBUG)

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
        print(f"\nSuccess! Result:")
        print(f"Response: {result['response']}")
        print(f"Spike detected: {result['spike_detected']}")
        print(f"Phases integrated: {result['phases_integrated']}")
        print(f"Quality score: {result.get('reasoning_quality', 0.0)}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()