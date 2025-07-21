#!/usr/bin/env python3
"""Run only the InsightSpike portion of the experiment"""

import sys
import json
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
    
    # Run InsightSpike for each question
    results = []
    for i, question in enumerate(experiment.questions, 1):
        print(f"\nQuestion {i}/{len(experiment.questions)}: {question}")
        try:
            result = experiment.run_insightspike(question)
            results.append(result)
            print(f"  Success! Spike: {result['spike_detected']}, Phases: {result['phases_integrated']}")
        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                'method': 'insightspike',
                'question': question,
                'error': str(e)
            })
    
    # Save results
    results_file = experiment.results_dir / "outputs" / "insightspike_only_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")