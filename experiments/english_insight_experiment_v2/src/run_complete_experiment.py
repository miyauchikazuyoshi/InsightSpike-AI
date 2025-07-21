#!/usr/bin/env python3
"""Run the complete experiment with all three approaches"""

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
    
    # Run the complete experiment
    experiment.run_experiment()
    
    print("\nExperiment completed!")
    print(f"Results saved to: {experiment.results_dir}")