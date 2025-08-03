#!/usr/bin/env python3
"""
Test minimal solution experiment with small dataset
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run minimal test"""
    print("Running minimal solution experiment test...")
    print("="*50)
    
    # Change to experiment directory
    experiment_dir = Path(__file__).parent.parent
    
    # Run the experiment with minimal test data
    cmd = [
        sys.executable,
        "src/run_minimal_solution_experiment.py",
        "--knowledge", "data/input/knowledge_base/minimal_test_knowledge.json",
        "--questions", "data/input/questions/minimal_test_questions.json",
        "--output", "results/test"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    print("="*50)
    
    try:
        result = subprocess.run(cmd, cwd=experiment_dir, check=True)
        print("\nTest completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\nTest failed with error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())