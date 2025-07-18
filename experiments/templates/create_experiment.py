#!/usr/bin/env python3
"""
Create New Experiment from Template
===================================

This script helps create new experiments following InsightSpike standards.

Usage:
    python create_experiment.py --name my_experiment --type standard
    python create_experiment.py --name rat_test --type rat
"""

import argparse
import shutil
from pathlib import Path
from datetime import datetime
import json
import sys


def create_experiment(name: str, experiment_type: str = "standard"):
    """Create a new experiment directory with proper structure"""
    
    # Validate experiment type
    valid_types = ["standard", "rat", "qa", "performance"]
    if experiment_type not in valid_types:
        print(f"Error: Invalid experiment type '{experiment_type}'")
        print(f"Valid types: {', '.join(valid_types)}")
        return False
    
    # Create experiment path
    experiments_dir = Path(__file__).parent.parent
    experiment_path = experiments_dir / name
    
    if experiment_path.exists():
        print(f"Error: Experiment '{name}' already exists")
        return False
    
    # Create directory structure
    print(f"Creating experiment: {name}")
    
    # Standard directories
    dirs = [
        "src",
        "data/input",
        "data/processed",
        "data/knowledge_base",
        "results/metrics",
        "results/outputs",
        "results/visualizations",
        "results/logs",
        "results/configs",
        "data_snapshots"
    ]
    
    for dir_path in dirs:
        (experiment_path / dir_path).mkdir(parents=True, exist_ok=True)
        print(f"  Created: {dir_path}/")
    
    # Create .gitignore
    gitignore_content = """# Large data files
data/input/*.pkl
data/input/*.h5
data/processed/*.pkl
data/knowledge_base/*.pt

# Temporary files
*.tmp
__pycache__/
.DS_Store

# Keep snapshots metadata only
data_snapshots/*/
!data_snapshots/*.json

# Results (optional - remove if you want to track)
# results/outputs/*
# results/visualizations/*
"""
    
    with open(experiment_path / ".gitignore", "w") as f:
        f.write(gitignore_content)
    
    # Create README.md
    readme_content = f"""# Experiment: {name}

**Type**: {experiment_type.upper()} Experiment  
**Created**: {datetime.now().strftime("%Y-%m-%d")}  
**Status**: 🚧 In Development

## Purpose
[Describe the main objective of this experiment]

## Hypothesis
[State your hypothesis clearly]

## Methods

### Data
- **Source**: [Describe data source]
- **Size**: [Number of samples]
- **Preprocessing**: [Any preprocessing steps]

### Baseline
[Describe baseline method(s)]

### Proposed Method
[Describe how InsightSpike will be used]

### Evaluation Metrics
- [Metric 1]
- [Metric 2]

## How to Run

```bash
cd experiments/{name}
python src/run_experiment.py
```

## Results
[Results will be added here after experiments]

## Key Findings
[Summarize findings]

## Next Steps
[Future work or improvements]

---
*Following guidelines from /CLAUDE.md and experiments/EXPERIMENT_GUIDELINES.md*
"""
    
    with open(experiment_path / "README.md", "w") as f:
        f.write(readme_content)
    
    # Create main experiment file based on type
    if experiment_type == "standard":
        create_standard_experiment(experiment_path, name)
    elif experiment_type == "rat":
        create_rat_experiment(experiment_path, name)
    elif experiment_type == "qa":
        create_qa_experiment(experiment_path, name)
    elif experiment_type == "performance":
        create_performance_experiment(experiment_path, name)
    
    # Create config file
    config_content = {
        "experiment": {
            "name": name,
            "type": experiment_type,
            "created": datetime.now().isoformat(),
            "insightspike_preset": "experiment",
            "random_seed": 42,
            "parameters": {
                # Add experiment-specific parameters here
            }
        }
    }
    
    with open(experiment_path / "experiment_config.yaml", "w") as f:
        import yaml
        yaml.dump(config_content, f, default_flow_style=False)
    
    print(f"\n✅ Experiment '{name}' created successfully!")
    print(f"📁 Location: {experiment_path}")
    print("\nNext steps:")
    print("1. Update the README.md with your specific experiment details")
    print("2. Add your data to data/input/")
    print("3. Implement the experiment logic in src/run_experiment.py")
    print("4. Run the experiment and analyze results")
    
    return True


def create_standard_experiment(experiment_path: Path, name: str):
    """Create a standard experiment runner"""
    
    runner_content = f'''#!/usr/bin/env python3
"""
Run {name} Experiment
{'=' * (len(name) + 15)}

This script runs the {name} experiment following InsightSpike standards.
"""

import sys
from pathlib import Path

# Add templates to path
sys.path.append(str(Path(__file__).parent.parent.parent / "templates"))

from standard_experiment import StandardExperiment
from typing import Dict, Any


class {name.replace("_", " ").title().replace(" ", "")}Experiment(StandardExperiment):
    """Implementation of {name} experiment"""
    
    def _check_data_integrity(self) -> bool:
        """Check data integrity"""
        self.logger.info("Checking data integrity...")
        
        # TODO: Implement data validation
        # - Check if data exists
        # - Validate format
        # - Ensure no cheating/direct answers
        
        return True
    
    def prepare_data(self) -> Dict[str, Any]:
        """Prepare experiment data"""
        self.logger.info("Preparing data...")
        
        # TODO: Load and prepare your data
        data = {{
            'test': [],  # Test samples
            'metadata': {{
                'description': '{name} experiment data'
            }}
        }}
        
        return data
    
    def run_baseline(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run baseline method"""
        self.logger.info("Running baseline...")
        
        # TODO: Implement baseline
        results = {{
            'method': 'Baseline',
            'accuracy': 0.0,
            'results': []
        }}
        
        return results
    
    def run_proposed(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run InsightSpike method"""
        self.logger.info("Running InsightSpike...")
        
        # TODO: Use self.agent to process data
        results = {{
            'method': 'InsightSpike',
            'accuracy': 0.0,
            'results': []
        }}
        
        return results
    
    def evaluate_results(self, baseline: Dict, proposed: Dict) -> Dict[str, Any]:
        """Evaluate and compare results"""
        self.logger.info("Evaluating results...")
        
        # TODO: Compare baseline vs proposed
        metrics = {{
            'baseline_accuracy': baseline.get('accuracy', 0),
            'proposed_accuracy': proposed.get('accuracy', 0),
            'improvement': 0.0
        }}
        
        return metrics


if __name__ == "__main__":
    # Run the experiment
    experiment = {name.replace("_", " ").title().replace(" ", "")}Experiment("{name}")
    result = experiment.run()
    
    if result['success']:
        print(f"\\nExperiment completed! See results at: {{result['report_path']}}")
    else:
        print(f"\\nExperiment failed: {{result['error']}}")
'''
    
    with open(experiment_path / "src" / "run_experiment.py", "w") as f:
        f.write(runner_content)
    
    # Make it executable
    import os
    os.chmod(experiment_path / "src" / "run_experiment.py", 0o755)


def create_rat_experiment(experiment_path: Path, name: str):
    """Create a RAT experiment runner"""
    
    runner_content = f'''#!/usr/bin/env python3
"""
Run {name} RAT Experiment
{'=' * (len(name) + 19)}

Remote Associates Test experiment following InsightSpike standards.
"""

import sys
from pathlib import Path

# Add templates to path
sys.path.append(str(Path(__file__).parent.parent.parent / "templates"))

from rat_experiment_template import RATExperiment


class {name.replace("_", " ").title().replace(" ", "")}RATExperiment(RATExperiment):
    """Custom RAT experiment implementation"""
    
    def __init__(self):
        super().__init__("{name}")
        
        # You can customize RAT problems here
        # self.answer_words = ["CUSTOM", "WORDS", "HERE"]


if __name__ == "__main__":
    # Run the experiment
    experiment = {name.replace("_", " ").title().replace(" ", "")}RATExperiment()
    result = experiment.run()
    
    if result['success']:
        print(f"\\nRAT experiment completed! See results at: {{result['report_path']}}")
    else:
        print(f"\\nRAT experiment failed: {{result['error']}}")
'''
    
    with open(experiment_path / "src" / "run_experiment.py", "w") as f:
        f.write(runner_content)
    
    # Make it executable
    import os
    os.chmod(experiment_path / "src" / "run_experiment.py", 0o755)


def create_qa_experiment(experiment_path: Path, name: str):
    """Create a Q&A experiment runner"""
    # Similar structure to standard but focused on Q&A
    create_standard_experiment(experiment_path, name)


def create_performance_experiment(experiment_path: Path, name: str):
    """Create a performance experiment runner"""
    # Similar structure but with timing and resource tracking
    create_standard_experiment(experiment_path, name)


def main():
    parser = argparse.ArgumentParser(
        description="Create a new InsightSpike experiment from template"
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Name of the experiment (use underscores for spaces)"
    )
    parser.add_argument(
        "--type",
        default="standard",
        choices=["standard", "rat", "qa", "performance"],
        help="Type of experiment to create"
    )
    
    args = parser.parse_args()
    
    success = create_experiment(args.name, args.type)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()