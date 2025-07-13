#!/bin/bash
# Helper script to create new experiments

if [ $# -eq 0 ]; then
    echo "Usage: ./create_experiment.sh <experiment_name>"
    echo "Example: ./create_experiment.sh causal_inference/covid_impact_analysis"
    exit 1
fi

EXPERIMENT_NAME=$1
EXPERIMENT_PATH="experiments/$EXPERIMENT_NAME"

# Create directory structure
echo "Creating experiment structure for: $EXPERIMENT_NAME"
mkdir -p "$EXPERIMENT_PATH"/{src,data/{input,processed},results/{metrics,outputs,visualizations},data_snapshots}

# Create README template
cat > "$EXPERIMENT_PATH/README.md" << EOF
# Experiment: $EXPERIMENT_NAME

## Overview
- **Created**: $(date +%Y-%m-%d)
- **Author**: [Enter name]
- **Status**: 🔄 In preparation

## Purpose
[Describe the hypothesis to test or goals to achieve]

## Background
[Why this experiment is needed, which aspects of InsightSpike to validate]

## Methods
### Data
- **Input data**: 
- **Data source**: 
- **Preprocessing**: 

### Algorithm
[Describe methods and models used]

### Evaluation Metrics
- [ ] Metric 1: 
- [ ] Metric 2: 
- [ ] Metric 3: 

## How to Run
\`\`\`bash
cd $EXPERIMENT_PATH
python src/main_experiment.py --config config.yaml
\`\`\`

## Dependencies
\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Experiment Log
### $(date +%Y-%m-%d)
- Created experiment folder

## Results
[Document experiment results here]

### Key Findings
- 

### Performance Metrics
| Metric | Value | Notes |
|--------|-------|-------|
| | | |

## Discussion
[Interpretation and implications]

## Next Steps
- [ ] 
- [ ] 

## References
- 
EOF

# Create basic Python files
cat > "$EXPERIMENT_PATH/src/main_experiment.py" << 'EOF'
#!/usr/bin/env python3
"""
Main experiment script
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

def setup_experiment(config_path: str = None):
    """Setup experiment environment"""
    experiment_info = {
        "start_time": datetime.now().isoformat(),
        "config": config_path,
        "python_version": sys.version,
        "working_dir": os.getcwd()
    }
    
    # Create result directories
    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs("results/outputs", exist_ok=True)
    os.makedirs("results/visualizations", exist_ok=True)
    
    return experiment_info

def run_experiment():
    """Main experiment logic"""
    print("Starting experiment...")
    
    # TODO: Implement main experiment logic here
    
    results = {
        "status": "completed",
        "metrics": {},
        "outputs": []
    }
    
    return results

def save_results(results, experiment_info):
    """Save experiment results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Combine experiment info and results
    full_results = {
        "experiment_info": experiment_info,
        "results": results,
        "end_time": datetime.now().isoformat()
    }
    
    # Save as JSON
    output_path = f"results/outputs/experiment_results_{timestamp}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {output_path}")
    
    return output_path

def create_snapshot(metadata):
    """Create snapshot of experiment data"""
    from snapshot_utils import create_data_snapshot
    create_data_snapshot(".", metadata)

def main():
    parser = argparse.ArgumentParser(description='Run experiment')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--snapshot', action='store_true', help='Create data snapshot after experiment')
    args = parser.parse_args()
    
    # Setup
    experiment_info = setup_experiment(args.config)
    
    try:
        # Run experiment
        results = run_experiment()
        
        # Save results
        output_path = save_results(results, experiment_info)
        
        # Create snapshot if requested
        if args.snapshot:
            create_snapshot({
                "experiment": os.path.basename(os.getcwd()),
                "results_file": output_path,
                "status": results.get("status", "unknown")
            })
            
    except Exception as e:
        print(f"Error during experiment: {e}")
        raise

if __name__ == "__main__":
    main()
EOF

# Create snapshot utility
cat > "$EXPERIMENT_PATH/src/snapshot_utils.py" << 'EOF'
"""Data snapshot creation utility"""

import shutil
import json
from datetime import datetime
import os

def get_directory_size(path):
    """Get directory size in bytes"""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total += os.path.getsize(filepath)
    return total

def create_data_snapshot(experiment_path, metadata):
    """Create snapshot of experiment data"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_dir = f"{experiment_path}/data_snapshots/snapshot_{timestamp}"
    
    # Only copy if data folder exists and has content
    data_path = f"{experiment_path}/data"
    if os.path.exists(data_path) and os.listdir(data_path):
        print(f"Creating snapshot of {data_path}...")
        shutil.copytree(data_path, snapshot_dir)
        
        # Extend metadata
        metadata['timestamp'] = timestamp
        metadata['data_size_mb'] = round(get_directory_size(snapshot_dir) / 1024 / 1024, 2)
        metadata['file_count'] = sum(len(files) for _, _, files in os.walk(snapshot_dir))
        
        # Save metadata
        metadata_path = f"{experiment_path}/data_snapshots/snapshot_{timestamp}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"Snapshot created: {snapshot_dir}")
        print(f"Metadata saved: {metadata_path}")
    else:
        print("No data to snapshot")
EOF

# .gitignoreを作成
cat > "$EXPERIMENT_PATH/.gitignore" << EOF
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# Virtual environments
venv/
env/
ENV/

# Large data files
data/input/*.pkl
data/input/*.csv
data/input/*.json
data/input/*.h5
data/input/*.parquet
data/processed/*.pkl
data/processed/*.h5

# Temporary files
*.tmp
*.log
.DS_Store

# Keep structure but ignore large snapshots
data_snapshots/snapshot_*/
!data_snapshots/*_metadata.json

# Jupyter
.ipynb_checkpoints/
*.ipynb

# IDE
.vscode/
.idea/
EOF

# requirements.txtテンプレートを作成
cat > "$EXPERIMENT_PATH/requirements.txt" << EOF
# InsightSpike core dependencies
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=0.24.0

# Visualization
matplotlib>=3.4.0
seaborn>=0.11.0

# Data processing
tqdm>=4.62.0

# Add experiment-specific dependencies below
EOF

# Create config file template
cat > "$EXPERIMENT_PATH/config.yaml" << EOF
# Experiment configuration
experiment:
  name: $EXPERIMENT_NAME
  version: "1.0.0"
  
data:
  input_path: "data/input/"
  processed_path: "data/processed/"
  
parameters:
  # Add experiment-specific parameters here
  
output:
  save_intermediate: true
  visualization: true
EOF

echo "✅ Created experiment folder: $EXPERIMENT_PATH"
echo ""
echo "Next steps:"
echo "1. cd $EXPERIMENT_PATH"
echo "2. Edit README.md to add experiment details"
echo "3. Implement experiment logic in src/main_experiment.py"
echo "4. Run experiment with: python src/main_experiment.py"