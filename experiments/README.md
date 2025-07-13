# InsightSpike Experiments Directory

## 📊 Experiment Summary

### 1. quick_validation/
**Purpose**: Rapidly validate InsightSpike's core value proposition
- **simple_baseline_demo.py**: Traditional RAG vs InsightSpike comparison (achieved 66.7% insight detection rate)
- **three_way_comparison.py**: Demonstrates stepwise improvement: Base LLM < RAG < InsightSpike (3.7x quality improvement)
- **Results**: Proven that InsightSpike significantly outperforms traditional methods

### 2. causal_inference/
**Purpose**: Evaluate and implement causal reasoning capabilities
- Experiments on causal relationship detection and inference
- Planned for future implementation

### 3. pattern_recognition/
**Purpose**: Implement pattern recognition and anomaly detection
- Pattern extraction from time-series and structured data
- Planned for future implementation

### 4. scaling_tests/
**Purpose**: Verify scalability and performance
- Performance evaluation on large-scale datasets
- Planned for future implementation

## 📁 Data Management Policy

### Experiment Directory Structure
Each experiment follows this standard structure:

```
experiments/
├── experiment_name/           # Experiment folder
│   ├── src/                  # Experiment programs
│   │   ├── main_experiment.py
│   │   └── utils.py
│   ├── data/                 # Experiment data
│   │   ├── input/           # Input data
│   │   └── processed/       # Processed data
│   ├── results/             # Experiment results
│   │   ├── metrics/         # Evaluation metrics
│   │   ├── outputs/         # Output files
│   │   └── visualizations/  # Graphs and charts
│   ├── data_snapshots/      # Data folder backups
│   │   ├── snapshot_YYYYMMDD_HHMMSS/
│   │   └── snapshot_metadata.json
│   └── README.md            # Detailed experiment description
```

### Data Management Rules

1. **Experiment Independence**
   - Each experiment managed in separate folder
   - Minimize dependencies on other experiments

2. **Version Control**
   - Experiment programs (src/) managed with Git
   - Large data files added to .gitignore
   - Important configurations managed in config files

3. **Data Snapshots**
   - Backup data/ folder to data_snapshots/ after experiment completion
   - Timestamped folder names (snapshot_YYYYMMDD_HHMMSS)
   - Save metadata (experiment conditions, parameters) as JSON

4. **Result Storage**
   - Systematically save all experiment results in results/
   - Include both English and Japanese headers in CSV files (UTF-8 with BOM)
   - Save visualizations in reproducible formats

5. **Documentation**
   - Create README.md in each experiment folder
   - Document purpose, methods, results, and insights
   - Clearly describe reproduction steps

### Experiment Creation Template

Steps to start a new experiment:

```bash
# 1. Create experiment folder
mkdir -p experiments/new_experiment/{src,data/{input,processed},results/{metrics,outputs,visualizations},data_snapshots}

# 2. Create README template
cat > experiments/new_experiment/README.md << EOF
# Experiment: [Experiment Name]

## Purpose
[Describe experiment purpose]

## Methods
[Describe experimental methods]

## How to Run
\`\`\`bash
python src/main_experiment.py
\`\`\`

## Results
[Document main results]

## Discussion
[Interpretation and next steps]
EOF

# 3. Create basic .gitignore
cat > experiments/new_experiment/.gitignore << EOF
# Large data files
data/input/*.pkl
data/input/*.h5
data/processed/*.pkl

# Temporary files
*.tmp
__pycache__/

# Keep snapshots metadata only
data_snapshots/*/
!data_snapshots/*.json
EOF
```

### Data Backup Procedure

Creating backups after experiment completion:

```python
import shutil
import json
from datetime import datetime
import os

def create_data_snapshot(experiment_path, metadata):
    """Create snapshot of experiment data"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_dir = f"{experiment_path}/data_snapshots/snapshot_{timestamp}"
    
    # Copy data folder
    shutil.copytree(f"{experiment_path}/data", snapshot_dir)
    
    # Save metadata
    metadata['timestamp'] = timestamp
    metadata['data_size_mb'] = get_directory_size(snapshot_dir) / 1024 / 1024
    
    with open(f"{experiment_path}/data_snapshots/snapshot_{timestamp}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"Snapshot created: {snapshot_dir}")
```

## 🎯 Future Directions

1. **Systematize Experiments**
   - Adopt standard structure across experiment categories (causal_inference, pattern_recognition, etc.)
   - Facilitate knowledge sharing between experiments

2. **Promote Automation**
   - Automate experiment execution, result collection, and report generation
   - Integration with CI/CD pipelines

3. **Ensure Reproducibility**
   - Dockerize experiment environments
   - Clear dependency management (requirements.txt)

4. **Result Visualization**
   - Adopt standardized dashboard formats
   - Interactive result exploration tools