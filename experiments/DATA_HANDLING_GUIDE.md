# Experiment Data Handling Guide

## Overview

This guide provides comprehensive instructions for managing experiment data in the InsightSpike-AI project. It covers data organization, backup strategies, version control best practices, and data lifecycle management.

## Directory Structure

```
experiments/
├── active_experiments/       # Currently running or recently completed experiments
├── archive/                  # Completed experiments (compressed)
├── data_preservation/        # Critical data backups
├── template/                 # Template for new experiments
└── colab_experiments/        # Google Colab-specific experiments
```

## Data Organization Standards

### 1. Experiment Naming Convention

All experiments should follow this naming pattern:
```
XX_descriptive_name/
```
- `XX`: Two-digit experiment number (e.g., 05, 10, 15)
- `descriptive_name`: Short, descriptive name using underscores

Examples:
- `05_episode_management_graph_growth`
- `08_large_scale_rag_1000`
- `11_graph_enhanced_rag_large`

### 2. Standard Experiment Structure

Each experiment directory should contain:
```
XX_experiment_name/
├── README.md                 # Experiment overview and objectives
├── data/                     # Input/output data
│   ├── raw/                 # Original input data
│   ├── processed/           # Processed datasets
│   └── results/             # Experiment outputs
├── src/                      # Source code
│   ├── main.py              # Main experiment script
│   ├── analysis/            # Analysis scripts
│   └── utils/               # Utility functions
├── results/                  # Results and visualizations
│   ├── figures/             # Plots and diagrams
│   ├── metrics/             # Performance metrics
│   └── reports/             # Summary reports
├── checkpoints/              # Model checkpoints (if applicable)
└── logs/                     # Execution logs
```

## Data Backup Strategy

### 1. Before Starting an Experiment

Always create a backup of existing data:
```bash
# Create backup directory with timestamp
mkdir -p data_backup_$(date +%Y%m%d_%H%M%S)

# Copy current data
cp -r data/* data_backup_$(date +%Y%m%d_%H%M%S)/
```

### 2. During Experiments

- **Checkpoint Frequently**: Save intermediate results every N iterations
- **Version Control**: Commit code changes regularly (exclude large data files)
- **Log Everything**: Maintain detailed logs of experiment progress

### 3. After Experiments

1. **Document Results**: Update experiment README with findings
2. **Clean Up**: Remove temporary files and failed attempts
3. **Archive**: Compress and move to archive if no longer active

## Large Data File Management

### 1. Git LFS (Large File Storage)

For files > 100MB, use Git LFS:
```bash
# Track large files
git lfs track "*.pkl"
git lfs track "*.npy"
git lfs track "*.h5"
git lfs track "*.parquet"
```

### 2. Data Exclusion

Add to `.gitignore`:
```
# Large data files
*.pkl
*.npy
*.h5
*.parquet
*.arrow
*.bin

# Temporary files
*.tmp
*.cache
__pycache__/

# Checkpoint files
checkpoints/
*.ckpt
```

### 3. External Storage

For very large datasets (> 1GB):
- Use cloud storage (Google Drive, S3)
- Document download instructions in README
- Include checksums for data integrity

## Data Versioning

### 1. Semantic Versioning for Datasets

Use version tags for datasets:
```
data/
├── dataset_v1.0/    # Initial version
├── dataset_v1.1/    # Minor updates
└── dataset_v2.0/    # Major changes
```

### 2. Metadata Tracking

Create `data_metadata.json` for each dataset:
```json
{
  "version": "1.0",
  "created_date": "2025-07-06",
  "source": "Wikipedia articles",
  "preprocessing": "tokenization, embedding",
  "size": "150MB",
  "num_samples": 10000,
  "checksum": "md5:abc123..."
}
```

## Data Pipeline Best Practices

### 1. Reproducibility

- **Random Seeds**: Always set and document random seeds
- **Environment**: Use requirements.txt or environment.yml
- **Configuration**: Store all parameters in config files

### 2. Data Validation

```python
# Example validation function
def validate_data(data_path):
    """Validate data integrity before experiments."""
    # Check file existence
    assert os.path.exists(data_path), f"Data not found: {data_path}"
    
    # Check data format
    data = load_data(data_path)
    assert isinstance(data, expected_type), "Invalid data format"
    
    # Check data completeness
    assert len(data) > 0, "Empty dataset"
    
    # Verify checksums if available
    if hasattr(data, 'checksum'):
        assert verify_checksum(data), "Checksum mismatch"
```

### 3. Data Loading Optimization

```python
# Use efficient data loading
import pandas as pd
import numpy as np

# For large CSV files
def load_large_csv(filepath, chunksize=10000):
    chunks = []
    for chunk in pd.read_csv(filepath, chunksize=chunksize):
        chunks.append(process_chunk(chunk))
    return pd.concat(chunks)

# For binary data
def load_binary_data(filepath):
    return np.load(filepath, mmap_mode='r')  # Memory-mapped for large files
```

## Experiment Data Lifecycle

### 1. Active Phase (0-2 weeks)
- Keep in `active_experiments/`
- Full data and code access
- Regular updates and modifications

### 2. Review Phase (2-4 weeks)
- Document all findings
- Clean up unnecessary files
- Prepare for archival

### 3. Archive Phase (> 4 weeks)
- Compress entire experiment directory
- Move to `archive/`
- Keep only essential results accessible

### 4. Deletion Policy
- Never delete without documentation
- Archive before deletion
- Maintain deletion log with reasons

## Data Recovery Procedures

### 1. From Git History
```bash
# Recover deleted file
git log --diff-filter=D --summary | grep delete
git checkout <commit>^ -- <file_path>
```

### 2. From Backups
```bash
# List available backups
ls -la data_backup_*

# Restore from backup
cp -r data_backup_20250706_120000/* data/
```

### 3. From Archives
```bash
# Extract archived experiment
tar -xzf archive/experiment_05.tar.gz -C active_experiments/
```

## Common Data Issues and Solutions

### 1. Memory Errors
- Use data generators/iterators
- Process in chunks
- Reduce precision (float64 → float32)

### 2. Disk Space
- Regular cleanup of temporary files
- Compress inactive data
- Use symbolic links for shared datasets

### 3. Data Corruption
- Implement checksums
- Regular validation checks
- Maintain multiple backups

## Monitoring and Alerts

### 1. Disk Usage Monitoring
```bash
# Check experiment sizes
du -sh experiments/active_experiments/*

# Find large files
find experiments/ -size +100M -type f
```

### 2. Data Integrity Checks
```python
# Automated integrity check script
def check_experiment_health(exp_dir):
    issues = []
    
    # Check required directories
    for required in ['data', 'src', 'results']:
        if not os.path.exists(f"{exp_dir}/{required}"):
            issues.append(f"Missing {required} directory")
    
    # Check for README
    if not os.path.exists(f"{exp_dir}/README.md"):
        issues.append("Missing README.md")
    
    # Check data size
    data_size = get_directory_size(f"{exp_dir}/data")
    if data_size > 1e9:  # 1GB
        issues.append(f"Large data directory: {data_size/1e9:.2f}GB")
    
    return issues
```

## Collaboration Guidelines

### 1. Sharing Experiments
- Document all dependencies
- Include setup instructions
- Provide sample data if full dataset is large

### 2. Concurrent Experiments
- Use separate branches for major experiments
- Coordinate data modifications
- Document shared resource usage

### 3. Code Reviews
- Review data processing code carefully
- Validate statistical methods
- Check for data leakage

## Emergency Procedures

### 1. Accidental Deletion
1. Stop all write operations immediately
2. Check git status for tracked files
3. Check backup directories
4. Use file recovery tools if needed

### 2. Corruption Detection
1. Run integrity checks
2. Isolate corrupted files
3. Restore from latest good backup
4. Document corruption cause

### 3. Out of Space
1. Identify large files: `du -sh * | sort -h`
2. Clean temporary files
3. Compress or archive old experiments
4. Move to external storage if needed

## Tools and Utilities

### 1. Data Validation
```bash
# experiments/scripts/validate_data.py
python validate_data.py --experiment 05
```

### 2. Backup Creation
```bash
# experiments/scripts/backup_experiment.sh
./backup_experiment.sh 05_episode_management
```

### 3. Archive Management
```bash
# experiments/scripts/archive_experiment.sh
./archive_experiment.sh 05_episode_management
```

## Conclusion

Following these guidelines ensures:
- **Reproducibility**: Experiments can be reliably reproduced
- **Traceability**: Clear history of changes and results
- **Scalability**: System can handle growing data needs
- **Recovery**: Data can be recovered from various failure modes

For questions or suggestions, please create an issue or contact the project maintainers.