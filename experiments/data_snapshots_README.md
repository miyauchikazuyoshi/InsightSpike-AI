# Data Snapshots Format Guide

This document describes the standard format for experiment data snapshots as required by CLAUDE.md.

## Directory Structure

Each snapshot follows this structure:
```
snapshot_YYYYMMDD_HHMMSS/
├── metadata.json         # Experiment metadata and statistics
├── episodes.json         # All episodes with C-values (optional)
├── hypotheses.json       # All hypotheses with evolution data (optional)
├── custom_data.json      # Experiment-specific data (optional)
├── data/                 # Copy of experiment data directory
│   ├── input/           # Original input data
│   └── processed/       # Processed data during experiment
└── results/             # Copy of experiment results
    ├── metrics/         # Performance metrics
    ├── outputs/         # Generated outputs
    └── visualizations/  # Charts and plots
```

## File Formats

### metadata.json
```json
{
  "timestamp": "2025-01-25T10:30:00",
  "experiment_path": "/path/to/experiment",
  "experiment_name": "Enhanced RAT Benchmark",
  "total_episodes": 1000,
  "total_hypotheses": 100,
  "data_integrity": "ok",
  "custom_metrics": {
    "accuracy": 0.85,
    "insight_reuse_rate": 0.45
  }
}
```

### episodes.json
```json
{
  "episode_id_1": {
    "text": "The connection between X and Y reveals...",
    "c_value": 0.85,
    "metadata": {
      "question": "Original question that triggered this",
      "timestamp": "2025-01-25T10:15:00",
      "refinement_count": 3
    }
  }
}
```

### hypotheses.json
```json
[
  {
    "id": "hyp_001",
    "text": "Abstract pattern: When X then Y",
    "c_value": 0.92,
    "support_count": 15,
    "contradiction_count": 2,
    "metadata": {
      "evolution_history": ["initial", "refined", "validated"],
      "domain": "physics"
    }
  }
]
```

## Knowledge Transfer

To transfer knowledge between experiments:

```python
from experiments.utils import SnapshotManager

# Load high-quality insights from one experiment
stats = SnapshotManager.load_knowledge_from_snapshot(
    snapshot_path=Path("snapshot_20250725_030534"),
    target_agent=my_agent,
    c_threshold=0.7  # Only transfer insights with C > 0.7
)

print(f"Transferred {stats['transferred_hypotheses']} hypotheses")
```

## Best Practices

1. **Always include metadata.json** with at least timestamp and experiment name
2. **Use consistent naming** for snapshot directories (YYYYMMDD_HHMMSS)
3. **Include data integrity check** results in metadata
4. **Preserve full episode/hypothesis data** for knowledge transfer
5. **Document custom_data.json** structure in experiment README

## Snapshot Lifecycle

1. **Creation**: At experiment completion via `finalize_experiment()`
2. **Storage**: Keep indefinitely for reproducibility
3. **Transfer**: Use for bootstrapping new experiments
4. **Archive**: Compress older snapshots if needed

## Example Usage

```python
from experiments.utils import SnapshotManager

# At end of experiment
SnapshotManager.finalize_experiment(
    experiment_path=Path("."),
    store=my_custom_store,
    additional_metadata={
        "model": "claude-3-haiku",
        "benchmark": "RAT-20"
    }
)
```