# Data Directory Structure

## Overview

This directory contains all data files for InsightSpike-AI. The structure is organized based on the DataStore abstraction layer.

## Current Directory Structure

```
data/
├── insight_store/          # Default DataStore location (when using filesystem)
│   ├── core/              # Core system files
│   ├── episodes/          # Episode data
│   ├── graphs/            # Graph structures
│   └── vectors/           # Vector indices
│
├── knowledge_base/         # Static knowledge files
│   ├── initial/           # Initial datasets
│   │   ├── indirect_knowledge.txt
│   │   ├── insight_dataset.txt
│   │   ├── simple_dataset.txt
│   │   └── test_sentences.txt
│   └── samples/           # Sample knowledge
│       ├── consciousness.txt
│       ├── entropy_info.txt
│       └── quantum_bio.txt
│
├── sqlite/                 # SQLite databases
│   ├── benchmark.db       # Benchmark results
│   ├── insightspike.db    # Main database
│   └── simple_benchmark.db
│
├── experiments/            # Experiment data and results
├── cache/                  # Temporary cache files
├── temp/                   # Temporary working files
├── logs/                   # Log files
│   ├── system/            # System logs
│   └── graph_operations/  # Graph operation logs
│
├── core/                   # Legacy core files (being migrated)
├── db/                     # Legacy databases (being migrated)
├── learning/               # Learning/training data
├── models/                 # Model files
│
├── MIGRATION_REPORT.md     # Data migration documentation
└── README.md              # This file
```

## DataStore Configuration

The DataStore location is configured in `config.yaml`:

```yaml
datastore:
  type: filesystem          # Options: filesystem, in_memory, sqlite
  root_path: ./data/insight_store
```

## Using DataStore

### Python API

```python
from insightspike.implementations.datastore import DataStoreFactory

# Create datastore (uses config.yaml settings)
datastore = DataStoreFactory.create("filesystem", base_path="data/insight_store")

# Save episodes
episodes = [
    {
        "text": "Example episode",
        "vec": embedding_vector,  # numpy array
        "c_value": 0.5,
        "timestamp": 1234567890,
        "metadata": {}
    }
]
datastore.save_episodes(episodes, namespace="my_experiment")

# Load episodes
loaded_episodes = datastore.load_episodes(namespace="my_experiment")

# Search vectors
results = datastore.search_vectors(
    query_vector=query_embedding,
    k=10,
    namespace="my_experiment"
)

# Save graph
datastore.save_graph(graph_data, namespace="my_experiment")

# Load graph
graph = datastore.load_graph(namespace="my_experiment")
```

### Namespace Organization

DataStore uses namespaces to organize data:

- `default`: Default namespace for MainAgent
- `l2_memory`: Layer 2 Memory Manager data
- `l3_graph`: Layer 3 Graph Reasoner data
- `experiments/[name]`: Experiment-specific data

## File Formats

### Episodes (JSON)
```json
[
  {
    "text": "Episode content",
    "vec": [0.1, 0.2, ...],      // 384-dimensional vector
    "c_value": 0.5,               // Confidence value
    "timestamp": 1234567890,
    "metadata": {
      "source": "user_input",
      "tags": ["example"]
    }
  }
]
```

### Graphs (PyTorch)
- Saved as `.pt` files using torch.save()
- Contains PyTorch Geometric Data objects

### Vector Indices
- NumPy arrays saved as `.npy` files
- Compatible with the VectorIndex abstraction

## Best Practices

1. **Use DataStore API**: Always use the DataStore API instead of direct file access
2. **Namespace isolation**: Use unique namespaces for experiments
3. **Clean up**: Remove experiment data after analysis
4. **Backup**: Regular backups of `insight_store/` directory

## Maintenance

### Clean temporary files
```bash
# Remove old cache files
find data/cache -type f -mtime +7 -delete

# Clear temp directory
rm -rf data/temp/*
```

### Retention (TTL) & Deletion Policy

- Cache (`data/cache/`): delete files older than 7 days
- Temp (`data/temp/`, `tmp/`): clear after each experiment batch
- Logs (`data/logs/`, `results/logs/`): rotate weekly; keep last 4 weeks
- Large artifacts (HTML viewers / step logs / SQLite):
  - Attach to a GitHub Release (e.g., `v0.1-assets`) and remove from the main branch
  - Keep only aggregated JSONs in `docs/paper/data/`
- Shadow migration (`migration_mode: shadow`): keep the shadow copies ≤14 days; aggregates remain the single source of truth

Recommended cron (example):
```cron
# Weekly cleanup (every Sunday 03:00)
0 3 * * 0 find data/cache -type f -mtime +7 -delete
0 3 * * 0 find results/logs -type f -mtime +28 -delete
```

### Check disk usage
```bash
du -sh data/*
```

## Migration Notes

The system is transitioning from direct file access to DataStore abstraction:

- **Old**: Direct file operations in `core/`, `db/`
- **New**: DataStore API with `insight_store/`

Legacy directories (`core/`, `db/`) are maintained for backward compatibility but will be phased out.

## Environment-Specific Configurations

Different environments can use different DataStore configurations:

```yaml
# Development
datastore:
  type: filesystem
  root_path: ./data/insight_store

# Testing  
datastore:
  type: in_memory
  
# Production
datastore:
  type: sqlite
  connection_string: "sqlite:///data/production.db"
```

## Troubleshooting

### Permission errors
```bash
# Fix permissions
chmod -R 755 data/
```

### Disk space issues
```bash
# Check large files
find data -type f -size +100M
```

### Data corruption
- Check logs in `data/logs/`
- Restore from backups if available
- Re-run experiments if needed
