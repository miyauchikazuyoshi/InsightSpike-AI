# Data Directory Structure

## Overview

This directory follows a structured organization for all InsightSpike data files:

```
data/
├── core/                    # Core system files (persistent)
│   ├── index.faiss         # FAISS vector index
│   ├── episodes.json       # Episode metadata  
│   ├── graph_pyg.pt        # PyTorch graph
│   └── graph_pyg_clean.pt  # Clean backup graph
│
├── db/                      # Databases (persistent)
│   ├── insight_facts.db    # Insight registry (40KB)
│   └── unknown_learning.db # Auto-learning system (108KB)
│
├── experiments/             # Experiment results (archivable)
│   ├── integrated_rag_memory/  # RAG experiments
│   └── processed_results/      # Analysis results
│       ├── comprehensive_rag_analysis.json
│       ├── experiment_results.json
│       ├── graph_visualization_results.json
│       ├── simple_metadata.json
│       └── test_questions.json
│
├── raw/                     # Input data (read-only)
│   ├── indirect_knowledge.txt
│   ├── insight_dataset.txt
│   ├── simple_dataset.txt
│   └── test_sentences.txt
│
├── clean_backup/            # Clean reference files
│   ├── episodes_clean.json
│   ├── graph_pyg_clean.pt
│   ├── index_clean.faiss
│   ├── insight_facts_clean.db
│   ├── unknown_learning_clean.db
│   └── README.md
│
├── logs/                   # Log files (rotatable)
│   ├── system/            # System logs (empty)
│   └── graph_operations/  # Graph ops logs (empty)
│
├── learning/               # Auto-learning data (empty)
├── cache/                  # Temporary cache (empty)
├── temp/                   # Temporary files (empty)
├── backup/                 # Backups (contains timestamps)
├── processed/              # Now empty (moved to experiments)
└── samples/                # Sample data
    └── benchmark_data.json
```

## Key Files

- `config.yaml` - Central configuration for data paths
- `validate.py` - Data validation script
- `cleanup.py` - Cleanup utility

## Core System Files

| File | Component | Description | Location |
|------|-----------|-------------|----------|
| `index.faiss` | Layer2 Memory Manager | Vector search index | `core/` |
| `episodes.json` | Layer2 Memory Manager | Episode metadata | `core/` |
| `graph_pyg.pt` | Layer3 Graph Reasoner | Knowledge graph | `core/` |
| `scalable_index.faiss` | Scalable Graph Manager | Scalable vectors | `core/` |
| `insight_facts.db` | Insight Registry | Insights database | `db/` |
| `unknown_learning.db` | Auto Learning System | Learning database | `db/` |

Component implementations are now in:
- `src/insightspike/implementations/layers/layer2_memory_manager.py`
- `src/insightspike/implementations/layers/layer3_graph_reasoner.py`
- `src/insightspike/implementations/memory/scalable_graph_manager.py`

## Usage

### Validate Data
```bash
python validate.py
```

### Clean Up
```bash
# Dry run to see what would be deleted
python cleanup.py --dry-run

# Actually clean up
python cleanup.py
```

### Data Store Integration

With the new DataStore abstraction:

```python
from insightspike.implementations.datastore import DataStoreFactory

# Create datastore (uses config.json settings)
datastore = DataStoreFactory.create("filesystem", base_path="data")

# Save episodes
datastore.save_episodes(episodes, namespace="l2_memory")

# Load episodes
episodes = datastore.load_episodes(namespace="l2_memory")

# Search vectors
indices, distances = datastore.search_vectors(query_vec, k=10)
```

## File Formats

### episodes.json
```json
[
  {
    "text": "Episode text content",
    "vec": [0.1, 0.2, ...],  // 384-dimensional vector
    "c": 0.5,                // C-value
    "timestamp": 1234567890,
    "metadata": {}
  }
]
```

### graph_pyg.pt
PyTorch tensor containing:
- `x`: Node features (embeddings)
- `edge_index`: Edge connectivity
- `num_nodes`: Total nodes

## Cleanup Policies

Based on `config.yaml`:

- **temp/**: Max 7 days, cleaned on startup
- **cache/**: Max 10GB, files older than 30 days
- **logs/**: Rotated at 100MB, kept for 90 days
- **experiments/**: Keep latest 10, archive older

## Migration from Old Structure

The data directory has been reorganized for better maintainability:

1. **Files moved to `core/`**:
   - `index.faiss`
   - `episodes.json`
   - `graph_pyg.pt`

2. **Files moved to `db/`**:
   - `insight_facts.db`
   - `unknown_learning.db`

3. **Experiment data moved to `experiments/`**

The old structure is preserved in `backup/` for reference.

## Best Practices

1. **Always use DataStore API** - Don't directly access files
2. **Regular validation** - Run `validate.py` weekly
3. **Cleanup before experiments** - Use `cleanup.py`
4. **Backup critical data** - Core and DB directories
5. **Monitor disk usage** - Especially cache and experiments

## Troubleshooting

**Missing files**: Run validation to check structure
```bash
python validate.py
```

**Disk space issues**: Run cleanup
```bash
python cleanup.py
```

**Corrupt files**: Restore from clean backup
```bash
cp clean_backup/episodes_clean.json core/episodes.json
cp clean_backup/index_clean.faiss core/index.faiss
cp clean_backup/graph_pyg_clean.pt core/graph_pyg.pt
cp clean_backup/insight_facts_clean.db db/insight_facts.db
cp clean_backup/unknown_learning_clean.db db/unknown_learning.db
```

## Future Extensions

The DataStore abstraction supports:
- PostgreSQL backend
- Vector databases (Pinecone, Weaviate)
- S3/Cloud storage
- Redis cache

Simply change the config:
```json
{
  "datastore": {
    "type": "postgresql",
    "params": {
      "connection_string": "..."
    }
  }
}
```