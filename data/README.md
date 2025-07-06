# Data Directory Structure

This directory contains all data files used by InsightSpike-AI. The structure and usage patterns are documented below to ensure consistency across all experiments and components.

## Directory Structure

```
data/
├── raw/                    # Raw input data (corpus files, etc.)
├── processed/              # Processed data files
├── embedding/              # Embedding models cache
├── logs/                   # Application logs
├── learning/               # Auto-learning system data
├── models/                 # Model checkpoints
├── experiments/            # Experiment-specific data
├── index.faiss            # FAISS vector index
├── episodes.json          # Episode metadata
├── graph_pyg.pt           # PyTorch Geometric graph
├── insight_facts.db       # SQLite database for insights
└── scalable_index.faiss   # Scalable FAISS index (when using ScalableGraphManager)
```

## File Types and Their Usage

### Core System Files

| File | Component | Description | Format |
|------|-----------|-------------|--------|
| `index.faiss` | L2MemoryManager | Vector search index | FAISS binary |
| `episodes.json` | L2MemoryManager | Episode metadata and text | JSON |
| `graph_pyg.pt` | L3GraphReasoner | Knowledge graph structure | PyTorch tensor |
| `scalable_index.faiss` | ScalableGraphManager | Scalable vector index | FAISS binary |
| `insight_facts.db` | InsightRegistry | Discovered insights database | SQLite |

### Auto-Learning System Files

Located in `data/learning/`:
- `unknown_concepts.json` - Concepts to learn
- `learning_sessions.json` - Learning history
- `concept_relationships.json` - Concept graph
- `auto_learned_knowledge.txt` - Accumulated knowledge

### Log Files

Located in `data/logs/`:
- `graph_operations/graph_operations_YYYYMMDD.jsonl` - Daily graph operation logs
- Application logs (various components)

## Path Configuration

All paths are configured in `src/insightspike/core/config.py`:

```python
@dataclass
class PathConfig:
    data_dir: Path = "data/raw"      # Raw data directory
    log_dir: Path = "data/logs"      # Log directory
    index_file: Path = "data/index.faiss"
    graph_file: Path = "data/graph_pyg.pt"
```

## Usage Guidelines

### 1. Loading Data

```python
# Load existing memory state
agent = MainAgent(config)
success = agent.load_state()  # Loads from configured paths

# Or specify custom path
agent.l2_memory.load(Path("data/backup/index.faiss"))
```

### 2. Saving Data

```python
# Save current state
agent.save_state()  # Saves to configured paths

# Or specify custom path
agent.l2_memory.save(Path("data/backup/index.faiss"))
```

### 3. File Format Specifications

#### episodes.json
```json
[
  {
    "id": 0,
    "text": "Episode text content",
    "c": 0.5,
    "metadata": {},
    "vec": [0.1, 0.2, ...]  // 384-dimensional vector
  }
]
```

#### graph_pyg.pt
PyTorch tensor containing:
- `x`: Node features (embeddings)
- `edge_index`: Edge connectivity
- `num_nodes`: Total nodes in graph

### 4. Best Practices

1. **Always use configured paths** - Don't hardcode paths in experiments
2. **Create directories before use** - Use `path.parent.mkdir(parents=True, exist_ok=True)`
3. **Handle missing files gracefully** - Check existence before loading
4. **Use consistent formats** - JSON for metadata, FAISS for vectors, PyTorch for graphs
5. **Clean up experiments** - Remove temporary files after experiments

### 5. Data Lifecycle

1. **Initialization**: Empty directories, no files
2. **Training**: Files created as episodes are added
3. **Persistence**: Save state before shutting down
4. **Recovery**: Load state on startup
5. **Backup**: Copy critical files before major changes

### 6. Troubleshooting

**Issue**: "File not found" errors
- **Solution**: Ensure data directory exists and paths are correctly configured

**Issue**: Incompatible file versions
- **Solution**: Check file format matches expected schema

**Issue**: Large file sizes
- **Solution**: Use `prune()` methods to remove low-value episodes

## Experimental Data Management

For experiments, follow this pattern:

```python
# 1. Backup existing data
shutil.copytree("data", f"data_backup_{timestamp}")

# 2. Run experiment with clean state
agent = MainAgent(config)
# ... run experiment ...

# 3. Save results
results_path = f"experiments/results/exp_{timestamp}.json"
save_results(results, results_path)

# 4. Restore if needed
shutil.copytree(f"data_backup_{timestamp}", "data")
```

## Notes

- The `raw/` subdirectory contains nested copies from previous experiments - these can be cleaned up
- FAISS indices can become large with many episodes - monitor disk usage
- SQLite databases should be properly closed to avoid corruption
- PyTorch files (.pt) require PyTorch to be installed for loading