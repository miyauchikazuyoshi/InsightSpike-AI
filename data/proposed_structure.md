# Proposed Data Directory Structure

## Current Issues
1. Core files (index.faiss, episodes.json, graph_pyg.pt) are missing from root
2. Database files are scattered in root directory
3. No clear separation between temporary and permanent data
4. No versioning or migration system
5. Experimental data mixed with core data

## Proposed New Structure

```
data/
├── core/                    # Core system files (persistent)
│   ├── index.faiss         # FAISS vector index
│   ├── episodes.json       # Episode metadata
│   ├── graph_pyg.pt        # PyTorch graph
│   └── scalable_index.faiss # Scalable FAISS index
│
├── db/                      # Databases (persistent)
│   ├── insight_facts.db    # Insight registry
│   └── unknown_learning.db # Auto-learning system
│
├── learning/                # Auto-learning data (persistent)
│   ├── unknown_concepts.json
│   ├── learning_sessions.json
│   └── concept_relationships.json
│
├── experiments/             # Experiment results (archivable)
│   ├── integrated_rag_memory/
│   └── [timestamp]_[name]/
│
├── cache/                   # Temporary cache (deletable)
│   ├── embeddings/
│   └── models/
│
├── raw/                     # Input data (read-only)
│   └── corpus/
│
├── processed/              # Processed data (regenerable)
│   ├── qa_pairs.json
│   └── test_questions.json
│
├── logs/                   # Log files (rotatable)
│   ├── system/
│   └── graph_operations/
│
├── backup/                 # Backups (archivable)
│   └── [timestamp]/
│
├── temp/                   # Temporary files (auto-cleaned)
│
├── config.yaml            # Data paths configuration
├── validate.py            # Data validation script
├── cleanup.py             # Cleanup utility
└── README.md              # Documentation
```

## Migration Plan

### Phase 1: Create Structure
```bash
mkdir -p data/{core,db,learning,experiments,backup,temp}
```

### Phase 2: Move Files
```bash
# Move core files (when they exist)
mv data/index.faiss data/core/
mv data/episodes.json data/core/
mv data/graph_pyg.pt data/core/

# Move databases
mv data/insight_facts.db data/db/
mv data/unknown_learning.db data/db/

# Move experiments
mv data/integrated_rag_memory_experiments data/experiments/
```

### Phase 3: Update Code
Update all path references in the codebase to use the new structure.

### Phase 4: Add Utilities
- Create `config.yaml` for centralized path management
- Add `validate.py` for data integrity checks
- Add `cleanup.py` for maintenance

## Benefits

1. **Clear Organization**: Each data type has its designated location
2. **Easy Backup**: Can backup just `core/` and `db/` for essential data
3. **Better Cleanup**: Can safely delete `temp/` and old `experiments/`
4. **Version Control**: Can track `config.yaml` for path changes
5. **Scalability**: Structure supports growth without confusion

## Implementation Priority

1. **High**: Move databases to `db/` - prevents accidental deletion
2. **High**: Create `core/` for essential files - clear persistence
3. **Medium**: Organize experiments - better result management
4. **Low**: Add utilities - nice to have but not critical