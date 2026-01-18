# Detailed Documentation

This document contains comprehensive information about InsightSpike-AI's implementation, configuration, and advanced features.

## 📁 Project Structure

```text
InsightSpike-AI/
├── src/insightspike/           # Core 4-layer architecture implementation
│   ├── core/                   # Base interfaces and abstract classes only
│   │   ├── agents/             # Agent interfaces
│   │   ├── base/               # Base classes
│   │   └── interfaces/         # Layer interfaces
│   ├── implementations/        # Concrete implementations
│   │   ├── agents/             # MainAgent, ConfigurableAgent
│   │   ├── datastore/          # Storage implementations
│   │   └── layers/             # Layer implementations (L1-L4)
│   ├── index/                  # Integrated vector-graph index (NEW)
│   │   ├── __init__.py
│   │   ├── integrated_index.py # Core index implementation
│   │   ├── backward_compat.py  # Compatibility wrapper
│   │   └── migration.py        # Migration helpers
│   ├── features/               # Additional features
│   │   ├── graph_reasoning/    # Graph analysis tools
│   │   └── query_transformation/ # Query processing
│   ├── algorithms/             # Core algorithms (GED, IG, entropy)
│   ├── config/                 # Pydantic-based configuration
│   ├── cli/                    # Command line interfaces
│   │   ├── spike.py            # Main CLI
│   │   └── commands/           # CLI command modules
│   └── utils/                  # Utilities and helpers
├── scripts/                    # Setup and utility scripts
├── docs/                       # Documentation
│   ├── architecture/           # System architecture docs
│   ├── api-reference/          # API documentation
│   └── development/            # Development guides
├── data/                       # Core data (READ-ONLY for experiments)
│   ├── episodes.json           # Episode memory
│   ├── graph.pt                # PyTorch graph data
│   ├── index.faiss             # FAISS vector index
│   └── *.db                    # SQLite databases
├── experiments/                # Isolated experiment directories
│   ├── templates/              # Experiment templates
│   └── [experiment_name]/      # Individual experiments
│       ├── src/                # Experiment code
│       ├── data/               # Experiment-specific data
│       │   ├── input/          # Copy of source data
│       │   └── processed/      # Generated data
│       ├── results/            # Experiment results
│       └── data_snapshots/     # Data backups
└── tests/                      # Test suite
    ├── unit/                   # Unit tests
    └── integration/            # Integration tests
```

## ⚙️ Configuration & Settings

### 📄 YAML Configuration File

InsightSpike-AI uses flexible configuration management:

#### **Configuration Locations**

1. **User Home** (Personal settings):
   ```bash
   ~/.insightspike/config.yaml
   ```

2. **Project Root** (Project-specific):
   ```bash
   ./config.yaml
   export INSIGHTSPIKE_CONFIG_PATH="./config.yaml"
   ```

3. **Custom Path** (Environment-specific):
   ```bash
   export INSIGHTSPIKE_CONFIG_PATH="/path/to/your/config.yaml"
   ```

### 🔧 Configuration Sections

#### **Core Language Model Settings**

```yaml
core:
  model_name: "paraphrase-MiniLM-L6-v2"  # Embedding model (384-dim)
  llm_provider: "local"                   # local, openai, anthropic, mock, clean
  llm_model: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  max_tokens: 256                         # LLM response length
  temperature: 0.3                        # Response creativity (0.0-1.0)
  device: "cpu"                          # cpu, cuda, mps
  use_gpu: false                         # Enable GPU acceleration
```

#### **Memory System Configuration**

```yaml
memory:
  max_retrieved_docs: 15                 # Maximum documents per retrieval
  short_term_capacity: 10                # Recent interactions buffer
  working_memory_capacity: 20            # Active processing capacity
  episodic_memory_capacity: 60           # Long-term episode storage
  pattern_cache_capacity: 15             # Pattern recognition cache
  
  # Memory aging settings (NEW - July 2025)
  enable_aging: true                     # Enable time-based memory decay
  aging_factor: 0.95                     # Decay factor per day
  min_age_days: 7                        # Don't age episodes younger than this
  max_age_days: 90                       # Maximum age before auto-pruning
  prune_on_overflow: true                # Auto-prune when reaching max_episodes
```

#### **geDIG Algorithm Parameters**

```yaml
# Graph configuration (replaces 'reasoning' section)
graph:
  # Core geDIG Weights
  weight_ged: 1.0                       # Graph Edit Distance weight
  weight_ig: 1.0                        # Information Gain weight
  weight_conflict: 0.5                  # Conflict detection weight
  
  # Spike Detection Thresholds
  spike_ged_threshold: -0.5             # GED threshold for "Aha!" moments
  spike_ig_threshold: 0.2               # IG threshold for insights
  conflict_threshold: 0.5               # Conflict detection threshold
  
  # Episode Management (via similarity)
  episode_merge_threshold: 0.8          # Merge episodes with cosine similarity > 0.8
  episode_split_threshold: 0.3          # Split conflicting episodes
  episode_prune_threshold: 0.1          # Remove low-value episodes
  
  # Graph Neural Network
  use_gnn: false                        # Enable Graph Neural Networks
  gnn_hidden_dim: 64                    # GNN layer dimensions
```

#### **Graph Processing & Spike Detection**

```yaml
```

### 🎛️ Configuration Priority

Settings are applied in the following order (later overrides earlier):

1. **Default Values** (in Pydantic models)
2. **YAML/JSON File** (`./config.yaml` or `~/.insightspike/config.yaml`)
3. **Environment Variables** (`INSIGHTSPIKE_*`)
4. **CLI Arguments** (`--option value`)

### 📋 Configuration Presets

**Research Mode (High Accuracy):**
```yaml
retrieval:
  top_k: 25
  similarity_threshold: 0.25
graph:
  weight_ged: 1.2
  weight_ig: 1.2
memory:
  enable_aging: false  # More stable for research
```

**Production Mode (Fast Response):**
```yaml
retrieval:
  top_k: 10
  similarity_threshold: 0.4
memory:
  episodic_memory_capacity: 1000
  prune_on_overflow: true
processing:
  batch_size: 16
  timeout_seconds: 120
```

**Educational Mode (Explainable):**
```yaml
output:
  verbose: true
  generate_visualizations: true
  save_results: true
core:
  temperature: 0.5
graph:
  weight_conflict: 0.8  # Emphasize conflict detection
```

## 🔧 Advanced Usage

### CLI Command Reference

#### New Improved CLI (`spike`) - Recommended ✨

```bash
# Query the knowledge base
poetry run spike query "What is quantum computing?"

# Embed documents into the knowledge base (with graph updates)
poetry run spike embed path/to/documents.txt
poetry run spike embed path/to/directory/      # Process all .txt and .md files

# Interactive chat mode
poetry run spike interactive                   # Start interactive mode

# Configuration management
poetry run spike config show                   # Show current config
poetry run spike config set core.temperature 0.5  # Change nested settings
poetry run spike config preset development     # Use preset configuration
poetry run spike config save my_config.yaml    # Save current config
poetry run spike config load my_config.yaml    # Load saved config
poetry run spike config validate               # Validate configuration
poetry run spike config export                 # Export full config with defaults

# Show statistics and insights
poetry run spike stats                         # Agent statistics
poetry run spike insights                      # Show discovered insights (top 5)
poetry run spike insights --limit 10           # Show more insights

# Interactive demo
poetry run spike demo                          # Run guided demo

# Run experiments
poetry run spike experiment --name simple      # Basic Q&A experiment
poetry run spike experiment --name insight     # Insight detection experiment
poetry run spike experiment --name math        # Mathematical reasoning

# Advanced features
poetry run spike discover "quantum physics"    # Discover insights about a topic
poetry run spike bridge "quantum" "biology"    # Bridge concepts across domains
poetry run spike graph visualize               # Visualize knowledge graph
poetry run spike graph analyze                 # Analyze graph structure

# Show version and help
poetry run spike version                       # Version info
poetry run spike --help                        # Show all commands
```

#### Legacy CLI (`insightspike`) - Deprecated ⚠️

The legacy CLI has been removed as of July 2025. All functionality has been migrated to the new `spike` CLI with improved features:

- `insightspike ask` → `spike query`
- `insightspike load-documents` → `spike embed` (with graph updates)
- `insightspike stats` → `spike stats`
- `insightspike config-info` → `spike config show`

Please use the new `spike` CLI for all operations.

### Python API Reference

#### Standard Data Management for Experiments

```python
import shutil
from datetime import datetime
from pathlib import Path

# Following CLAUDE.md guidelines for experiments
experiment_dir = Path("experiments/my_experiment")

# 1. Copy data to experiment directory (NEVER modify project root data/)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
shutil.copytree("data", experiment_dir / "data/input")

# 2. Initialize agent with experiment-specific config
from insightspike.config import load_config
from insightspike.implementations.agents import MainAgent

config = load_config(preset="experiment")
agent = MainAgent(config=config)
agent.initialize()

# 3. Run your experiment
# ... experiment code ...

# 4. Save results to experiment directory
experiment_results = {
    "timestamp": timestamp,
    "metrics": agent.get_stats(),
    # ... other results ...
}

# 5. Create data snapshot for reproducibility
shutil.copytree(
    experiment_dir / "data",
    experiment_dir / f"data_snapshots/snapshot_{timestamp}"
)
```

#### Modern Usage Example

```python
from insightspike.implementations.agents import MainAgent
from insightspike.config import load_config
from insightspike.config.presets import ConfigPresets

# Option 1: Use preset
config = load_config(preset="development")

# Option 2: Load from file
# config = load_config(config_path="./config.yaml")

# Initialize agent
agent = MainAgent(config=config)
agent.initialize()

# Load test data
documents = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with multiple layers.",
    "Transformers revolutionized natural language processing."
]

# Add knowledge with graph updates
for doc in documents:
    result = agent.add_knowledge(doc)
    if result.get('success'):
        print(f"✓ Added: {doc[:50]}...")
        if result.get('spike_detected'):
            print("  🚀 Insight spike detected!")

# Process a question
answer = agent.process_question(
    "How do transformers relate to deep learning?",
    max_cycles=3
)

print(f"\nAnswer: {answer.response}")
print(f"Quality: {answer.reasoning_quality:.3f}")
print(f"Spike detected: {answer.spike_detected}")

# Check statistics
stats = agent.get_stats()
print(f"\nTotal episodes: {stats['episodes']}")
print(f"Graph nodes: {stats.get('graph_nodes', 0)}")

# Get insights
insights = agent.get_insights(limit=3)
for insight in insights.get('insights', []):
    print(f"\nInsight: {insight['fact']}")
    print(f"Category: {insight['category']}")

# MUST save to persist
agent.save_state()
```

### Key API Methods

| Method | Purpose | Returns |
|--------|---------|----------|
| `initialize()` | Initialize agent components | `bool` (success) |
| `add_knowledge(text, source)` | Add document with graph update | `Dict` with success, spike_detected |
| `process_question(question, max_cycles, verbose)` | Process query through all layers | `CycleResult` object |
| `get_stats()` | Get agent statistics | `Dict` with metrics |
| `get_insights(limit)` | Retrieve discovered insights | `Dict` with insights list |
| `search_insights(concept, limit)` | Search insights by concept | `List[Dict]` |
| `save_state()` | Persist agent state to disk | `None` |
| `age_episodes()` | Apply time-based memory decay | `int` (pruned count) |
| `run_experiment(type, episodes)` | Run built-in experiments | `Dict` with results |

### Data Storage Structure

InsightSpike-AI uses a structured data directory system:

- `data/episodes.json` - Episode memory (text, embeddings, metadata)
- `data/graph.pt` - PyTorch graph structure (renamed from graph_pyg.pt)
- `data/index.faiss` - FAISS vector index for similarity search
- `data/insight_facts.db` - SQLite database for discovered insights
- `data/index.json` - Metadata index for episodes

**Important**: Project root `data/` is READ-ONLY for experiments. Always copy to experiment directories as per CLAUDE.md guidelines.

## 🔧 Development Setup

### Development Commands

```bash
# Setup models (first time only)
poetry run python scripts/setup_models.py

# Run tests with coverage
poetry run pytest tests/unit/ -v --cov=src/insightspike --cov-report=term-missing

# Format code
poetry run black src/ tests/
poetry run isort src/ tests/

# Lint code
poetry run flake8 src/ tests/

# Type checking
poetry run mypy src/

# Create new experiment
poetry run python experiments/templates/create_experiment.py --name "my_experiment" --type "standard"
```

### Environment Troubleshooting

**Common Issues & Solutions:**

**1. CLI commands not found:**
```bash
# Make sure you're in Poetry shell
poetry shell
spike --help

# OR use poetry run prefix
poetry run spike --help

# If still not working, reinstall
poetry install
```

**2. Import errors in local development:**
```bash
# Activate Poetry environment first
poetry shell
# OR run commands within Poetry environment
poetry run python your_script.py
poetry run jupyter lab

# Manual PYTHONPATH (fallback only)
export PYTHONPATH="${PYTHONPATH}:/path/to/InsightSpike-AI/src"
```

**3. Version conflicts (especially NumPy/PyTorch):**
```bash
# Check conflicting versions
pip check

# Clean reinstall
poetry lock --no-update
poetry install
```

**4. Package installation fails:**
```bash
# Update pip and poetry
pip install --upgrade pip poetry poetry-core

# Clean Poetry cache
poetry cache clear --all pypi

# Alternative installation
pip install torch torchvision torchaudio faiss-cpu typer click pydantic
pip install -e .
```

## 📊 Recent Improvements (July 2025)

### 🎯 Code Quality Enhancements

#### **Configuration System Overhaul**
- ✅ **Unified Schema**: Removed dual config.reasoning management
- ✅ **Pydantic Models**: Type-safe configuration with validation
- ✅ **Preset System**: Development, experiment, production presets
- ✅ **Environment Override**: Full support for INSIGHTSPIKE_* variables

#### **Memory Management Improvements**
- 🔧 **Episode Merging**: Cosine similarity-based intelligent merging
- 📊 **Time-based Aging**: Exponential decay (0.95/day) for old memories
- ⚡ **Auto-pruning**: Configurable overflow handling
- 🎯 **Size Enforcement**: Automatic episode limit management

#### **Code Cleanup Results**
- **Removed**: 300+ lines of deprecated methods
- **Deleted**: Legacy functions (_detect_spike, save_graph, load_graph)
- **Fixed**: 18 instances of hasattr(config, "reasoning") checks
- **Improved**: Test coverage from 17% to 23%

### 📊 Algorithm Performance

**geDIG (Graph Edit Distance + Information Gain):**
- **Spike Detection**: ΔGED ≤ -0.5, ΔIG ≥ 0.2
- **Conflict Resolution**: Automatic detection and handling
- **Scalability**: O(n log n) construction, O(log n) search

**Memory Performance (Enhanced with Integrated Index):**
- **Episode Capacity**: 10,000+ episodes supported
- **Vector Search Speed**: O(1) with pre-normalized vectors (NEW)
  - Old: O(n) normalization + O(n) search
  - New: Direct O(1) cosine similarity
- **Spatial Search**: O(log n) for position-based queries (NEW)
- **Cache Performance**: ~80% hit rate with LRU cache
- **Merge Efficiency**: 80%+ similarity required for auto-merge

## 🏗️ Technical Architecture Details

### Core Architecture Layers

1. **Layer 1: Error Monitor** (`implementations/layers/layer1_error_monitor.py`)
   - Brain analog: Cerebellum
   - Query validation and uncertainty detection
   
2. **Layer 2: Memory Manager** (`implementations/layers/layer2_memory_manager.py`)
   - Brain analog: Hippocampus + Locus Coeruleus
   - Integrated vector-graph indexed episodic memory (NEW)
   - Intelligent episode merging and pruning with aging
   
3. **Layer 3: Graph Reasoner** (`implementations/layers/layer3_graph_reasoner.py`)
   - Brain analog: Prefrontal Cortex
   - geDIG algorithm for insight detection
   - PyTorch-based graph analysis
   
4. **Layer 4: Language Interface** (`implementations/layers/layer4_llm_interface.py`)
   - Brain analog: Broca's + Wernicke's areas
   - Multi-provider LLM support
   - Context-aware response generation

### Configuration Architecture

```text
config/
├── models.py          # Pydantic configuration models
├── loader.py          # Configuration loading logic
├── presets.py         # Pre-defined configurations
└── converter.py       # Legacy format conversion

index/                 # Integrated Vector-Graph Index (NEW)
├── integrated_index.py    # Core index with pre-normalized vectors
├── backward_compat.py     # 100% API compatibility wrapper
└── migration_helper.py    # Gradual migration support
```

### Key Algorithms

- **geDIG**: Graph Edit Distance + Information Gain
  - Detects structural changes in knowledge
  - Identifies "Eureka" moments
  - Optional analogy bonus (prototype + cross-domain; default off) adds to IG
  
- **Similarity Entropy**: Information-theoretic complexity
  - Measures knowledge diversity
  - Guides memory organization
  
- **Cosine Similarity**: Episode comparison
  - 384-dimensional embeddings (768 for newer models)
  - Pre-normalized vectors for O(1) search (NEW)
  - Threshold: 0.8 for merging

### Integrated Index Features (NEW - January 2025)

- **Dual Vector Management**: Stores normalized vectors + norms separately
  - Eliminates O(n) normalization bottleneck
  - Enables efficient raw vector reconstruction
  
- **Spatial Indexing**: Position-based O(log n) lookups
  - Essential for navigation tasks (maze solving)
  - Integrated with vector similarity search
  
- **Graph Integration**: NetworkX graph with similarity edges
  - Unified management of vectors and graph structure
  - Enables multi-hop reasoning with geDIG
  
- **FAISS Auto-switching**: Transparent optimization for scale
  - NumPy backend for < 100k vectors
  - FAISS backend for larger datasets
  
- **Performance Monitoring**: Built-in metrics collection
  - Search time percentiles (p50, p95, p99)
  - Cache hit rates
  - Memory usage tracking

## 🙏 Acknowledgments

This research builds on insights from neuroscience, graph theory, and bio-inspired computing. Special thanks to the open-source community for foundational tools and libraries.
# Detailed API Documentation

> Note (2025-09): For stable, user‑facing entry points, prefer the Public API in `insightspike.public` (see `docs/api-reference/public_api.md`). Internal modules referenced here are subject to change.
