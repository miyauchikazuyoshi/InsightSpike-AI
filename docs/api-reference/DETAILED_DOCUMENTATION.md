# Detailed Documentation

This document contains comprehensive information about InsightSpike-AI's implementation, configuration, and advanced features.

## 📁 Project Structure

```text
InsightSpike-AI/
├── src/insightspike/           # Core 4-layer architecture implementation
│   ├── core/                   # InsightSpikeSystem, Memory Manager, Graph Reasoner
│   ├── models/                 # geDIG algorithm, neural networks, vector quantization
│   ├── memory/                 # FAISS-indexed episodic memory with C-value weighting
│   ├── graph/                  # PyTorch Geometric GNN reasoning
│   └── utils/                  # Utilities and helper functions
├── scripts/                    # Production utilities & enterprise tools
│   ├── debugging/              # System diagnostics
│   ├── testing/                # Component tests
│   ├── validation/             # Quality assurance
│   ├── production/             # Production deployment tools
│   ├── utilities/              # Data restore
│   ├── ci/                     # CI support
│   └── git-hooks/              # Pre-push validation automation
├── monitoring/                 # Real-time system monitoring
│   ├── production_monitor.py   # System health metrics
│   └── performance_dashboard.py # Web dashboard
├── templates/                  # Production integration templates
│   ├── production_integration_template.py
│   └── generated/              # Enterprise, Research, Educational, Content, Real-time
├── benchmarks/                 # Performance benchmarking suite
│   ├── performance_suite.py    # Comprehensive testing
│   └── results/                # Benchmark execution history
├── data/                       # Core data & enterprise backup system
│   ├── clean_backup/           # Clean state backup & restore
│   ├── episodes.json           # Episode memory
│   ├── graph_pyg.pt            # PyTorch graph data
│   ├── index.faiss             # FAISS vector index
│   ├── index.json              # Metadata index
│   └── *.db                    # SQLite databases
├── english_insight_experiment/ # Latest experimental results
├── docs/                       # Documentation & research
├── experiments/                # Research validation & analysis
└── tests/                      # Comprehensive test suite
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
  llm_provider: "local"                   # local, openai, anthropic
  llm_model: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  max_tokens: 256                         # LLM response length
  temperature: 0.3                        # Response creativity (0.0-1.0)
  device: "cpu"                          # cpu, cuda, mps
  use_gpu: false                         # Enable GPU acceleration
  safe_mode: false                       # Use mock providers for testing
```

#### **Memory System Configuration**

```yaml
memory:
  max_retrieved_docs: 15                 # Maximum documents per retrieval
  short_term_capacity: 10                # Recent interactions buffer
  working_memory_capacity: 20            # Active processing capacity
  episodic_memory_capacity: 60           # Long-term episode storage
  pattern_cache_capacity: 15             # Pattern recognition cache
```

#### **geDIG Algorithm Parameters**

```yaml
reasoning:
  # Core geDIG Weights
  weight_ged: 1.0                       # Graph Edit Distance weight
  weight_ig: 1.0                        # Information Gain weight
  weight_conflict: 0.5                  # Conflict detection weight
  
  # Episode Integration (Smart Memory)
  episode_integration_similarity_threshold: 0.85  # Vector similarity ≥ 0.85
  episode_integration_content_threshold: 0.4      # Content overlap ≥ 0.4
  episode_integration_c_threshold: 0.3            # C-value difference ≤ 0.3
  
  # Episode Management
  episode_merge_threshold: 0.8          # Merge similar episodes
  episode_split_threshold: 0.3          # Split conflicting episodes
  episode_prune_threshold: 0.1          # Remove low-value episodes
```

#### **Graph Processing & Spike Detection**

```yaml
graph:
  spike_ged_threshold: 0.5              # GED threshold for "Aha!" moments
  spike_ig_threshold: 0.2               # IG threshold for insights
  use_gnn: false                        # Enable Graph Neural Networks
  gnn_hidden_dim: 64                    # GNN layer dimensions
```

### 🎛️ Configuration Priority

Settings are applied in the following order (later overrides earlier):

1. **Default Values** (`src/insightspike/core/config.py`)
2. **YAML File** (`~/.insightspike/config.yaml`)
3. **Environment Variables** (`INSIGHTSPIKE_*`)
4. **CLI Arguments** (`--option value`)

### 📋 Configuration Presets

**Research Mode (High Accuracy):**
```yaml
retrieval:
  top_k: 25
  similarity_threshold: 0.25
reasoning:
  episode_integration_similarity_threshold: 0.9
  weight_ged: 1.2
  weight_ig: 1.2
```

**Production Mode (Fast Response):**
```yaml
retrieval:
  top_k: 10
  similarity_threshold: 0.4
reasoning:
  episode_integration_similarity_threshold: 0.8
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
reasoning:
  weight_conflict: 0.8  # Emphasize conflict detection
```

## 🔧 Advanced Usage

### CLI Command Reference

#### New Improved CLI (`spike`) - Recommended ✨

```bash
# Query the knowledge base
poetry run spike query "What is quantum computing?"
poetry run spike q "What is quantum computing?"  # alias

# Embed documents into the knowledge base (with graph updates)
poetry run spike embed path/to/documents.txt
poetry run spike e path/to/documents.txt  # alias

# Interactive chat mode
poetry run spike chat
poetry run spike c  # alias

# Configuration management
poetry run spike config show                    # Show current config
poetry run spike config set safe_mode false     # Change settings
poetry run spike config preset experiment       # Use preset
poetry run spike config save my_config.json    # Save config
poetry run spike config load my_config.json    # Load config

# Show statistics and insights
poetry run spike stats                         # Agent statistics
poetry run spike insights                      # Show discovered insights
poetry run spike insights-search "quantum"     # Search insights by concept

# Interactive demo
poetry run spike demo                          # Run guided demo

# Run experiments
poetry run spike experiment --name simple --episodes 10
poetry run spike experiment --name insight --episodes 5
poetry run spike experiment --name math --episodes 7

# Show version and help
poetry run spike version                       # Version info
poetry run spike --help                        # Show all commands
```

#### Legacy CLI (`insightspike`) - Limited Functionality

```bash
# Basic commands (with deprecation warnings)
poetry run insightspike legacy-ask "What is quantum computing?"
poetry run insightspike legacy-stats

# Limited functionality commands
poetry run insightspike load-documents path/to/documents.txt  # No graph update
poetry run insightspike config-info                           # Show config
poetry run insightspike deps list                            # Dependency management

# Show help
poetry run insightspike --help
```

### Python API Reference

#### Standard Data Management for Experiments

```python
import shutil
from datetime import datetime

# 1. Backup existing data before experiment
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
shutil.copytree("data", f"data_backup_{timestamp}")

# 2. Initialize fresh agent for clean experiment
agent = MainAgent()
agent.initialize()

# 3. Run your experiment
# ... experiment code ...

# 4. Save experiment results
experiment_results = {
    "timestamp": timestamp,
    "metrics": agent.get_stats(),
    # ... other results ...
}

# 5. Optionally restore original data
# shutil.rmtree("data")
# shutil.copytree(f"data_backup_{timestamp}", "data")
```

#### Data Growth Example

```python
from insightspike.core.agents.main_agent import MainAgent

agent = MainAgent()
agent.initialize()

# Load test data
documents = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with multiple layers.",
    "Transformers revolutionized natural language processing."
]

# Add with graph updates
for doc in documents:
    result = agent.add_episode_with_graph_update(doc)
    if result['success']:
        print(f"✓ Added: {doc[:50]}...")

# Check growth
initial_stats = agent.get_stats()
print(f"Total episodes: {initial_stats['episodes']}")
print(f"Graph nodes: {initial_stats['graph_nodes']}")

# MUST save to persist
agent.save_state()
```

### Key API Differences

| Feature | CLI | Python API |
|---------|-----|------------|
| Add documents | ✓ | ✓ |
| Update graph | ✗ | ✓ (with `add_episode_with_graph_update`) |
| Save data | ✗ | ✓ (with `save_state`) |
| Query processing | ✓ | ✓ |
| Full control | ✗ | ✓ |

### Data Storage Structure

InsightSpike-AI uses a structured data directory system:

- `data/episodes.json` - Episode memory (text, embeddings, metadata)
- `data/graph_pyg.pt` - PyTorch Geometric graph structure
- `data/index.faiss` - FAISS vector index for similarity search
- `data/insight_facts.db` - SQLite database for discovered insights
- `data/learning/` - Auto-learning system data

## 🔧 Development Setup

### Enable Pre-Push Validation

```bash
# Enable pre-push validation (recommended for contributors)
cp scripts/git-hooks/pre-push .git/hooks/
chmod +x .git/hooks/pre-push

# Restore clean data state if needed
python scripts/utilities/restore_clean_data.py

# Monitor system health
python monitoring/production_monitor.py

# Run performance benchmarks
python benchmarks/performance_suite.py
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

## 📊 Experimental Results Details

### 🎯 Latest Production Validation (January 2025)

#### **Integrated Production System**

- ✅ **Data Integrity**: Clean backup system with 5 core data files validated
- ✅ **Monitoring Infrastructure**: Production-ready system health monitoring
- ✅ **Git Integration**: Pre-push validation hooks ensure code quality
- ✅ **Production Templates**: 5 deployment scenarios validated
- ✅ **Performance Benchmarking**: CI-compatible testing suite

#### **Core System Validation Results**

**Architecture Component Testing:**

- 🔧 **Memory Manager**: Episode integration thresholds (0.85 similarity, 0.7 content) validated
- 📊 **Graph Reasoner**: PyTorch Geometric implementation with 1-node baseline
- ⚡ **Vector Search**: FAISS-indexed 384-dimensional embeddings optimized
- 🎯 **System Integration**: All 4 layers functioning in production environment

#### **Smart Episode Integration**

- **Threshold-based Decision**: Vector similarity ≥ 0.85, Content overlap ≥ 0.7
- **Integration Score**: 0.5×Similarity + 0.3×Content + 0.2×C-Value
- **Dynamic Memory**: FAISS-indexed efficient search with C-value weighting

### 📊 Historical Experimental Results

**Proof-of-Concept Validation (2025-06-30):**

- **Performance Improvement**: +133.3% quality increase in controlled experiments
- **Insight Detection**: Unique capability demonstrated vs baseline systems
- **Processing Efficiency**: Significant speed improvements observed
- **Statistical Confidence**: Results significant at p < 0.001 level

## 🏗️ Technical Architecture Details

### Core Architecture Layers

1. **Error Monitor** (Cerebellum analog) - Query analysis and validation
2. **Memory Manager** (Hippocampus analog) - Graph-centric episodic memory (C-value free)
3. **Graph Reasoner** (Prefrontal cortex analog) - Scalable PyTorch Geometric GNN with geDIG
4. **Language Interface** (Language area analog) - Natural language synthesis and interaction

### Scalable Graph Implementation

**Hierarchical Architecture for Large-Scale Processing:**

```text
IntegratedHierarchicalManager
├── GraphCentricMemoryManager (Episode Management)
│   ├── Dynamic importance from graph structure
│   ├── Graph-informed integration/splitting
│   └── No C-values - pure graph-based
└── HierarchicalGraphBuilder (Scalable Search)
    ├── Level 0: Individual episodes
    ├── Level 1: Topic clusters (√n size)
    └── Level 2: Super-clusters
```

**Performance Characteristics:**

| Dataset Size | Build Time | Search Time | Compression |
|-------------|------------|-------------|-------------|
| 1,000       | 150ms      | 0.5ms       | 100x        |
| 10,000      | 1.5s       | 2ms         | 200x        |
| 100,000     | 15s        | 5ms         | 500x        |

### Key Technologies

- **geDIG Algorithm**: Graph Edit Distance + Information Gain for insight detection
- **Scalable Graph Builder**: FAISS-based O(n log n) construction, O(log n) search
- **Graph-Centric Memory**: Dynamic importance, no C-values, self-attention-like behavior
- **Hierarchical Management**: 3-layer structure for 100K+ episode handling
- **Vector Quantization**: FAISS-indexed 384-dimensional embeddings
- **Dynamic Reasoning**: PyTorch Geometric graph neural networks

## 🙏 Acknowledgments

This research builds on insights from neuroscience, graph theory, and bio-inspired computing. Special thanks to the open-source community for foundational tools and libraries.