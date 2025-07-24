# InsightSpike-AI: geDIG Framework Implementation

**Graph Structure-Based Insight Generation Framework**

[![License: InsightSpike Responsible AI](https://img.shields.io/badge/License-InsightSpike--Responsible--AI--1.0-blue)](https://github.com/miyauchikazuyoshi/InsightSpike-AI/blob/main/LICENSE)  
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/dependency-Poetry-blue)](https://python-poetry.org/)
[![Paper](https://img.shields.io/badge/Paper-geDIG--v3-green)](docs/paper/geDIG_paper_ja.tex)

![InsightSpike Knowledge Graph Animation](experiments/comprehensive_gedig_evaluation/results/visualizations/insight_detection_animation_latest.gif)

*Knowledge graph reorganization when insight is detected (99.5% confidence) - showing how concepts connect in real-time when asked "What is the fundamental nature of reality?"*

## 🎯 What is InsightSpike-AI?

InsightSpike-AI implements the **geDIG (Graph Edit Distance + Information Gain)** framework - the first system to use structural graph changes as intrinsic rewards for insight detection. By integrating structural novelty (ΔGED) and information compression (ΔIG), it achieves what traditional AI cannot: **detecting multi-concept integration that characterizes human "Aha!" moments**.

### 🧠 4-Layer Neurobiological Architecture

```mermaid
graph TD
    Query[User Query] --> L1[Layer 1: Error Monitor<br/>Cerebellum Analog]
    L1 --> L2[Layer 2: Memory Manager<br/>Hippocampus Analog]
    L2 --> L3[Layer 3: Graph Reasoner<br/>Prefrontal Cortex Analog]
    L3 --> L4[Layer 4: Language Interface<br/>Broca's/Wernicke's Analog]
    L4 --> Response[Insight Response]
    
    L3 -.->|Spike Detection| Insight[💡 Eureka Moment!]
```

### 🔬 Key Innovation: The geDIG Formula (with Temporal Consistency)

```
F = w₁·ΔGED - kT·ΔIG

where:
ΔGED = GED(G_after, G_before)  // Instantaneous structural change
ΔIG = H(G_before) - H(G_after)  // Instantaneous information gain
```

Key innovation:
- **ΔGED**: Measures structural change from immediately previous state (negative = simplification/insight)
- **ΔIG**: Measures entropy reduction from previous state (positive = information organization)
- **F**: Combined reward with temporally consistent measurements

### 📊 Breakthrough Results (Updated with Instantaneous ΔGED)

**Difficulty Reversal Phenomenon**: Unlike traditional AI, InsightSpike-AI shows **higher accuracy on harder questions**:

| Difficulty | Accuracy | N | Key Finding |
|------------|----------|---|-------------|
| Easy       | 75%      | 4 | Single-concept retrieval |
| Medium     | 81.8%    | 11| Local structure changes |
| **Hard**   | **100%** | 5 | Multi-concept integration detected |

- **85% overall accuracy** (17/20 questions) - maintained after theoretical consistency fix
- **37ms average processing** (CPU-only: Intel i7-9750H) - 18% faster with instantaneous calculations
- **84.1% average confidence** - Higher confidence with consistent measurements
- **99.5% peak confidence** on fundamental reality question
- **Temporal consistency**: Both ΔGED and ΔIG now measure instantaneous changes

### 🔬 Real Example: Multi-Concept Integration

**Question**: "What is the essence of reality - matter, energy, or information?"

**InsightSpike Detection**:
- **Spike Confidence**: 99.5%
- **ΔGED**: -2.3 (structural simplification)
- **ΔIG**: 0.61 (information organization)
- **Integrated Concepts**: Quantum mechanics, Information theory, Entropy, Wave function, Observer effect

The system detected deep connections across 5 hierarchical knowledge layers, demonstrating true multi-concept integration rather than simple retrieval.

## 🚀 Quick Start

### Python Code (Simplest)

```python
from insightspike import create_agent

# Create agent with zero configuration
agent = create_agent()

# Add knowledge
agent.add_knowledge("The Earth orbits around the Sun")
agent.add_knowledge("Gravity causes objects to attract each other")

# Ask questions
result = agent.process_question("Why do planets move?")
print(result.response)
```

### Different LLM Providers

```python
from insightspike import create_agent

# Use OpenAI (requires OPENAI_API_KEY environment variable)
agent = create_agent(provider="openai")

# Use mock provider for testing
agent = create_agent(provider="mock")

# Use small local model (CPU-friendly, ~77MB)
# ⚠️ Note: Local provider is currently not implemented. Use mock or external APIs.
# agent = create_agent(provider="local", model="google/flan-t5-small")
```

### Interactive Demo

```python
from insightspike import quick_demo
quick_demo()  # Runs an interactive demonstration
```

### Google Colab (Full Features)

**⚡ One-Step Setup:**

```python
# Run this single cell to set up everything
!git clone https://github.com/miyauchikazuyoshi/InsightSpike-AI.git
%cd InsightSpike-AI
!bash scripts/colab/setup_unified.sh
```

**🧪 Quick Test:**

```python
# Verify installation
!python -c "from insightspike.implementations.agents import MainAgent; print('✅ InsightSpike-AI Ready!')"

# Try it out
!python scripts/validation/complete_system_validation.py
```

### Local Installation (Advanced)

```bash
# Clone and install
git clone https://github.com/miyauchikazuyoshi/InsightSpike-AI.git
cd InsightSpike-AI

# Basic installation
pip install insightspike-ai

# With specific provider support
pip install insightspike-ai[openai]     # For OpenAI
pip install insightspike-ai[local]      # For local models
pip install insightspike-ai[all]        # Everything

# Using Poetry (for development)
poetry install
poetry run python scripts/setup_models.py

# CLI usage
spike demo
spike query "What is consciousness?"
```

### 🚀 Try the Comprehensive geDIG Experiment

Experience the full 85% accuracy insight detection:

```bash
# Run the comprehensive evaluation
cd experiments/comprehensive_gedig_evaluation
poetry run python src/run_evaluation.py

# See insights being generated in real-time!
```

## 🌟 Key Features

### 1. **geDIG Technology**
Mathematical modeling of insight generation using Graph Edit Distance + Information Gain.

### 2. **Configurable Agent System**
- Multiple agent implementations:
  - **MainAgent**: Full-featured agent with in-memory storage
  - **DataStoreAgent**: Scalable agent with transaction-based persistence
- 6 operation modes: Basic, Enhanced, Query Transform, Advanced, Optimized, Graph-Centric
- Feature toggles for fine-grained control
- Production-ready with caching and async processing

### 3. **Clean Architecture**
- Clear separation between abstractions and implementations
- Dependency injection ready
- Easy to test and extend

### 4. **Flexible Data Persistence**
- Abstract DataStore interface
- Support for filesystem, databases, vector stores
- Easy migration between backends

### 5. **Graph-Based Reasoning**
- PyTorch Geometric integration
- Similarity graph construction
- Spike detection algorithms

**Note**: Full graph reasoning capabilities require PyTorch and PyTorch Geometric. The system will work without them but with reduced graph analysis features.

## 🌟 Key Innovation: geDIG Technology

### The Science Behind Insights

InsightSpike uses **geDIG** (Graph Edit Distance + Information Gain) to mathematically model insight generation:

- **ΔGED**: Measures how knowledge structure simplifies during understanding (negative values = better)
- **ΔIG**: Quantifies information entropy changes as concepts connect (positive values = learning)
- **Reward Formula**: R = w₁×ΔGED + w₂×ΔIG (simplified from previous 3-term formula)
- **Spike Detection**: Triggers when both metrics indicate significant restructuring

### Multi-Phase Knowledge Integration

The system organizes knowledge into 5 phases, mimicking human cognitive development:

1. **Basic Concepts** - Fundamental building blocks
2. **Relationships** - Connections between concepts  
3. **Deep Integration** - Complex understanding
4. **Emergent Insights** - Novel discoveries
5. **Integration & Circulation** - Unified knowledge

When knowledge from 3+ different phases converges on a query, InsightSpike detects an insight opportunity.

## 📊 Proven Results

### 🏆 Latest Experiment: Temporally Consistent geDIG (July 2025)

After addressing theoretical consistency issues, our latest validation shows **maintained performance**:

| Metric | Result | Details |
|--------|--------|---------|
| **Insight Detection Rate** | **85%** | 17 out of 20 complex questions (unchanged) |
| **Average Confidence** | **84.1%** | Higher confidence with consistent measurements |
| **Processing Time** | **37ms avg** | 18% faster with instantaneous calculations |
| **Knowledge Graph** | **100 nodes, 962 edges** | Rich semantic network maintained |
| **Peak Confidence** | **99.5%** | "What is the fundamental nature of reality?" |

**Top Performing Examples:**
- **Q: "Can consciousness emerge from quantum processes?"**
  - ✅ Spike Detected (confidence: 91.6%)
  - Connectivity ratio: 0.84, Phase diversity: 0.80
  - Connected 10 concepts across 4 knowledge phases
  
- **Q: "How does information theory relate to thermodynamics?"**
  - ✅ Spike Detected (confidence: 84.5%)
  - Integrated concepts: Maxwell's demon, Shannon entropy, fundamental reality

- **Q: "What happens at the intersection of chaos and order?"**
  - ✅ Spike Detected (confidence: 91.0%)
  - Bridged concepts: Complexity, emergence, self-organization, consciousness

### 🎯 Validated Capabilities

| Capability | Achievement | Evidence |
|------------|-------------|----------|
| **Insight Detection** | 85% accuracy | Successfully identifies conceptual integration across phases |
| **Knowledge Graph** | 100 nodes, 962 edges | Dense semantic network with avg 0.57 similarity |
| **Real-time Processing** | 37ms average | 18% faster with instantaneous calculations |
| **Multi-phase Integration** | 4-phase system | Tracks knowledge evolution from basic to transcendent |
| **High Confidence** | 84.1% average | Strong signals for genuine insights |

**Key Achievements:**

- 🧠 **Insight Detection**: 85% of complex questions trigger genuine insight discovery
- 🔗 **Knowledge Graph Construction**: Builds rich semantic networks with ~10 edges per node
- ⚡ **Ultra-Efficient**: Average 37ms processing time enables real-time applications
- 🎯 **Conceptual Integration**: Detects cross-phase connections and emergent understanding
- 📊 **Quantifiable Metrics**: Connectivity ratio, phase diversity, and similarity measures

**Types of Insights Detected:**

1. **Conceptual Connections**: Energy ↔ Information ↔ Entropy relationships
2. **Cross-domain Links**: Consciousness ↔ Information Theory bridges
3. **Emergent Patterns**: Knowledge graph reorganizes when insights are detected

These insights demonstrate InsightSpike's ability to **discover hidden connections** between concepts.

### Why This Matters: Graph-Based Understanding

Unlike traditional approaches, InsightSpike uses graph structure to detect insights:

**Traditional Approach**:
- Searches for similar text
- Returns matching documents
- No understanding of relationships

**InsightSpike Approach**:
- Builds knowledge graph from concepts
- Calculates graph edit distance (GED) changes
- Detects information gain (IG) spikes
- Identifies when disparate concepts suddenly connect

For example, when asked about energy and information:
- Traditional: Returns documents mentioning both terms
- InsightSpike: Detects that Energy, Information, and Entropy form a fundamental conceptual unity through Maxwell's demon and Shannon entropy (confidence: 84.5%)


## 🏗️ Architecture Overview

### Clean Architecture Design

```
src/insightspike/
├── core/                    # 🎯 Abstract Interfaces & Contracts
│   ├── base/               # Base interfaces for all components
│   ├── contracts/          # System-wide contracts
│   ├── episode.py          # Episode data structures
│   ├── error_handler.py    # Core error handling
│   └── exceptions.py       # Custom exceptions
│
├── implementations/         # 🔧 Concrete Implementations
│   ├── agents/            # Agent implementations
│   ├── layers/            # 4-layer brain-inspired components
│   ├── memory/            # Memory management
│   └── datastore/         # Data persistence
│
├── features/               # ✨ Feature Modules
│   ├── query_transformation/
│   └── graph_reasoning/
│
├── algorithms/             # 🧮 Core Algorithms
│   ├── graph_edit_distance.py
│   └── information_gain.py
│
├── providers/              # 🤖 LLM Providers
│   ├── openai_provider.py
│   ├── anthropic_provider.py
│   └── local_provider.py   # Local model support
│
├── cli/                    # 💻 Command Line Interface
│   ├── spike.py           # Main CLI entry point
│   └── commands/          # CLI commands
│
├── config/                 # ⚙️ Configuration System
│   ├── models.py          # Pydantic config models
│   ├── loader.py          # Config loading logic
│   └── constants.py       # System constants
│
└── tools/                  # 🛠️ Standalone Tools
    ├── standalone/
    └── experiments/
```

### Data Store Abstraction

```python
# Flexible data persistence - easily swap backends
datastore = DataStoreFactory.create("filesystem")  # or "postgresql", "pinecone", etc.
datastore.save_episodes(episodes)
```


## 📖 Usage

### Quick Examples

**CLI (New `spike` command):**

```bash
# Ask questions and get insights
poetry run spike query "What is the relationship between energy and information?"

# Add new knowledge with graph updates
poetry run spike embed documents.txt

# Interactive chat mode
poetry run spike chat
```

**Python API:**

```python
from insightspike.implementations.agents import MainAgent

# Initialize agent
agent = MainAgent()
agent.initialize()

# Process a question
result = agent.process_question("How does quantum entanglement relate to information theory?")

print(f"Response: {result.response}")
print(f"Spike Detected: {result.spike_detected}")
if result.spike_detected:
    print(f"Graph Analysis: {result.graph_analysis}")
    print(f"Reasoning Quality: {result.reasoning_quality}")
```

### Advanced Configuration

```python
from insightspike.implementations.agents import ConfigurableAgent, AgentMode

# Use specific agent mode
agent = ConfigurableAgent(mode=AgentMode.GRAPH_CENTRIC)

# Or with custom configuration
from insightspike.config import InsightSpikeConfig

config = InsightSpikeConfig(
    core={"llm_model": "gpt-4"},
    reasoning={"spike_threshold": 0.7},
    datastore={"type": "postgresql", "params": {...}}
)
agent = ConfigurableAgent(config=config)
```

### Using DataStoreAgent for Scalable Deployments

```python
from insightspike.implementations.agents.datastore_agent import DataStoreMainAgent
from insightspike.implementations.datastore.sqlite_store import SQLiteDataStore

# Create a persistent datastore
datastore = SQLiteDataStore("knowledge.db")

# Initialize DataStoreAgent for scalable, transaction-based processing
agent = DataStoreMainAgent(datastore=datastore)

# Process questions with automatic persistence
result = agent.process("What is consciousness?")
print(f"Response: {result['response']}")
print(f"Spike detected: {result.get('has_spike', False)}")
```

### 📚 Full Documentation

- **[Complete CLI Reference](docs/DETAILED_DOCUMENTATION.md#cli-command-reference)** - All commands, options, and examples
- **[Python API Guide](docs/DETAILED_DOCUMENTATION.md#python-api-reference)** - MainAgent methods, data management, best practices
- **[Configuration Guide](docs/DETAILED_DOCUMENTATION.md#configuration--settings)** - YAML settings, presets, troubleshooting

## 🔬 Applications

- **Research**: Discover hidden connections in scientific literature
- **Education**: Detect when students achieve true understanding
- **Enterprise**: Generate strategic insights from disparate business data
- **Innovation**: Accelerate R&D by finding non-obvious connections

## 📄 License & Patents

Licensed under the InsightSpike AI Responsible Use License v1.0.

**Patent Pending:**

- JP 特願2025-082988: "ΔGED／ΔIG 内発報酬生成方法および生成ＡＩ装置 "
- JP 特願2025-082989: "階層ベクトル量子化による動的エピソードメモリ方法および生成ＡＩ装置 "

## 📧 Contact

- **Email**: `miyauchikazuyoshi@gmail.com`
- **Issues**: [GitHub Issues](https://github.com/miyauchikazuyoshi/InsightSpike-AI/issues)

## 🙏 Acknowledgments

- Brain-inspired architecture based on neuroscience research
- Community contributors and testers
- Open source dependencies that make this possible

---

**"Connecting the dots to create new knowledge"** 🧠✨

## 📊 Project Structure

```
InsightSpike-AI/
├── src/insightspike/       # Main package (see detailed structure above)
├── experiments/            # Research experiments & validations
├── data/                   # Data directory
│   ├── cache/             # Cached data
│   ├── knowledge_base/    # Knowledge base files
│   ├── logs/              # System and operation logs
│   ├── models/            # Model files
│   └── sqlite/            # SQLite databases
├── docs/                   # Documentation
│   ├── architecture/      # Architecture docs
│   ├── development/       # Development guides
│   ├── diagrams/          # Technical diagrams
│   └── research/          # Research notes
├── scripts/               # Utility scripts
│   ├── benchmarks/        # Performance benchmarks
│   ├── colab/             # Google Colab scripts
│   ├── debugging/         # Debug utilities
│   ├── monitoring/        # Production monitoring
│   └── validation/        # Validation scripts
├── tests/                 # Test suites
│   ├── unit/              # Unit tests
│   ├── fixtures/          # Test data
│   └── development/       # Development test scripts
├── config/                # Configuration files
│   └── examples/          # Example configs
├── benchmarks/            # Benchmark results
└── examples/              # Usage examples
```

## 🧪 Testing

```bash
# Run all tests
poetry run pytest

# Run specific test suite
poetry run pytest tests/test_core_interfaces.py

# Run with coverage
poetry run pytest --cov=insightspike
```

## 📖 Documentation

- [Architecture Overview](docs/architecture/README.md)
- [Layer Architecture](docs/architecture/layer_architecture.md)
- [Agent Types](docs/architecture/agent_types.md)
- [CLI Commands](docs/architecture/cli_commands.md)
- [Configuration Guide](docs/development/config_migration.md)
- [Data Store Guide](data/README.md)

## 🔄 Recent Updates (July 2025)

### geDIG Paper v3 Release! 📄
- **Strong Accept** from peer review with Best Paper Award recommendation
- Complete theoretical framework with mathematical proofs
- Experimental validation showing 100% accuracy on complex questions
- Statistical analysis with confidence intervals and p-values

### Technical Implementation
- **ΔGED Approximation**: A* search with ε≤0.1 error bound
- **ΔIG Calculation**: Clustering-based entropy using silhouette scores
- **Real-time Performance**: 45ms on Intel i7-9750H (CPU-only)
- **Scalability**: Sub-linear growth - 15% increase for 10x data

## 🚦 Development Status

- ✅ Core architecture refactored
- ✅ Configuration system unified
- ✅ Data persistence abstracted
- ✅ Legacy code cleaned up
- ✅ **Temporal consistency achieved** (ΔGED and ΔIG use same time scale)
- ✅ **Performance maintained** after theoretical fixes
- 🚧 Test coverage expansion
- 🚧 Documentation updates

## 📖 Research Paper

**"geDIG: Graph Structure-Based Insight Generation Framework"**

Our research introduces the theoretical foundation and experimental validation of the geDIG framework. Key contributions:

- First framework to use structural graph changes (ΔGED) as intrinsic rewards
- Discovery of "difficulty reversal phenomenon" - harder questions yield higher accuracy
- Mathematical formulation inspired by hippocampal replay and synaptic pruning
- 85% overall accuracy with 100% on complex multi-concept questions

📄 [Read the full paper](docs/paper/geDIG_paper_ja.tex) | 🔬 [View experiments](experiments/)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md).

### Development Setup

```bash
# Install with dev dependencies
poetry install --with dev

# Run linters
poetry run ruff check .
poetry run mypy .

# Format code
poetry run black .
```

## 📚 Additional Documentation

- [Technical Architecture](docs/diagrams/README.md)
- [Contributing Guide](CONTRIBUTING.md)
- [Data Management](data/README.md)
- [Experimental Results](experiments/comprehensive_gedig_evaluation/)
- [geDIG Research Paper](docs/paper/)

## ⚙️ Configuration

InsightSpike-AI uses YAML configuration files for customization:

```bash
# User config (recommended)
~/.insightspike/config.yaml

# Project config
./config.yaml

# Custom path
export INSIGHTSPIKE_CONFIG_PATH="/path/to/config.yaml"
```

Key settings include:

- Model selection and parameters
- Memory capacity and thresholds
- Insight detection sensitivity
- Performance optimization options

For detailed configuration options, see the [Detailed Documentation](docs/DETAILED_DOCUMENTATION.md).