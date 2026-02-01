# InsightSpike Directory Structure

> **Last Updated**: 2026-02-01

## 📁 Overview

InsightSpike follows a clean architecture with clear separation of concerns. The most recent refactoring (February 2026) modularized the geDIG package into 10 specialized modules.

## 🗂️ Directory Layout

```
src/insightspike/
├── core/                        # Core interfaces and base classes ONLY
│   ├── agents/                 # Agent interfaces
│   │   └── generic_agent.py    # GenericReasoner interface
│   ├── base/                   # Base classes
│   │   └── datastore.py       # Abstract DataStore interface
│   ├── interfaces/             # Layer interfaces
│   │   ├── generic_interfaces.py  # EnvironmentState, etc.
│   │   ├── layer_interfaces.py    # L1-L4 interfaces
│   │   └── __init__.py
│   ├── episode.py             # Core Episode data structure
│   └── error_handler.py       # Exception classes and logging
│
├── implementations/            # All concrete implementations
│   ├── agents/                # Agent implementations
│   │   ├── main_agent.py      # MainAgent for Q&A (primary)
│   │   └── configurable_agent.py  # Configurable agent
│   ├── layers/                # Layer implementations (unified)
│   │   ├── layer1_error_monitor.py     # L1: Error detection
│   │   ├── layer2_memory_manager.py    # L2: Memory + aging
│   │   ├── layer2_compatibility.py     # L2: Backward compat
│   │   ├── layer2_working_memory.py    # L2: DataStore variant
│   │   ├── layer3_graph_reasoner.py    # L3: Graph analysis
│   │   ├── layer4_llm_interface.py     # L4: LLM providers
│   │   ├── layer4_prompt_builder.py    # L4: Prompt generation
│   │   └── scalable_graph_builder.py   # Graph construction
│   └── datastore/             # Storage implementations
│       ├── filesystem_store.py  # FileSystem storage
│       └── __init__.py
│
├── features/                  # Optional feature modules
│   ├── graph_reasoning/       # Graph analysis features
│   │   ├── graph_analyzer.py  # Graph metrics calculator
│   │   └── reward_calculator.py  # Reward computation
│   └── query_transformation/  # Query processing (unused)
│
├── algorithms/                # Core algorithms
│   ├── gedig/                # geDIG modular package (2026-02 refactored)
│   │   ├── __init__.py       # Public API (18 exports)
│   │   ├── types.py          # ProcessingMode, SpikeDetectionMode, HopResult, GeDIGResult
│   │   ├── config.py         # GeDIGConfig (from_env, from_kwargs, preset)
│   │   ├── spike.py          # detect_spike, compute_rewards
│   │   ├── graph_utils.py    # Graph utility functions (11 functions)
│   │   ├── monitor.py        # GeDIGMonitor
│   │   ├── logger.py         # GeDIGLogger (rotating CSV)
│   │   ├── selector.py       # TwoThresholdCandidateSelector, compute_gedig
│   │   ├── linkset.py        # compute_linkset_metrics
│   │   ├── multihop.py       # calculate_multihop
│   │   └── ab_writer_helper.py
│   ├── ig/                   # Information Gain package
│   │   ├── __init__.py
│   │   ├── types.py          # EntropyMethod, IGResult
│   │   └── methods.py        # ImprovedEntropyMethods
│   ├── gedig_core.py         # GeDIGCore orchestration (779 lines)
│   ├── graph_edit_distance.py # GED calculation
│   ├── information_gain.py    # IG calculation (607 lines)
│   ├── similarity_entropy.py  # Entropy metrics
│   └── __init__.py
│
├── config/                    # Pydantic configuration system
│   ├── models.py             # Config models (InsightSpikeConfig)
│   ├── loader.py             # Configuration loading
│   ├── presets.py            # Built-in presets
│   ├── converter.py          # Legacy conversion
│   ├── constants.py          # Default values
│   └── legacy_config.py      # Legacy support (deprecated)
│
├── cli/                       # Command-line interfaces
│   ├── spike.py              # Main CLI (improved)
│   ├── commands/             # CLI command modules
│   │   ├── discover.py       # Discover insights
│   │   ├── bridge.py         # Bridge concepts
│   │   ├── graph.py          # Graph visualization
│   │   └── __init__.py
│   └── __init__.py
│
├── public/                    # Public API surface (stable)
│   ├── __init__.py            # Re-exports and helpers
│   └── wrapper.py             # InsightAppWrapper for local apps
│
├── metrics/                   # Metric calculations
│   ├── graph_metrics.py      # Graph analysis metrics
│   └── __init__.py
│
└── utils/                     # Utility functions
    ├── file_utils.py         # File operations
    ├── embedding_utils.py    # Embedding helpers
    └── __init__.py
```

## 🧩 Top-Level Apps

```
apps/
└── knowledge_app.py           # Streamlit UI for the local knowledge app
```

## 🏗️ Architecture Principles

### 1. **Core Package**
- Contains only abstract interfaces and data structures
- No concrete implementations
- Defines contracts that implementations must follow
- Minimal dependencies

### 2. **Implementations Package**
- All concrete implementations of core interfaces
- Organized by type (agents, layers, memory)
- Can have external dependencies
- Easily swappable implementations

### 3. **Features Package**
- Self-contained feature modules
- Can be enabled/disabled independently
- Clear boundaries and interfaces
- Examples: query transformation, graph reasoning

### 4. **Tools Package**
- Standalone utilities that can work independently
- Experimental frameworks
- Research tools
- Can be extracted to separate projects

## 🔄 Import Examples

### Current structure (July 2025):
```python
# Interfaces (from core)
from insightspike.core.interfaces.layer_interfaces import L2MemoryInterface
from insightspike.core.interfaces.generic_interfaces import EnvironmentState
from insightspike.core.agents.generic_agent import GenericReasoner

# Implementations
from insightspike.implementations.agents.main_agent import MainAgent
from insightspike.implementations.layers.layer2_memory_manager import L2MemoryManager
from insightspike.implementations.layers.layer3_graph_reasoner import L3GraphReasoner

# Features
from insightspike.features.graph_reasoning.reward_calculator import RewardCalculator
from insightspike.features.graph_reasoning.graph_analyzer import GraphAnalyzer

# Algorithms
from insightspike.algorithms.graph_edit_distance import calculate_graph_edit_distance
from insightspike.algorithms.information_gain import calculate_information_gain

# geDIG (refactored 2026-02)
from insightspike.algorithms.gedig import GeDIGConfig, GeDIGResult
from insightspike.algorithms.gedig import detect_spike, compute_rewards
from insightspike.algorithms.gedig import calculate_multihop, compute_linkset_metrics
from insightspike.algorithms.gedig_core import GeDIGCore

# Using GeDIGConfig
config = GeDIGConfig.from_env()  # From environment variables
config = GeDIGConfig.from_kwargs(lambda_weight=0.7)  # With overrides
config = GeDIGConfig.preset("maze")  # From presets
core = GeDIGCore(config=config)

# Configuration
from insightspike.config import load_config, InsightSpikeConfig
from insightspike.config.presets import ConfigPresets
```

## 🎯 Key Changes (February 2026)

### geDIG Modular Refactoring:
- ✅ `gedig_core.py` reduced from 2,159 → 779 lines (-64%)
- ✅ Extracted 10 specialized modules in `algorithms/gedig/`
- ✅ Centralized configuration in `GeDIGConfig` (from_env, from_kwargs, preset)
- ✅ Test coverage improved from 54% → 84%
- ✅ Added 174 new tests (53 → 227 total)
- ✅ Created `algorithms/ig/` package for Information Gain

### New Modules:
| Module | Lines | Responsibility |
|--------|-------|----------------|
| types.py | 128 | ProcessingMode, SpikeDetectionMode, HopResult, GeDIGResult, LinksetMetrics |
| config.py | 310 | GeDIGConfig with from_env, from_kwargs, preset methods |
| spike.py | 114 | detect_spike, compute_rewards |
| graph_utils.py | 416 | 11 graph utility functions |
| linkset.py | 218 | compute_linkset_metrics |
| multihop.py | 370 | calculate_multihop |
| monitor.py | 193 | GeDIGMonitor |
| logger.py | 137 | GeDIGLogger (rotating CSV) |
| selector.py | 270 | TwoThresholdCandidateSelector, compute_gedig |

---

## 🎯 Key Changes (July 2025)

### Removed/Deprecated:
- ❌ `config.reasoning` → Use `config.graph` instead
- ❌ Legacy methods: `_detect_spike`, `save_graph`, `load_graph`
- ❌ 13 experimental CLI commands
- ❌ `tools/` package (moved to experiments)
- ❌ Multiple duplicate layer implementations

### Added/Improved:
- ✅ Unified layer implementations
- ✅ Memory aging system (time-based decay)
- ✅ Intelligent episode merging (cosine similarity)
- ✅ Pydantic configuration with presets
- ✅ Clean separation of interfaces and implementations
- ✅ Test coverage improved (17% → 23%)

## 🎯 Benefits

1. **Clear Separation**: Interfaces in `core/`, implementations in `implementations/`
2. **Better Testing**: Easy to mock interfaces
3. **Modularity**: Features are self-contained
4. **Type Safety**: Pydantic models for configuration
5. **Extensibility**: Easy to add new implementations
6. **Clean Architecture**: Following SOLID principles

## 📦 Package Dependencies

```mermaid
graph TD
    CLI --> Implementations
    Implementations --> Core
    Features --> Core
    Tools --> Core
    Config --> Core
    
    style Core fill:#f9f,stroke:#333,stroke-width:4px
    style Implementations fill:#bbf,stroke:#333,stroke-width:2px
    style Features fill:#bfb,stroke:#333,stroke-width:2px
    style Tools fill:#fbf,stroke:#333,stroke-width:2px
```

## 🚀 Migration Guide

If you're updating from the old structure:

1. Update imports to use new paths
2. Core interfaces remain the same
3. All functionality is preserved
4. Backward compatibility maintained through `__init__.py`

See the migration documentation for detailed steps.
