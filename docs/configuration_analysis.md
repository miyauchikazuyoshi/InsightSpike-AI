# InsightSpike Configuration Analysis

## Overview

InsightSpike has a complex configuration system with multiple layers and some inconsistencies:

1. **Root config.yaml** - Main configuration file with comprehensive settings
2. **core/config.py** - Primary Python dataclass configuration structure
3. **core/config_manager.py** - Alternative configuration system with different structure
4. **config/simple_config.py** - Simplified configuration system with presets

## Configuration Flow

### 1. Root config.yaml Structure

The root `config.yaml` file contains these sections:
- `core`: Model settings (embedding model, LLM provider, device settings)
- `memory`: Memory system capacity settings
- `retrieval`: Top-K and similarity threshold settings
- `graph`: Spike detection thresholds and algorithm selection
- `reasoning`: Episode management and integration thresholds
- `spike`: Eureka spike detection thresholds
- `unknown_learner`: Unknown information learning settings
- `paths`: File paths for data, logs, index, graph
- `environment`: Execution environment (local, colab, ci)
- `processing`: Batch size, workers, timeout
- `output`: Output format and verbosity settings

### 2. Python Configuration Classes

#### core/config.py (Primary System)

Defines these dataclasses:
- `EmbeddingConfig`: Embedding model settings
- `LLMConfig`: Language model configuration
- `RetrievalConfig`: Retrieval layer settings
- `SpikeConfig`: Spike detection thresholds
- `GraphConfig`: Graph processing settings
- `ReasoningConfig`: Reasoning layer configuration
- `MemoryConfig`: Memory management settings
- `PathConfig`: File paths
- `ScalableGraphConfig`: Scalable graph operations
- `UnknownLearnerConfig`: Unknown learner settings
- `Config`: Main configuration object containing all above

#### core/config_manager.py (Alternative System)

Defines a different structure:
- `GNNConfig`: Graph Neural Network settings
- `MemoryConfig`: Different memory configuration (vector_dim, max_items)
- `InsightDetectionConfig`: Insight detection weights and thresholds
- `AlgorithmConfig`: GED and IG algorithm settings
- `ExperimentalConfig`: Random seed, batch size, learning rate
- `InsightSpikeConfig`: Contains all above configurations

#### config/simple_config.py (Simplified System)

Provides:
- `SimpleConfig`: Flat configuration structure with sensible defaults
- `ConfigPresets`: Pre-defined configurations (development, testing, production, etc.)
- `ConfigManager`: Manages configuration with environment variable support
- Conversion method `to_legacy_config()` to convert to core/config.py format

## Issues and Mismatches

### 1. Multiple Configuration Systems

**Problem**: Three different configuration systems exist simultaneously:
- core/config.py (used by most components)
- core/config_manager.py (seems unused in main codebase)
- config/simple_config.py (provides simplified interface)

### 2. YAML to Python Mapping

**Problem**: The root `config.yaml` structure doesn't directly map to any Python configuration class:

- YAML has `core` section with mixed settings
- Python has separate `EmbeddingConfig` and `LLMConfig`
- No automatic YAML loading mechanism in the main configuration flow

### 3. Unused Configuration Loading

**Problem**: Despite having YAML support in `config_manager.py`, the main codebase doesn't use it:
- `load_from_file()` method exists but isn't called in main flow
- Environment variable `INSIGHTSPIKE_CONFIG_PATH` is checked but not used
- Most code uses programmatic configuration or presets

### 4. Duplicate Definitions

**Problem**: Same settings defined in multiple places:
- `MemoryConfig` exists in both core/config.py and core/config_manager.py with different fields
- GED/IG algorithm selection in both `GraphConfig` and `ReasoningConfig`
- Spike thresholds duplicated across `SpikeConfig`, `GraphConfig`, and `ReasoningConfig`

### 5. Configuration Flow

Actual configuration flow:
1. CLI uses `get_config()` with presets from simple_config.py
2. `ConfigManager` converts `SimpleConfig` to legacy `Config` format
3. Main components use the legacy `Config` object
4. Root `config.yaml` is not loaded unless explicitly specified

## YAML to Python Mapping Analysis

### Current Mapping (if YAML were loaded)

If the root `config.yaml` were loaded, here's how it would need to map to `core/config.py`:

```yaml
core:                          # No direct mapping - mixed content
  model_name: "..."            # -> EmbeddingConfig.model_name
  llm_provider: "..."          # -> LLMConfig.provider
  llm_model: "..."             # -> LLMConfig.model_name
  max_tokens: 256              # -> LLMConfig.max_tokens
  temperature: 0.3             # -> LLMConfig.temperature
  device: "cpu"                # -> EmbeddingConfig.device, LLMConfig.device
  use_gpu: false               # -> LLMConfig.use_gpu
  safe_mode: false             # -> LLMConfig.safe_mode

memory:                        # -> MemoryConfig
  max_retrieved_docs: 15       # Direct mapping
  short_term_capacity: 10      # No mapping in MemoryConfig
  working_memory_capacity: 20  # No mapping in MemoryConfig
  episodic_memory_capacity: 60 # No mapping in MemoryConfig
  pattern_cache_capacity: 15   # No mapping in MemoryConfig

retrieval:                     # -> RetrievalConfig
  similarity_threshold: 0.35   # Direct mapping
  top_k: 15                    # Direct mapping
  layer1_top_k: 20            # Direct mapping
  layer2_top_k: 15            # Direct mapping
  layer3_top_k: 12            # Direct mapping

graph:                         # -> GraphConfig
  spike_ged_threshold: 0.5     # Direct mapping
  spike_ig_threshold: 0.2      # Direct mapping
  use_gnn: false              # No mapping in GraphConfig (in ReasoningConfig)
  gnn_hidden_dim: 64          # No mapping in GraphConfig (in ReasoningConfig)
  ged_algorithm: "hybrid"      # Direct mapping
  ig_algorithm: "hybrid"       # Direct mapping
  hybrid_weights:              # No mapping in Python configs
    structure: 0.4
    semantic: 0.4
    quality: 0.2

reasoning:                     # -> ReasoningConfig
  # Most fields map directly
  
spike:                         # -> SpikeConfig
  spike_ged: 0.5              # Direct mapping
  spike_ig: 0.2               # Direct mapping
  eta_spike: 0.2              # Direct mapping

unknown_learner:              # -> UnknownLearnerConfig
  # All fields map directly

paths:                        # -> PathConfig
  data_dir: "data/raw"        # Would need conversion to Path objects
  log_dir: "data/logs"        
  index_file: "data/index.faiss"
  graph_file: "data/graph_pyg.pt"

environment: "local"          # -> Config.environment

processing:                   # No mapping in core config
  batch_size: 32
  max_workers: 4
  timeout_seconds: 300

output:                       # No mapping in core config
  default_format: "text"
  save_results: true
  generate_visualizations: false
  verbose: false
```

### Missing Mappings

1. **From YAML to Python**:
   - `processing` section has no target
   - `output` section has no target
   - `memory` capacity fields (short_term, working, episodic, pattern_cache) have no target
   - `graph.hybrid_weights` has no target

2. **From Python to YAML**:
   - `ScalableGraphConfig` - entire config class not in YAML
   - `LLMConfig.use_direct_generation` and related fields
   - `LLMConfig.use_layer4_pipeline` and related fields
   - Many `MemoryConfig` fields (merge_ged, split_ig, prune_c, etc.)
   - `PathConfig.root_dir`

## Recommendations

1. **Consolidate Configuration Systems**: Choose one primary configuration system and remove others
2. **Implement YAML Loading**: Add proper YAML loading in the main configuration flow
3. **Fix Structure Mapping**: Align YAML structure with Python dataclass structure
4. **Remove Duplicates**: Consolidate duplicate configuration definitions
5. **Document Configuration**: Clearly document which configuration system to use and how
6. **Create Config Loader**: Implement a proper YAML to Config object mapper that handles the structural differences

## Current State Summary

### What Works

1. **SimpleConfig System**: The simplified configuration in `config/simple_config.py` is actively used
2. **Presets**: ConfigPresets provide easy configuration for different scenarios
3. **Legacy Conversion**: `to_legacy_config()` successfully converts SimpleConfig to core Config format
4. **CLI Integration**: The CLI properly uses SimpleConfig with presets

### What Doesn't Work

1. **YAML Loading**: Despite documentation suggesting `export INSIGHTSPIKE_CONFIG_PATH="./config.yaml"`, this is not implemented
2. **ConfigManager in core**: The `core/config_manager.py` with its YAML loading capability is unused
3. **Direct YAML Usage**: The root `config.yaml` file is essentially documentation - not actively loaded

### Actual Configuration Flow

```
1. CLI/Code calls get_config(preset="development")
2. SimpleConfig object created with preset values
3. ConfigManager wraps SimpleConfig
4. to_legacy_config() converts to core.config.Config format
5. Components use the Config object
```

The root `config.yaml` file serves more as a template/documentation than an active configuration source.
