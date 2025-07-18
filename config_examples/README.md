# InsightSpike Configuration Examples

This directory contains example configuration files for InsightSpike.

## Available Examples

### ged_ig_algorithms.yaml
Configuration examples for different GED (Graph Edit Distance) and IG (Information Gain) algorithm combinations:
- **simple_config**: Fastest algorithms, suitable for development
- **advanced_config**: Balanced performance and accuracy
- **networkx_config**: Most accurate but slowest
- **mixed_config**: Custom combinations
- **full_example**: Complete configuration with all settings

## Usage

### Method 1: Direct Loading
```python
from insightspike.config import load_config

# Load a specific example configuration
config = load_config(config_path="config_examples/ged_ig_algorithms.yaml")
```

### Method 2: Copy to Project Root
```bash
# Copy to project root as config.yaml
cp config_examples/ged_ig_algorithms.yaml config.yaml

# The config will be automatically loaded
```

### Method 3: Environment Variable
```bash
export INSIGHTSPIKE_CONFIG_PATH="config_examples/ged_ig_algorithms.yaml"
```

### Method 4: Merge with Preset
```python
from insightspike.config import load_config, ConfigLoader
from insightspike.config.presets import ConfigPresets

# Start with a preset
config = load_config(preset="experiment")

# Load additional settings from example
loader = ConfigLoader()
example_config = loader._load_yaml("config_examples/ged_ig_algorithms.yaml")

# Manually merge specific settings
config.graph.ged_algorithm = example_config["networkx_config"]["graph"]["ged_algorithm"]
config.graph.ig_algorithm = example_config["networkx_config"]["graph"]["ig_algorithm"]
```

## Configuration Format

All configuration files follow the new Pydantic-based format:

```yaml
# Section structure matches InsightSpikeConfig model
core:
  model_name: "..."
  llm_provider: "..."
  
memory:
  max_retrieved_docs: 15
  
graph:
  ged_algorithm: "..."
  spike_ged_threshold: 0.5
  
# etc...
```

## Environment Variable Overrides

You can override any setting using environment variables:

```bash
# Pattern: INSIGHTSPIKE_<SECTION>_<KEY>=<value>
export INSIGHTSPIKE_GRAPH_GED_ALGORITHM=networkx
export INSIGHTSPIKE_GRAPH_SPIKE_GED_THRESHOLD=0.8
export INSIGHTSPIKE_CORE_LLM_PROVIDER=openai
```

## See Also

- [Configuration Documentation](/docs/architecture/configuration.md)
- [Config Migration Guide](/docs/development/config_migration_plan.md)