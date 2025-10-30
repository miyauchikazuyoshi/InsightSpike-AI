# InsightSpike Configuration System

## 📋 Overview

InsightSpike uses a modern, Pydantic-based configuration system that provides:
- Type safety and validation
- YAML/JSON file support
- Environment variable overrides
- Configuration presets
- Backward compatibility

## 🏗️ Architecture

### Core Components

1. **`config/models.py`** - Pydantic models defining configuration structure
2. **`config/loader.py`** - Configuration loading and management
3. **`config/presets.py`** - Pre-defined configuration sets
4. **`config/converter.py`** - Legacy format conversion

## 📝 Configuration Structure

```python
from insightspike.config.models import InsightSpikeConfig

# Main configuration model
InsightSpikeConfig:
  ├── core: CoreConfig
  │   ├── model_name: str
  │   ├── llm_provider: Literal["local", "openai", "anthropic", "mock", "clean"]
  │   ├── llm_model: str
  │   ├── max_tokens: int
  │   ├── temperature: float
  │   └── device: str
  ├── memory: MemoryConfig
  │   ├── max_retrieved_docs: int
  │   ├── episodic_memory_capacity: int
  │   └── ...
  ├── retrieval: RetrievalConfig
  │   ├── similarity_threshold: float
  │   ├── top_k: int
  │   └── ...
  ├── graph: GraphConfig
  │   ├── spike_ged_threshold: float
  │   ├── spike_ig_threshold: float
  │   └── ...
  ├── gedig: GeDIGConfig  # NEW (2025-08)
  │   ├── use_refactored_gedig: bool
  │   ├── use_refactored_reward: bool
  │   ├── lambda_weight: float
  │   ├── mu: float
  │   ├── warmup_steps: int
  │   ├── enable_multihop: bool
  │   ├── max_hops: int
  │   ├── decay_factor: float
  │   ├── enable_spectral: bool
  │   ├── spectral_weight: float
  │   ├── spike_threshold: float  # (legacy fallback until modes added)
  │   ├── log_path: str
  │   ├── max_log_lines: int
  │   ├── max_log_bytes: int
  │   └── ... (future: spike_detection_mode, tau_s, tau_i)
  └── ...
```

## 🚀 Usage Examples

### Loading Configuration

```python
from insightspike.config import load_config, get_config
from insightspike.config.loader import ConfigLoader

# Method 1: Auto-load from default locations
config = load_config()

# Method 2: Load from specific file
config = load_config(config_path="./my_config.yaml")

# Method 3: Load with preset
config = load_config(preset="production")

# Method 4: Direct loader usage
loader = ConfigLoader()
config = loader.load(config_path="./config.yaml")
```

### Using Presets

```python
from insightspike.config.presets import ConfigPresets

# Get available presets
presets = ["development", "experiment", "production", "testing", "cloud"]

# Load preset configuration
dev_config = ConfigPresets.get_preset("development")
prod_config = ConfigPresets.get_preset("production")
```

### Environment Variables

Override any configuration value using environment variables:

```bash
# Override LLM provider
export INSIGHTSPIKE_CORE_LLM_PROVIDER=openai

# Override memory capacity
export INSIGHTSPIKE_MEMORY_EPISODIC_MEMORY_CAPACITY=100

# Override similarity threshold
export INSIGHTSPIKE_RETRIEVAL_SIMILARITY_THRESHOLD=0.8
```

### Metrics Defaults (2025-10)

標準設定では、グラフメトリクスは GeDIGCore（advanced）を用い、クエリ中心の局所サブグラフ評価を有効化しています。

```yaml
graph:
  ged_algorithm: advanced
  ig_algorithm: advanced

metrics:
  query_centric: true
  query_topk_centers: 3
  query_radius: 1
```

軽量に戻したい場合は `pyg` を指定し、局所評価を無効化してください。

```yaml
graph:
  ged_algorithm: pyg
  ig_algorithm: pyg
metrics:
  query_centric: false
```

### Creating Custom Configuration

```python
from insightspike.config.models import (
    InsightSpikeConfig, CoreConfig, MemoryConfig
)

# Create custom config
config = InsightSpikeConfig(
    core=CoreConfig(
        llm_provider="clean",
        temperature=0.5
    ),
    memory=MemoryConfig(
        episodic_memory_capacity=200
    )
)

# Save to file
config.save("custom_config.yaml")
```

## 🔄 Legacy Compatibility

For backward compatibility with old code:

```python
from insightspike.config.converter import ConfigConverter

# Convert new config to legacy format
legacy_config = ConfigConverter.insight_config_to_legacy(new_config)

# Or from preset
preset_dict = ConfigPresets.get_preset("development")
legacy_config = ConfigConverter.preset_dict_to_legacy_config(preset_dict)
```

## 📁 Configuration Files

### Default Locations

The configuration loader searches in this order:
1. `INSIGHTSPIKE_CONFIG_PATH` environment variable
2. `./config.yaml` (current directory)
3. `./config.json`
4. `~/.insightspike/config.yaml` (user home)
5. Built-in presets

### Example `config.yaml`

```yaml
# Core Language Model Settings
core:
  model_name: "paraphrase-MiniLM-L6-v2"
  llm_provider: "clean"
  llm_model: "clean"
  max_tokens: 256
  temperature: 0.3
  device: "cpu"

# Memory Configuration
memory:
  max_retrieved_docs: 15
  episodic_memory_capacity: 60

# Retrieval Settings
retrieval:
  similarity_threshold: 0.35
  top_k: 15

# Graph Processing
graph:
  spike_ged_threshold: 0.5
  spike_ig_threshold: 0.2

# GeDIG Refactored Metrics (NEW)
gedig:
  use_refactored_gedig: true
  use_refactored_reward: true
  lambda_weight: 0.5
  mu: 0.5
  warmup_steps: 10
  enable_multihop: false
  max_hops: 3
  decay_factor: 0.7
  enable_spectral: false
  spectral_weight: 0.3
  spike_threshold: -0.5  # Temporary until SpikeDetectionMode (Phase C)
  log_path: logs/gedig/gedig_metrics.csv
  max_log_lines: 50000
  max_log_bytes: 52428800
  max_log_rotate: 32  # optional future

```

## 🧪 Validation

All configuration values are validated using Pydantic:

```python
from pydantic import ValidationError

try:
    config = InsightSpikeConfig(
        core={"temperature": 3.0}  # Too high!
    )
except ValidationError as e:
    print(e)
    # temperature: ensure this value is less than or equal to 2.0
```

## 🎯 Best Practices

1. **Use Presets**: Start with a preset and override only what you need
2. **Environment Variables**: Use for deployment-specific settings
3. **Version Control**: Keep `config.yaml` in version control with safe defaults
4. **Validation**: Let Pydantic validate your configuration
5. **Type Safety**: Use the config models for type hints

## 🔮 Future Plans

- [ ] Configuration schema documentation
- [ ] Web-based configuration editor
- [ ] Configuration migration tools
- [ ] Remote configuration support
- [ ] Configuration profiles

## 📚 Related Documentation

- [Migration Guide](/docs/development/config_migration_plan.md)
- [Directory Structure](/docs/architecture/directory_structure.md)
- [MainAgent Behavior](/docs/architecture/mainagent_behavior.md)
