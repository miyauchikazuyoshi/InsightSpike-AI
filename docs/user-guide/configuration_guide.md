# InsightSpike Configuration Guide

## Overview

InsightSpike now provides a simplified configuration system that makes it easy to:
- Switch between different execution modes (CPU/GPU, safe mode/real LLM)
- Use pre-configured settings for common scenarios
- Override settings via environment variables
- Save and load configurations

## Quick Start

### Using Presets

```python
from insightspike.config import get_config, ConfigPresets

# Get a preset configuration
config = get_config("experiment")  # Real LLM, moderate performance

# Or use preset class directly
dev_config = ConfigPresets.development()  # Fast, safe, debug-friendly
prod_config = ConfigPresets.production()  # Optimized for performance
```

### Available Presets

1. **development** - Fast iteration, mock LLM, debug enabled
2. **testing** - Isolated paths, reproducible, small limits
3. **production** - GPU enabled, real LLM, optimized
4. **experiment** - Real LLM, CPU mode, moderate settings
5. **cloud** - For API-based LLMs (OpenAI, Anthropic)

## Configuration Options

### Core Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `mode` | `"cpu"` | Execution mode: "cpu", "gpu", "mps" |
| `safe_mode` | `True` | Use mock LLM (no model loading) |
| `debug` | `False` | Enable debug logging |

### Model Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `embedding_model` | `"paraphrase-MiniLM-L6-v2"` | Sentence embedding model |
| `llm_model` | `"TinyLlama/TinyLlama-1.1B-Chat-v1.0"` | Language model |
| `llm_provider` | `"local"` | Provider: "local", "openai", "anthropic" |

### Performance Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `max_tokens` | `256` | Maximum tokens for LLM response |
| `temperature` | `0.3` | LLM temperature (0.0-1.0) |
| `batch_size` | `32` | Batch size for processing |

### Spike Detection

| Setting | Default | Description |
|---------|---------|-------------|
| `spike_ged_threshold` | `0.5` | Graph Edit Distance threshold |
| `spike_ig_threshold` | `0.2` | Information Gain threshold |
| `spike_sensitivity` | `1.0` | Multiplier for thresholds |

### Datastore Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `datastore.type` | `"filesystem"` | Persistence backend: `filesystem`, `in_memory` |
| `datastore.root_path` | `"./data/insight_store"` | Base directory for filesystem backend (alias: `base_path`) |

Example YAML:

```yaml
datastore:
  type: filesystem
  root_path: ./data/insight_store  # or absolute path
```

Tip: CLI bootstrap honors `INSIGHTSPIKE_DATA_DIR` to override the base path quickly.

## Usage Examples

### Custom Configuration

```python
from insightspike.config import SimpleConfig

config = SimpleConfig(
    mode="cpu",
    safe_mode=False,  # Use real LLM
    max_tokens=512,
    spike_sensitivity=1.5,  # More sensitive spike detection
    debug=True
)
```

### Using ConfigManager

```python
from insightspike.config import ConfigManager, ConfigPresets

# Create manager with preset
manager = ConfigManager(ConfigPresets.experiment())

# Get values
mode = manager.get("mode")
tokens = manager.get("max_tokens", default=256)

# Update values
manager.set("debug", True)
manager.update(
    max_tokens=1024,
    temperature=0.7
)
```

### Environment Variables

Override any setting using environment variables:

```bash
export INSIGHTSPIKE_MODE=gpu
export INSIGHTSPIKE_SAFE_MODE=false
export INSIGHTSPIKE_MAX_TOKENS=1024
export INSIGHTSPIKE_DEBUG=true
```

### Save and Load

```python
# Save configuration
config = ConfigPresets.experiment()
config.save("my_experiment.json")

# Load configuration
loaded = SimpleConfig.load("my_experiment.json")
```

## Integration with MainAgent

```python
from insightspike.config import ConfigManager, ConfigPresets
from insightspike.core.agents.main_agent import MainAgent

# Setup configuration
manager = ConfigManager(ConfigPresets.experiment())

# Convert to legacy format (temporary compatibility)
legacy_config = manager.to_legacy_config()

# Create agent
agent = MainAgent(config=legacy_config)
agent.initialize()
```

## Best Practices

1. **Start with presets** - Use presets as a starting point and customize as needed
2. **Use environment variables** - For deployment, use env vars instead of hardcoding
3. **Save configurations** - Save successful configurations for reproducibility
4. **Spike sensitivity** - Adjust `spike_sensitivity` instead of individual thresholds

## Migration from Legacy Config

The new system is designed to work alongside the existing configuration. The `to_legacy_config()` method provides compatibility:

```python
# Old way
from insightspike.core.config import Config
config = Config()
config.llm.safe_mode = False
config.spike.spike_ged = 0.3

# New way
from insightspike.config import SimpleConfig
config = SimpleConfig(safe_mode=False, spike_ged_threshold=0.3)
```

## Troubleshooting

### GPU Not Available

If you request GPU mode but it's not available, the system automatically falls back to CPU:

```
WARNING: GPU requested but not available, falling back to CPU
```

### Model Not Found

If using `safe_mode=False` but the model isn't downloaded:

```python
# Check if model is available
manager = ConfigManager()
if not manager.validate():
    print("Configuration issues detected")
```

### Environment Variable Not Working

Ensure the variable follows the naming pattern:
- Prefix: `INSIGHTSPIKE_`
- Setting name in UPPERCASE
- Example: `INSIGHTSPIKE_MAX_TOKENS`
