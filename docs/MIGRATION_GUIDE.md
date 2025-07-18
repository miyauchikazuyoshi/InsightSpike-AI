# Configuration Migration Guide

## Overview

InsightSpike is migrating from the legacy configuration system to a new Pydantic-based configuration system that fully supports YAML files and environment variables.

## Migration Steps

### 1. Update Imports

**Old way:**
```python
from insightspike.core.config import Config, get_config
from insightspike.config import SimpleConfig, ConfigManager
```

**New way:**
```python
from insightspike.config import InsightSpikeConfig, load_config, get_config
```

### 2. Loading Configuration

**Old way:**
```python
# Legacy Config
config = get_config()  # Returns legacy Config object

# SimpleConfig
simple_config = SimpleConfig.from_preset("development")
manager = ConfigManager(simple_config)
legacy_config = manager.to_legacy_config()
```

**New way:**
```python
# Load from YAML file
config = load_config("config.yaml")

# Load with preset
config = load_config(preset="development")

# Load with environment variables
# Set: export INSIGHTSPIKE_LLM_MODEL="gpt-4"
config = load_config()

# Get current config (singleton)
config = get_config()
```

### 3. Accessing Configuration Values

**Old way:**
```python
# Legacy Config
model_name = config.embedding.model_name
llm_provider = config.llm.provider
top_k = config.retrieval.top_k

# SimpleConfig
model_name = simple_config.embedding_model
llm_provider = simple_config.llm_provider
top_k = simple_config.retrieval_top_k
```

**New way:**
```python
# Unified structure matching config.yaml
model_name = config.core.model_name
llm_provider = config.core.llm_provider
top_k = config.retrieval.top_k

# Direct access to sub-configs
memory_config = config.memory
graph_config = config.graph
```

### 4. Creating Custom Configuration

**Old way:**
```python
from insightspike.core.config import Config, EmbeddingConfig, LLMConfig

config = Config(
    embedding=EmbeddingConfig(model_name="custom-model"),
    llm=LLMConfig(provider="openai", model_name="gpt-4")
)
```

**New way:**
```python
from insightspike.config import InsightSpikeConfig

# Create with overrides
config = load_config(overrides={
    "core": {
        "model_name": "custom-model",
        "llm_provider": "openai",
        "llm_model": "gpt-4"
    }
})

# Or create directly
from insightspike.config.models import InsightSpikeConfig, CoreConfig

config = InsightSpikeConfig(
    core=CoreConfig(
        model_name="custom-model",
        llm_provider="openai",
        llm_model="gpt-4"
    )
)
```

### 5. Using config.yaml

The new system fully supports the `config.yaml` file in your project root:

```yaml
# config.yaml
core:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  llm_provider: "local"
  llm_model: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  device: "cuda"
  use_gpu: true

retrieval:
  top_k: 20
  similarity_threshold: 0.4
```

Load it with:
```python
config = load_config("config.yaml")
```

### 6. Environment Variables

All configuration values can be overridden with environment variables:

```bash
export INSIGHTSPIKE_MODEL_NAME="custom-model"
export INSIGHTSPIKE_LLM_PROVIDER="openai"
export INSIGHTSPIKE_USE_GPU=true
export INSIGHTSPIKE_TOP_K=25
```

### 7. Testing

**Old way:**
```python
from insightspike.core.config import Config
config = Config(safe_mode=True)
```

**New way:**
```python
from insightspike.config import load_config
config = load_config(preset="test")  # Uses mock LLM, minimal settings
```

## Backward Compatibility

During the migration period, you can still use the legacy format:

```python
# Get new config but convert to legacy format
from insightspike.config import get_config
new_config = get_config()
legacy_config = new_config.to_legacy_config()
```

## Common Patterns

### For CLI Applications
```python
from insightspike.config import load_config, ConfigPresets

# List available presets
presets = ConfigPresets.list_presets()

# Load with user-specified preset
config = load_config(preset=args.preset)
```

### For Tests
```python
from insightspike.config import load_config

@pytest.fixture
def config():
    return load_config(preset="test")
```

### For Experiments
```python
from insightspike.config import load_config

# Load experiment config with custom overrides
config = load_config(
    preset="experiment",
    overrides={
        "graph": {"use_gnn": True},
        "processing": {"batch_size": 64}
    }
)
```

## Deprecation Timeline

1. **Now**: Both systems work, deprecation warnings shown
2. **v2.0**: Legacy imports will require explicit flag
3. **v3.0**: Legacy support will be removed

## Need Help?

- Check the [config package documentation](./architecture/configuration.md)
- See [example configurations](../experiments/templates/)
- Report issues on GitHub