# Configuration Hierarchy and Management

## Overview

InsightSpike uses a hierarchical configuration system that allows flexible control through multiple sources. Understanding this hierarchy is crucial for effective configuration management.

## Configuration Priority (Highest to Lowest)

1. **Command-line arguments** - Override everything
2. **Environment variables** - `INSIGHTSPIKE_*` prefix
3. **Config file** - `config.yaml`, `config.json`, or specified file
4. **Preset** - Named configuration sets
5. **Defaults** - Built-in fallback values

## File Locations

The system searches for configuration files in this order:

1. Explicitly specified path (`--config path/to/config.yaml`)
2. Environment variable (`INSIGHTSPIKE_CONFIG_PATH`)
3. `config.yaml` (current directory)
4. `.insightspike.yaml` (hidden config)
5. `config.json` (legacy format)

## One Central Config File

The recommended approach is to use **one central `config.yaml`** file in the project root:

```yaml
# config.yaml - Main configuration
environment: production

memory:
  enable_graph_search: false  # Set to true for experiments
  max_retrieved_docs: 10

processing:
  enable_insight_registration: true
  enable_layer1_bypass: false

llm:
  provider: mock  # Change to anthropic/claude for production
```

## Using Alternative Configurations

For experiments or special cases, you can:

### 1. Modify config.yaml directly
```yaml
# Enable graph search by uncommenting
memory:
  enable_graph_search: true      # Was commented out
  graph_hop_limit: 2
  graph_neighbor_threshold: 0.4
  graph_path_decay: 0.7
```

### 2. Use command-line overrides
```bash
# Temporarily enable graph search
spike --memory.enable_graph_search true query "test"

# Multiple overrides
spike --memory.enable_graph_search true \
      --memory.graph_hop_limit 3 \
      --processing.enable_layer1_bypass true \
      query "complex question"
```

### 3. Use environment variables
```bash
export INSIGHTSPIKE_MEMORY__ENABLE_GRAPH_SEARCH=true
export INSIGHTSPIKE_MEMORY__GRAPH_HOP_LIMIT=3
spike query "test"
```

## Environment Variables

Any configuration can be overridden via environment:

```bash
# Format: INSIGHTSPIKE_<SECTION>_<KEY>
export INSIGHTSPIKE_MEMORY_ENABLE_GRAPH_SEARCH=true
export INSIGHTSPIKE_LLM_PROVIDER=anthropic/claude
export INSIGHTSPIKE_PROCESSING_ENABLE_LEARNING=true
```

## Command-Line Overrides

Highest priority, useful for testing:

```bash
# Override specific settings
spike --memory.enable_graph_search true \
      --memory.graph_hop_limit 3 \
      --processing.enable_learning true \
      query "complex question"
```

## Presets vs Config File

Presets are pre-defined configurations in code:
- Good for: Standard use cases, quick testing
- Example: `--preset graph_enhanced`

Config files are external YAML/JSON:
- Good for: Custom configurations, production tuning
- Example: `--config my_custom_config.yaml`

## Best Practices

1. **Development**: Use `config.yaml` with safe defaults
2. **Experiments**: Create specific config files or use presets
3. **Production**: Use environment variables for secrets, config file for settings
4. **Testing**: Use command-line overrides for quick changes

## Example Workflow

```bash
# 1. Default development (uses config.yaml)
spike query "test"

# 2. Test with graph search
spike --memory.enable_graph_search true query "test"

# 3. Production with custom config
export ANTHROPIC_API_KEY="..."
spike --config config_production.yaml query "test"

# 4. Quick experiment with preset
spike --preset adaptive_learning query "test"
```

## Debugging Configuration

View the active configuration:
```bash
spike config show  # Shows current configuration
spike config validate  # Validates configuration
```

## Migration from Old Config

If you have old configuration files:

1. The system auto-converts legacy format
2. But it's recommended to migrate to new format
3. Use `spike config migrate old_config.json` (if available)

## Summary

- **One `config.yaml`** in project root for most users
- Use **presets** for standard scenarios
- Create **separate config files** for experiments
- Use **environment variables** for secrets
- Use **command-line** for quick tests

This approach keeps configuration simple while allowing flexibility when needed.