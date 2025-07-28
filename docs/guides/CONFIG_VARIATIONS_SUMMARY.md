# InsightSpike Configuration Variations Summary

## Quick Reference

### 🚀 Basic Configurations

```yaml
# 1. Minimal (Development/Testing)
llm:
  provider: mock
datastore:
  type: in_memory

# 2. FileSystem Storage
datastore:
  type: filesystem
  root_path: ./data/insight_store

# 3. Production (requires API key)
llm:
  provider: openai
  model: gpt-3.5-turbo
```

### 🔬 Advanced Metrics

```yaml
# Spectral GED (NEW!)
metrics:
  spectral_evaluation:
    enabled: true
    weight: 0.3

# Multi-hop Analysis
metrics:
  use_multihop_gedig: true
  multihop_config:
    max_hops: 3
    decay_factor: 0.7

# All Metrics
metrics:
  use_normalized_ged: true
  use_entropy_variance_ig: true
  use_multihop_gedig: true
  spectral_evaluation:
    enabled: true
```

### ⚡ Performance Features

```yaml
# Layer1 Bypass (10x faster for known queries)
processing:
  enable_layer1_bypass: true
  bypass_uncertainty_threshold: 0.2

# Insight Auto-Registration
processing:
  enable_insight_registration: true
  enable_insight_search: true
  max_insights_per_query: 5

# Graph Search
graph:
  enable_graph_search: true
  hop_limit: 2
  neighbor_threshold: 0.4
```

### 🎯 Use Case Presets

```yaml
# High Performance
performance:
  enable_cache: true
  parallel_workers: 8
processing:
  enable_layer1_bypass: true
memory:
  max_retrieved_docs: 5

# High Quality
reasoning:
  max_cycles: 15
  convergence_threshold: 0.85
metrics:
  use_normalized_ged: true
  spectral_evaluation:
    enabled: true
output:
  response_style: detailed
  show_reasoning: true
```

## Testing Status

| Configuration | Status | Notes |
|--------------|--------|-------|
| Minimal | ✅ Passed | Default for tests |
| Spectral GED | ✅ Passed | Mathematical enhancement |
| Multi-hop | ✅ Passed | Requires more memory |
| Layer1 Bypass | ✅ Passed | Fast path |
| Graph Search | ✅ Passed | Associative retrieval |
| FileSystem | ✅ Passed | Persistent storage |
| All Features | ✅ Passed | Stress test |

## Key Settings Explained

### Spectral Evaluation (NEW)
- **Purpose**: Detect structural quality improvements using Laplacian eigenvalues
- **Default**: Disabled (backward compatible)
- **Impact**: Better insight detection for graph reorganization

### Layer1 Bypass
- **Purpose**: Skip full processing for high-confidence known queries
- **Default**: Disabled
- **Impact**: 10x speedup in production

### Graph Search
- **Purpose**: Multi-hop traversal for associative memory retrieval
- **Default**: Disabled
- **Impact**: Better context understanding

## Environment Variables

```bash
# Override config file
export INSIGHTSPIKE_CONFIG=./custom_config.yaml

# Set provider API keys
export OPENAI_API_KEY=your_key
export ANTHROPIC_API_KEY=your_key
```

## See Also

- [Architecture Documentation](docs/architecture/README.md)
- [Spectral GED Feature](docs/architecture/spectral_ged_feature.md)
- [Test Report](docs/test_results/config_variations_report_2025_01.md)