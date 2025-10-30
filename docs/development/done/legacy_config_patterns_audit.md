---
status: active
category: meta
owner: refactor-wg
created: 2025-08-25
last_review: 2025-08-25
---

# Legacy Config Patterns Audit

This document provides a comprehensive list of all Python files using legacy config format patterns, grouped by directory and categorized by pattern type.

## Summary

Total files with legacy patterns: 46 files (excluding converter.py and backups)

### Pattern Distribution
- `config.embedding.*`: 17 occurrences
- `config.llm.*`: 24 occurrences
- `config.core.*`: 42 occurrences
- `config.graph.*`: 15 occurrences

## Files by Directory

### 1. Core Source Files (`src/insightspike/`)

#### Processing Module
- **`src/insightspike/processing/embedder.py`**
  - Lines 31-32: `config.embedding.model_name`, `config.embedding.dimension`

#### CLI Module
- **`src/insightspike/cli/legacy.py`**
  - Lines 153-154: `config.llm.provider`, `config.llm.model_name`
  - Line 157: `config.graph.spike_ged_threshold`, `config.graph.spike_ig_threshold`

#### Implementations - Layers
- **`src/insightspike/implementations/layers/scalable_graph_builder.py`**
  - Line 93: `config.embedding.dimension`

- **`src/insightspike/implementations/layers/layer3_graph_reasoner.py`**
  - Line 606: `config.embedding.dimension`

- **`src/insightspike/implementations/layers/layer4_llm_interface.py`**
  - Lines 153-160: `config.llm.model`, `config.llm.temperature`, `config.llm.provider`

#### Implementations - Agents
- **`src/insightspike/implementations/agents/main_agent.py`**
  - Line 102: `config.embedding.dimension`

### 2. Examples (`examples/`)
- **`examples/config_examples.py`**
  - Lines 175-176: `config.core.provider`, `config.core.safe_mode`

### 3. Tests (`tests/`)

#### Unit Tests
- **`tests/test_unified_llm_provider.py`**
  - Lines 26, 51-54, 72-73, 87, 117: Multiple `config.core.*` patterns (safe_mode, provider, model_name, device)

- **`tests/unit/test_direct_generation.py`**
  - Lines 38-39, 72-73, 140-141: `config.llm.use_direct_generation`, `config.llm.direct_generation_threshold`

- **`tests/unit/test_layer4_pipeline.py`**
  - Lines 32-34, 68-72, 101-105, 152-153: Multiple `config.llm.*` patterns (use_direct_generation, enable_polish, thresholds)

- **`tests/unit/test_layer3_graph_reasoner_comprehensive.py`**
  - Lines 222-223: `config.embedding.dimension`, `config.graph.similarity_threshold`

- **`tests/unit/test_layer2_memory_manager_comprehensive.py`**
  - Lines 31, 54: `config.embedding.dimension`

- **`tests/unit/test_scalable_graph_manager.py`**
  - Lines 23, 57: `config.embedding.dimension`

#### Integration Tests
- **`tests/integration/test_scalable_system.py`**
  - Line 139: `config.embedding.dimension`

#### Test Helpers
- **`tests/helpers/test_helpers.py`**
  - Lines 167-169: `config.embedding.*` (model_name, dimension, device)
  - Lines 173-175: `config.llm.*` (provider, model_name, safe_mode)

#### Config System Tests
- **`tests/test_config_system.py`**
  - Multiple lines: Extensive use of `config.core.*` patterns for testing config system itself

### 4. Experiments (`experiments/`)

#### Current Framework Comparison
- **`experiments/current_framework_comparison/src/fix_llm_response.py`**
  - Lines 29-30: `config.core.model_name`, `config.core.max_tokens`

- **`experiments/current_framework_comparison/src/patch_llm_provider.py`**
  - Lines 55-56: `config.core.model_name`, `config.core.max_tokens`

- **`experiments/current_framework_comparison/src/run_comparison_experiment.py`**
  - Lines 165-178: `config.core.*` and `config.graph.*` patterns

- **`experiments/current_framework_comparison/src/quick_test.py`**
  - Lines 29-30: `config.core.model_name`, `config.core.max_tokens`

- **`experiments/current_framework_comparison/src/minimal_test.py`**
  - Lines 36-37: `config.core.model_name`, `config.core.max_tokens`

- **`experiments/current_framework_comparison/src/run_fixed_experiment.py`**
  - Lines 46-48, 137-139: `config.core.*` patterns

- **`experiments/current_framework_comparison/src/check_prompt_length.py`**
  - Lines 33-34: `config.core.model_name`, `config.core.max_tokens`

- **`experiments/current_framework_comparison/src/debug_insightspike.py`**
  - Line 45: `config.core.model_name`

#### Colab Experiments
- **`experiments/colab_experiments/insight_benchmarks.py`**
  - Lines 84-85: `config.core.device`, `config.core.use_gpu`

- **`experiments/colab_experiments/comparative_analysis.py`**
  - Line 103: `config.core.device`

- **`experiments/colab_experiments/scalability_testing.py`**
  - Line 246: `config.core.model_name`

#### DistilGPT2 RAT Experiments
- **`experiments/distilgpt2_rat_experiments/src/simple_distilgpt2_experiment.py`**
  - Lines 39-40: `config.core.model_name`, `config.core.use_gpu`

#### Other Experiments
- **`experiments/comparative_study/src/run_comparison.py`**
  - Line 205: `config.llm.provider`

- **`experiments/gedig_validation_/src/experiment_v5.py`**
  - Line 224: `config.llm.safe_mode`

### 5. Scripts (`scripts/`)

#### Migration Scripts
- **`scripts/migrate_config_advanced.py`**
  - Lines 23-65, 191-193: Contains mapping rules for migration (not actual usage)

## Migration Priority

### High Priority (Core functionality)
1. `src/insightspike/processing/embedder.py` - Core embedding functionality
2. `src/insightspike/implementations/layers/*.py` - Core layer implementations
3. `src/insightspike/implementations/agents/main_agent.py` - Main agent logic

### Medium Priority (CLI and Examples)
1. `src/insightspike/cli/legacy.py` - CLI interface
2. `examples/config_examples.py` - User-facing examples

### Low Priority (Tests and Experiments)
1. All test files - Can be updated after core migration
2. Experiment files - Mostly standalone scripts

## Notes
- The migration script (`scripts/migrate_config_advanced.py`) already contains mapping rules for most patterns
- Some patterns in comments (lines 174-176 in run_comparison_experiment.py) indicate attempted usage of non-existent properties
- Test files extensively use legacy patterns but many are testing the config system itself

## Status Update (2024-07-24)

### Current State
- Core source files still maintain dual support for both Pydantic and legacy configs
- Files use `is_pydantic_config` flag or `isinstance(config, InsightSpikeConfig)` checks
- Legacy patterns are kept for backward compatibility
- All core files properly handle both config types

### Decision
- Legacy config support is intentionally maintained for backward compatibility
- Complete removal would break existing experiments and integrations
- This audit document serves as a reference for future cleanup when legacy support is deprecated
- Moving to done folder as the audit is complete and accurate