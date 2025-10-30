---
status: active
category: infra
owner: refactor-wg
created: 2025-08-25
last_review: 2025-08-25
---

# Minimal Config Migration Plan

## ✅ Migration Complete!

All essential files have been successfully migrated to use Pydantic config directly. ConfigConverter has been removed from the codebase.

### Core Source Files (6) - All Completed ✅

1. **`src/insightspike/processing/embedder.py`**
   - [x] Accept `InsightSpikeConfig` directly
   - [x] Use `config.embedding.model_name` and `config.embedding.dimension`

2. **`src/insightspike/cli/legacy.py`**
   - [x] Accept `InsightSpikeConfig` directly
   - [x] Use proper config attributes

3. **`src/insightspike/implementations/layers/scalable_graph_builder.py`**
   - [x] Accept `InsightSpikeConfig` in constructor
   - [x] Use `config.embedding.dimension`

4. **`src/insightspike/implementations/layers/layer3_graph_reasoner.py`**
   - [x] Accept `InsightSpikeConfig` in constructor
   - [x] Use `config.embedding.dimension`

5. **`src/insightspike/implementations/layers/layer4_llm_interface.py`**
   - [x] Accept `InsightSpikeConfig` in constructor
   - [x] Use `config.llm.*` attributes

6. **`src/insightspike/implementations/agents/main_agent.py`**
   - [x] Accept `InsightSpikeConfig` directly (not legacy format)
   - [x] Remove need for ConfigConverter

### Example File (1) - Completed ✅

7. **`examples/config_examples.py`**
   - [x] Update to use new config system
   - [x] Show proper usage patterns

### Additional Updates - Completed ✅

- **`src/insightspike/config/presets.py`**
  - [x] Updated to return Pydantic models directly
  - [x] Added static methods for each preset
  
- **`src/insightspike/config/models.py`**
  - [x] Added new config classes (LLMConfig, EmbeddingConfig, etc.)
  - [x] Updated InsightSpikeConfig with new structure
  - [x] Added PathsConfig for backward compatibility

- **`src/insightspike/cli/spike.py`**
  - [x] Updated DependencyFactory to pass Pydantic config directly
  - [x] Removed ConfigConverter dependency

- **`src/insightspike/config/converter.py`**
  - [x] DELETED - No longer needed!

## 🎉 Results

- **Minimal Risk**: Only touched essential files
- **Quick Migration**: Completed in a single session
- **Clean Codebase**: No more ConfigConverter
- **Future Ready**: Tests and experiments can be built fresh with new config
- **Backward Compatible**: Added necessary fields to support legacy code

## 📝 Notes

The remaining 39 files (tests and experiments) that still use legacy config patterns can be updated when they are rewritten or modified. The current implementation provides backward compatibility through:

1. The Pydantic config includes a `paths` field for legacy code
2. Layer files can handle both Pydantic and legacy config objects
3. The system gracefully falls back to defaults when needed

## 🔧 Usage

With ConfigConverter removed, all new code should use Pydantic config directly:

```python
# Old way (no longer works)
from insightspike.config.converter import ConfigConverter  # DELETED!
config = ConfigConverter.to_legacy(pydantic_config)

# New way (direct usage)
from insightspike.config.models import InsightSpikeConfig
from insightspike.config.presets import ConfigPresets

config = ConfigPresets.development()  # Returns InsightSpikeConfig
agent = MainAgent(config)  # Direct usage!
```

The migration is complete! 🚀