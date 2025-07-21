# Configuration System Migration Plan

## 📅 Migration Timeline

### ✅ Completed Phases (2025-07-17)

#### Phase 1: New Configuration System
- Created unified Pydantic-based configuration (`InsightSpikeConfig`)
- Full YAML/JSON file support
- Environment variable overrides
- Presets for common use cases

#### Phase 2: CLI Migration  
- Updated `spike` CLI to use new configuration system
- Removed direct `ConfigManager` dependencies
- Maintained backward compatibility

#### Phase 3: Test Updates
- Migrated 25 test files using automatic migration script
- Updated imports from `legacy_config` to new system
- Fixed attribute access patterns

#### Phase 4: Experiment Scripts
- Updated all experiment scripts to use new configuration
- Maintained functionality while using new API

### ✅ Current Status (2025-01-21) - MIGRATION COMPLETE!

**Legacy Files Removed:**
- ✅ **DELETED** `/src/insightspike/config/legacy_config.py`
- ✅ **DELETED** `/src/insightspike/config/simple_config.py`
- ✅ **DELETED** `/src/insightspike/config/models_backup.py`

**Migration Completed Today (2025-01-21):**
- ✅ Updated `layer4_llm_interface.py` to remove legacy_config parameter
- ✅ Updated `layer2_memory_manager.py` to use new config directly
- ✅ Updated `layer2_compatibility.py` to remove legacy references
- ✅ Cleaned up `config/__init__.py` to remove legacy imports
- ✅ Fixed `__init__.py` to remove LegacyConfigModule
- ✅ All tests pass with new configuration system

**Previously Completed (2025-07-18):**
- ✅ Migrated 7 essential files to use Pydantic config directly
- ✅ Updated ConfigPresets to return Pydantic models
- ✅ Updated DependencyFactory to pass Pydantic config directly
- ✅ **DELETED `/src/insightspike/config/converter.py`** - No longer needed!
- ✅ All core functionality now works without ConfigConverter

### 📋 Phase 5: Legacy Removal Plan

#### Step 1: Deprecation Period (2025-07-17 → 2025-08-17)
- [x] Add deprecation warnings in `core/config.py`
- [x] Monitor usage and fix any issues that arise
  - Fixed MainAgent config attribute access issues (2025-07-18)
  - Fixed layer4_llm_interface config access issues (2025-07-18)
  - Implemented ConfigConverter for complete legacy format support (2025-07-18)
- [ ] Update documentation to emphasize new system
- [ ] Announce deprecation in release notes

#### Step 2: Final Migration (2025-08-17 → 2025-08-24)
- [x] Update remaining layer files to use new config directly:
  - [x] `layer1_error_monitor.py` (already uses new config)
  - [x] `layer2_memory_manager.py` (only uses legacy in deprecated methods)
  - [x] `layer4_prompt_builder.py` (updated 2025-07-18 - now accepts config as parameter)
  - [x] `scalable_graph_builder.py` (updated 2025-07-18 - now accepts config as parameter)
- [x] Remove legacy imports from `__init__.py` (core/__init__.py already cleaned)
- [x] Update `main_agent.py` to only accept new config format (via ConfigConverter)

#### Step 3: ConfigConverter Migration ✅ COMPLETE!
- [x] Migrated 7 essential files from legacy patterns to direct Pydantic usage
- [x] Priority order:
  - [x] High: Core functionality (6 files in src/)
  - [x] Medium: Examples and CLI (1 file)
  - [ ] Low: Tests and experiments (39 files) - To be done when rewriting
- [x] Removed ConfigConverter completely from codebase
- [x] Tested all migrations - system works perfectly!

#### Step 4: Legacy Removal ✅ COMPLETED (2025-01-21)
- [x] Delete `/src/insightspike/config/legacy_config.py`
- [x] Delete `/src/insightspike/config/simple_config.py`
- [x] Delete `/src/insightspike/config/converter.py` (already done 2025-07-18)
- [x] Remove legacy config support from all files
- [x] Clean up compatibility imports in `config/__init__.py`
- [ ] Update all documentation (partially done)

### 🔍 Pre-removal Checklist

Before removing legacy files, ensure:

- [x] All tests pass with new configuration (ConfigConverter ensures compatibility)
- [ ] No deprecation warnings in CI/CD
- [ ] Documentation fully updated
- [ ] Migration guide prominently displayed
- [ ] Version bump (2.0.0) prepared

### 📊 Current Status Summary (2025-07-18)

**Completed:**

- ✅ Phase 1-4: New config system, CLI migration, test updates, experiment scripts
- ✅ ConfigConverter implementation for robust legacy support
- ✅ Removed `/src/insightspike/core/config.py`
- ✅ Fixed all config access issues in MainAgent and layers
- ✅ All layer files now using new config or compatible
- ✅ Step 2 of Phase 5 completed (all layer files updated)

**Remaining Work:**

- 📝 Update documentation to emphasize new system
- 📢 Announce deprecation in release notes
- 🗑️ Final removal of legacy files after deprecation period

### 📝 Notes

1. **Migration Script Available:** `scripts/migrate_config.py` can automatically update remaining files

2. **Backward Compatibility:** The current setup maintains full backward compatibility through:
   - ~~Import redirects in `core/config.py`~~ (removed 2025-07-18)
   - `to_legacy_config()` conversion method
   - Re-exports in `config/__init__.py`
   - NEW: `ConfigConverter` class for comprehensive legacy format generation

3. **Risk Assessment:** 
   - **Current**: Low risk with ConfigConverter in place
   - **Migration**: Medium risk - requires updating 46 files
   - **Recommendation**: Keep ConfigConverter until all files migrated

4. **Recent Issues Fixed (2025-07-18):**
   - `'Config' object has no attribute 'core'` → Fixed with ConfigConverter
   - `'CoreConfig' object has no attribute 'provider'` → Fixed llm.provider access
   - `'CompatibleL2MemoryManager' object has no attribute 'embedder'` → Fixed convergence check

5. **Cleanup Done (2025-07-18):**
   - Removed `/src/insightspike/core/config.py` (no longer needed)
   - All imports now use `/src/insightspike/config/` directly

6. **Additional Fixes (2025-07-18):**
   - Fixed episode storage return type mismatch in MainAgent
   - Added "clean" LLM provider to allowed values in InsightSpikeConfig
   - Fixed DependencyFactory to merge config.yaml values with presets
   - All layer files now accept config as parameter instead of calling get_config()

### 🚀 Benefits After Migration

1. **Cleaner Codebase:** Remove ~1000 lines of legacy code
2. **Single Source of Truth:** One configuration system
3. **Better Validation:** Pydantic provides automatic validation
4. **YAML Support:** Direct config.yaml integration
5. **Environment Variables:** Full support for deployment configs

### ✅ High Priority Migration Checklist - COMPLETE!

All 7 essential files have been successfully migrated:

1. **`src/insightspike/processing/embedder.py`** ✅
   - [x] Now accepts InsightSpikeConfig directly
   - [x] Handles both Pydantic and legacy config

2. **`src/insightspike/cli/legacy.py`** ✅
   - [x] Updated config display to handle both formats
   - [x] Properly accesses config attributes

3. **`src/insightspike/implementations/layers/scalable_graph_builder.py`** ✅
   - [x] Accepts InsightSpikeConfig directly
   - [x] Uses config.embedding.dimension

4. **`src/insightspike/implementations/layers/layer3_graph_reasoner.py`** ✅
   - [x] Accepts InsightSpikeConfig directly
   - [x] Properly handles embedding dimension

5. **`src/insightspike/implementations/layers/layer4_llm_interface.py`** ✅
   - [x] Accepts InsightSpikeConfig directly via Union type
   - [x] Factory function handles Pydantic config

6. **`src/insightspike/implementations/agents/main_agent.py`** ✅
   - [x] Accepts InsightSpikeConfig directly
   - [x] No more ConfigConverter dependency!

7. **`examples/config_examples.py`** ✅
   - [x] Completely rewritten for new config system
   - [x] Shows proper Pydantic usage patterns

### 📞 Contact

Questions about migration? Contact the InsightSpike team or check:

- [Migration Guide](/docs/MIGRATION_GUIDE.md)
- [Configuration Documentation](/docs/architecture/configuration.md)
- [Legacy Config Patterns Audit](./legacy_config_patterns_audit.md)
