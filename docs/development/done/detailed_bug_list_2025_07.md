# InsightSpike Detailed Bug List

Date: 2025-07-27
Status: After pipeline testing

## 🔴 Critical Bugs (Must Fix)

### 1. C-Value Field Inconsistency
- **Location**: Throughout codebase
- **Issue**: Two fields exist: `c` and `c_value`
- **Current Behavior**:
  - Episode storage uses `c`
  - Some code expects `c_value`
  - Display shows `c`, API expects `c_value`
- **Impact**: Data inconsistency, confusion
- **Fix**: Standardize to `c_value` everywhere

### 2. Missing Config Converter Module
- **Location**: `insightspike/config/converter.py`
- **Issue**: Module referenced but doesn't exist
- **Impact**: Cannot use Pydantic configs
- **Fix**: Either create module or remove references

### 3. Config Access Pattern Chaos
- **Location**: All layers
- **Issue**: Mixed dict/object access
- **Examples**:
  ```python
  # Some code does this:
  config.processing.enable_learning
  # Other code does this:
  config.get("processing", {}).get("enable_learning", False)
  ```
- **Impact**: AttributeError when patterns don't match
- **Fix**: Add universal accessor utility

## 🟡 High Priority Bugs

### 4. CachedMemoryManager Episodes Property
- **Location**: `src/insightspike/implementations/layers/cached_memory_manager.py`
- **Issue**: Returns only cached episodes, not all
- **Warning**: "Accessing episodes property returns only cached episodes"
- **Impact**: Incomplete data retrieval
- **Fix**: Return all episodes from datastore

### 5. ScalableGraphManager Not Used
- **Location**: Layer2 integration
- **Warning**: "Layer2 not using ScalableGraphManager"
- **Impact**: Performance degradation
- **Fix**: Investigate why it's not initialized

### 6. Advanced Metrics Missing
- **Location**: Metrics system
- **Warning**: "Advanced metrics not available"
- **Impact**: Reduced spike detection accuracy
- **Fix**: Implement or remove warning

## 🟢 Medium Priority Bugs

### 7. Memory Usage Warnings
- **Issue**: Constant warnings about 500MB threshold
- **Impact**: Noisy logs
- **Fix**: Adjust threshold or optimize memory

### 8. SQLite DataStore Vector Issues
- **Location**: `sqlite_store.py`
- **Issue**: Cannot handle vector storage properly
- **Status**: Users forced to use memory/filesystem stores
- **Fix**: Implement proper vector serialization

### 9. Episode Text Duplication
- **Issue**: Q&A episodes have identical text
- **Example**: All responses are "I don't have enough information"
- **Impact**: Poor user experience
- **Fix**: Improve LLM prompting

## 🔵 Fixed Bugs (Completed)

### ✅ MemoryStore save_episodes
- **Issue**: Was overwriting instead of appending
- **Fix**: Changed to use extend()
- **Status**: Fixed

### ✅ FileSystemDataStore save_episodes
- **Issue**: Same overwriting problem
- **Fix**: Load existing + append new
- **Status**: Fixed

### ✅ Config dict access errors
- **Issue**: MainAgent assumed object config
- **Fix**: Added dict.get() fallbacks
- **Status**: Partially fixed (needs comprehensive solution)

### ✅ Numpy shape errors
- **Issue**: Extra dimensions in vectors
- **Fix**: Added flatten() operations
- **Status**: Fixed

## Warning Summary

Current warnings in every run:
1. "Memory usage warning: 530+ MB (threshold: 500 MB)"
2. "Reduced cache size to 1 due to memory pressure"
3. "Accessing episodes property on CachedMemoryManager returns only cached episodes!"
4. "Layer2 not using ScalableGraphManager, falling back to full rebuild"
5. "Advanced metrics not available"
6. "Requested IG algorithm 'advanced' not available, using simple"

## Root Causes

1. **Legacy Code Debt**: Mix of old and new config systems
2. **Incomplete Refactoring**: Started but not finished migrations
3. **Missing Tests**: No tests caught these issues
4. **Rapid Development**: Features added without integration testing
5. **Documentation Gap**: No clear architecture guidelines

## Immediate Action Items

1. **Create safe_config_access utility**:
   ```python
   def safe_get(config, *keys, default=None):
       """Works with both dict and object configs"""
   ```

2. **Standardize c_value field**:
   - Global search/replace `"c"` → `"c_value"`
   - Add data migration

3. **Suppress known warnings**:
   - Add warning filter for development
   - Fix root causes in parallel

4. **Create integration tests**:
   - Test each config route
   - Test each provider
   - Test episode lifecycle

## Long-term Solutions

1. **Single Config System**: Choose dict OR object, not both
2. **Schema Validation**: Use Pydantic everywhere
3. **Proper Logging Levels**: Debug vs Info vs Warning
4. **Comprehensive Test Suite**: Unit + Integration + E2E
5. **Architecture Documentation**: Clear patterns and guidelines