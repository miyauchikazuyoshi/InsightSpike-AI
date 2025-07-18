# Deprecation and Cleanup Plan

## 📅 Overview

This document outlines the plan for removing deprecated code and legacy modules from InsightSpike.

## 🗑️ Items to Remove

### 1. Deprecated Agent Implementations
**Location**: `src/insightspike/implementations/agents/deprecated/`

**Files**:
- `main_agent_advanced.py`
- `main_agent_enhanced.py`
- `main_agent_graph_centric.py`
- `main_agent_optimized.py`
- `main_agent_with_query_transform.py`

**Status**: All functionality has been consolidated into `ConfigurableAgent`
**Action**: Delete entire `deprecated/` directory

### 2. Legacy Package
**Location**: `src/insightspike/legacy/`

**Files**:
- `agent_loop.py` - Provides `cycle()` and `adaptive_loop()` functions

**Usage**: Still referenced by 8 files for backward compatibility
**Action**: Keep for now, but mark for removal in v1.0

### 3. Legacy Config System
**Location**: `src/insightspike/config/legacy_config.py`

**Status**: New Pydantic-based config system is in place
**Action**: Keep for migration period (3 months), then remove

## 📋 Removal Schedule

### Immediate (Now)
1. ✅ Delete `implementations/agents/deprecated/` directory
2. ✅ Remove `__pycache__` directories
3. ✅ Update documentation to remove references to deprecated agents

### Short Term (1 month)
1. Add deprecation warnings to legacy package
2. Update all internal code to stop using legacy imports
3. Document migration path in CHANGELOG

### Medium Term (3 months)
1. Remove `legacy_config.py` and related imports
2. Update all examples and tests

### Long Term (v1.0 release)
1. Remove entire `legacy/` package
2. Clean up any remaining backward compatibility code

## 🔍 Impact Analysis

### Files using legacy imports:
- `config/models.py` - Uses legacy config for conversion
- `__init__.py` - Exports legacy config for compatibility
- `layer4_llm_interface.py` - Might use legacy patterns
- `detection/eureka_spike.py` - Check for legacy usage
- `cli/__init__.py` - May depend on legacy functions

## ✅ Migration Guide

### For deprecated agents:
```python
# Old
from insightspike.implementations.agents.deprecated.main_agent_enhanced import EnhancedMainAgent

# New
from insightspike.implementations.agents import ConfigurableAgent, AgentMode
agent = ConfigurableAgent(mode=AgentMode.ENHANCED)
```

### For legacy config:
```python
# Old
from insightspike.config.legacy_config import Config

# New
from insightspike.config import InsightSpikeConfig
```

### For legacy functions:
```python
# Old
from insightspike.legacy import cycle

# New
# Use MainAgent.process_question() directly
```

## 📝 Communication Plan

1. **Add deprecation warnings** in code
2. **Update README** with migration guide
3. **Create GitHub issue** tracking removal
4. **Announce in release notes**

## 🎯 Success Criteria

- [ ] No deprecated directories remain
- [ ] All tests pass without legacy code
- [ ] Documentation is updated
- [ ] Users have clear migration path
- [ ] Git history preserves old code