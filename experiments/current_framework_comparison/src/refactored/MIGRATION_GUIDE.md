# Migration Guide: UnifiedMainAgent

## Overview

The UnifiedMainAgent consolidates all 6 agent variants into a single, configurable class. This guide shows how to migrate from the old agents to the new unified approach.

## Quick Migration Table

| Old Agent | New Configuration |
|-----------|-------------------|
| `MainAgent` | `AgentMode.BASIC` |
| `EnhancedMainAgent` | `AgentMode.ENHANCED` |
| `MainAgentWithQueryTransform` | `AgentMode.QUERY_TRANSFORM` |
| `MainAgentAdvanced` | `AgentMode.ADVANCED` |
| `MainAgentOptimized` | `AgentMode.OPTIMIZED` |
| `GraphCentricMainAgent` | `AgentMode.GRAPH_CENTRIC` |

## Migration Examples

### 1. Basic MainAgent

**Old way:**
```python
from insightspike.core.agents.main_agent import MainAgent

agent = MainAgent(config)
agent.initialize()
result = agent.process_question("What is energy?")
```

**New way:**
```python
from unified_main_agent import UnifiedMainAgent, AgentConfig, AgentMode

config = AgentConfig.from_mode(AgentMode.BASIC)
agent = UnifiedMainAgent(config)
agent.initialize()
result = agent.process_question("What is energy?")
```

### 2. Enhanced MainAgent

**Old way:**
```python
from insightspike.core.agents.main_agent_enhanced import EnhancedMainAgent

agent = EnhancedMainAgent(config)
agent.initialize()
result = agent.process_question_enhanced("What is consciousness?")
```

**New way:**
```python
config = AgentConfig.from_mode(AgentMode.ENHANCED)
agent = UnifiedMainAgent(config)
agent.initialize()
result = agent.process_question("What is consciousness?")  # Same method name!
```

### 3. Query Transform Agent

**Old way:**
```python
from insightspike.core.agents.main_agent_with_query_transform import MainAgentWithQueryTransform

agent = MainAgentWithQueryTransform(config, enable_query_transformation=True)
agent.initialize()
result = agent.process_question("Complex question")
```

**New way:**
```python
config = AgentConfig.from_mode(AgentMode.QUERY_TRANSFORM)
agent = UnifiedMainAgent(config)
agent.initialize()
result = agent.process_question("Complex question")
```

### 4. Custom Configuration

**Old way (mixing features):**
```python
# Had to choose one agent class, couldn't mix features easily
agent = MainAgentOptimized(config)
# Couldn't add query transformation without using different class
```

**New way (flexible configuration):**
```python
config = AgentConfig(
    mode=AgentMode.BASIC,
    enable_query_transform=True,    # Add query transformation
    enable_caching=True,            # Add caching
    enable_async_processing=False,  # But skip async for simplicity
    cache_size=1000,
    max_cycles=5
)
agent = UnifiedMainAgent(config)
```

## Feature Flags

The UnifiedMainAgent supports these feature flags:

- `enable_query_transform`: Enable query transformation through graph
- `enable_graph_aware_memory`: Use graph-aware memory management
- `enable_multi_hop`: Enable multi-hop reasoning
- `enable_query_branching`: Enable parallel query exploration
- `enable_caching`: Cache query results
- `enable_async_processing`: Use async/parallel processing
- `enable_gpu_acceleration`: Use GPU for embeddings/GNN
- `enable_evolution_tracking`: Track query evolution

## API Compatibility

The UnifiedMainAgent maintains backward compatibility:

1. **Same initialization**: `agent.initialize()`
2. **Same main method**: `agent.process_question(question, max_cycles=3)`
3. **Same result format**: Returns dict with standard fields
4. **Same memory methods**: `agent.add_episode()` works as before

## Performance Considerations

1. **Basic mode** is fastest (no graph analysis)
2. **Enhanced mode** adds graph awareness with moderate overhead
3. **Query transform** adds transformation cycles
4. **Advanced mode** explores multiple paths (slower but thorough)
5. **Optimized mode** uses caching and parallelization for speed
6. **Graph-centric** mode focuses purely on graph operations

## Testing Your Migration

```python
# Test that results are similar
old_agent = MainAgent(config)
new_agent = UnifiedMainAgent(AgentConfig.from_mode(AgentMode.BASIC))

old_agent.initialize()
new_agent.initialize()

old_result = old_agent.process_question("Test question")
new_result = new_agent.process_question("Test question")

# Results should be functionally equivalent
assert old_result['success'] == new_result['success']
assert len(old_result['response']) > 0
assert len(new_result['response']) > 0
```

## Gradual Migration Strategy

1. **Phase 1**: Replace imports, use basic mode
2. **Phase 2**: Enable specific features as needed
3. **Phase 3**: Remove old agent files
4. **Phase 4**: Optimize configuration for your use case

## Common Issues

### Import Errors
```python
# Old
from insightspike.core.agents.main_agent_enhanced import EnhancedMainAgent

# New (update imports in experiments/)
from refactored.unified_main_agent import UnifiedMainAgent, AgentConfig, AgentMode
```

### Method Name Changes
- All agents now use `process_question()` (not `process_question_enhanced()` etc.)
- Configuration is passed via `AgentConfig`, not constructor arguments

### Feature Availability
- Some experimental features might not be fully migrated yet
- Check the source code comments for feature status

## Getting Help

If you encounter issues:
1. Check the `config_examples.py` for working examples
2. Look at the UnifiedMainAgent source for feature implementations
3. Use verbose mode to debug: `config.verbose = True`