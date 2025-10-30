---
status: active
category: meta
owner: refactor-wg
created: 2025-08-25
last_review: 2025-08-25
---

# Adaptive Loop Implementation Notes

**Date**: 2025-01-25  
**Purpose**: Track implementation progress and decisions

## Overview
Implementing adaptive exploration loop to reduce LLM API calls by detecting spikes through L1-L2-L3 iterations before calling L4.

## Key Design Decisions

### 1. Architecture Pattern
- **Clean Architecture**: Separate concerns into distinct modules
- **Dependency Injection**: Components receive dependencies, don't create them
- **Strategy Pattern**: Exploration strategies are pluggable
- **Interface Segregation**: Small, focused interfaces

### 2. Module Organization
```
src/insightspike/adaptive/
├── core/           # Core logic
├── strategies/     # Pluggable strategies  
├── calculators/    # Calculation utilities
└── learning/       # Pattern learning
```

### 3. Core Components
- **AdaptiveProcessor**: Orchestrates the adaptive loop
- **ExplorationLoop**: Manages L1→L2→L3 cycle (no LLM)
- **ExplorationStrategy**: Decides parameter adjustments
- **TopKCalculator**: Dynamic topK based on L1 analysis
- **PatternLearner**: Learns from successful explorations

### 4. Integration Points
- MainAgent uses AdaptiveProcessor when `enable_adaptive_loop=True`
- Existing layers (L1-L3) remain unchanged
- L4 called only after spike detection

### 5. Configuration
```yaml
processing:
  enable_adaptive_loop: true
  adaptive_loop:
    exploration_strategy: "narrowing"
    max_exploration_attempts: 5
    initial_exploration_radius: 0.7
```

## Implementation Checklist

### Phase 1: Core Components
- [ ] Create adaptive module structure
- [ ] Define interfaces in adaptive/core/interfaces.py
- [ ] Port adaptive_topk.py from old implementation
- [ ] Implement ExplorationLoop
- [ ] Create strategy implementations

### Phase 2: Integration
- [ ] Implement AdaptiveProcessor
- [ ] Modify MainAgent to use adaptive processing
- [ ] Update configuration schema
- [ ] Create unit tests
- [ ] Integration testing

### Phase 3: Enhancement
- [ ] Pattern learning implementation
- [ ] Temperature management
- [ ] Performance benchmarks
- [ ] Documentation

## Key Files to Create/Modify

### New Files
1. `src/insightspike/adaptive/__init__.py`
2. `src/insightspike/adaptive/core/interfaces.py`
3. `src/insightspike/adaptive/core/adaptive_processor.py`
4. `src/insightspike/adaptive/core/exploration_loop.py`
5. `src/insightspike/adaptive/strategies/base.py`
6. `src/insightspike/adaptive/strategies/narrowing.py`
7. `src/insightspike/adaptive/calculators/adaptive_topk.py`

### Modified Files
1. `src/insightspike/config/models.py` - Add AdaptiveLoopConfig
2. `src/insightspike/implementations/agents/main_agent.py` - Add adaptive processing

## Success Criteria
1. API calls reduced by 50%+ for math concept evolution experiment
2. No degradation in response quality
3. Clean, testable code structure
4. All existing tests pass

## References
- Original adaptive_topk: `/ISbackups/0717/.../learning/adaptive_topk.py`
- Agent loop: `/ISbackups/0717/.../agent_loop.py`
- Clean architecture design: `adaptive_loop_clean_architecture.md`

---
Starting implementation now!