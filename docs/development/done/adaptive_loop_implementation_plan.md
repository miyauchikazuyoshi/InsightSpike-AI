# Adaptive Loop Implementation Plan

**Date**: 2025-01-25  
**Status**: Planning  
**Priority**: High  
**Impact**: API call reduction by ~80%

## Overview

Implement adaptive exploration loop where L1-L2-L3 iterate with changing exploration parameters until spike is detected, then call LLM only once.

## Core Concept

```
Question → L1 (Set exploration scope) → L2 (Memory search) → L3 (Spike detection)
              ↑                                                    ↓
              ←────────── No spike: Adjust scope ─────────────────┘
                                                                   ↓
                                                    Spike detected: → L4 (LLM) once
```

## Key Benefits

1. **API Call Reduction**: LLM called only when spike detected (1 call instead of 5)
2. **Quality Guarantee**: LLM receives high-quality context with confirmed insights
3. **Exploration Efficiency**: Systematic exploration of knowledge space

## Implementation Strategy

### 1. Configuration
```yaml
processing:
  enable_adaptive_loop: false  # Feature flag
  adaptive_loop:
    max_exploration_attempts: 5
    initial_exploration_radius: 0.7
    radius_decay_factor: 0.8
    exploration_strategy: "narrowing"  # or "expanding", "alternating"
    spike_threshold_override: null  # Use default from graph config
```

### 2. Key Components from Old Implementation to Adopt

#### A. **Adaptive TopK Algorithm** (`learning/adaptive_topk.py`)
- Dynamic topK adjustment based on Layer1 analysis
- Factors considered:
  - Synthesis requirement (1.5x multiplier)
  - Query complexity (up to 1.3x)
  - Unknown ratio (up to 1.2x)
  - Low confidence (1.4x multiplier)
- Chain reaction potential estimation (0-1 score)

#### B. **Agent Loop with Adaptive Processing** (`agent_loop.py`)
```python
# Key features:
- Layer1 analysis integration
- Adaptive topK calculation per query
- Dynamic cycle adjustment (3-7 cycles based on analysis)
- Early stopping on high quality (> 0.8)
- Insight fact extraction and registration
```

#### C. **AdaptiveExplorer** (`enhanced_query_transformer.py`)
- Temperature-based exploration (like simulated annealing)
- Success pattern learning and reuse
- Direction scoring based on:
  - Unexplored directions (-0.1 per previous exploration)
  - Connection density (+0.2 per connection)
  - Previous success (+0.3 per successful pattern)
  - Low confidence exploration boost (+0.2)

### 3. Implementation Architecture

#### A. New MainAgent Methods

```python
class MainAgent:
    def process_question_adaptive(self, question: str, **kwargs):
        """
        Adaptive processing with exploration loop.
        Key difference: LLM called only after spike detection.
        """
        
    def _adaptive_exploration_cycle(self, question: str, exploration_params: dict):
        """
        Single L1-L2-L3 exploration attempt.
        Returns: (spike_detected, graph_analysis, retrieved_docs)
        """
        
    def _calculate_adaptive_topk(self, l1_analysis: dict) -> dict:
        """Calculate dynamic topK based on Layer1 analysis"""
        
    def _adjust_exploration_radius(self, attempt: int, strategy: str) -> float:
        """Adjust exploration radius based on strategy"""
```

#### B. Modified Layer Methods

```python
# Layer1: Accept exploration radius
def analyze_uncertainty(self, question: str, exploration_radius: float = 0.7):
    # Use radius to filter concepts
    
# Layer2: Accept dynamic topK
def retrieve(self, question: str, k: int = None, similarity_threshold: float = None):
    # Use adaptive k and threshold
    
# Layer3: Return detailed metrics for spike decision
def analyze_documents(self, docs: List[dict]) -> dict:
    # Include detailed GED/IG metrics for exploration decision
```

### 4. Exploration Strategies

1. **Narrowing**: Start broad, focus down
   - Initial radius: 0.7 → 0.56 → 0.45 → ...
   - Good for: Finding specific insights

2. **Expanding**: Start narrow, broaden out  
   - Initial radius: 0.3 → 0.45 → 0.6 → ...
   - Good for: Connecting distant concepts

3. **Alternating**: Oscillate between broad/narrow
   - Pattern: 0.7 → 0.4 → 0.8 → 0.3 → ...
   - Good for: Comprehensive exploration

### 5. Integration Points

- **Layer1 Error Monitor**: Add exploration_params support
- **Layer2 Memory Manager**: Accept dynamic similarity thresholds  
- **Layer3 Graph Reasoner**: Return detailed exploration metrics
- **ConfigurableAgent**: Add ADAPTIVE mode

## Implementation Steps

1. [x] Create development branch and this planning document
2. [x] Review old implementation files for reusable components
3. [ ] Port adaptive_topk.py to current codebase
4. [ ] Add configuration schema for adaptive_loop
5. [ ] Implement core adaptive loop in MainAgent
6. [ ] Modify Layer1-3 to support exploration parameters
7. [ ] Add AdaptiveExplorer for temperature-based exploration
8. [ ] Create tests for different exploration strategies
9. [ ] Benchmark API call reduction with math concept experiment
10. [ ] Document usage and best practices

## Code Migration Plan

### Phase 1: Port Core Components
```bash
# Copy from old implementation
cp /ISbackups/0717/.../learning/adaptive_topk.py src/insightspike/learning/
# Adapt imports and integrate with current Layer1
```

### Phase 2: Extend Configuration
```python
# Add to ProcessingConfig
class AdaptiveLoopConfig(BaseModel):
    enable_adaptive_loop: bool = False
    max_exploration_attempts: int = 5
    initial_exploration_radius: float = 0.7
    exploration_strategy: str = "narrowing"
    # ... other parameters
```

### Phase 3: Implement in MainAgent
```python
def process_question(self, question: str, **kwargs):
    if self.config.processing.enable_adaptive_loop:
        return self.process_question_adaptive(question, **kwargs)
    else:
        return self._process_standard(question, **kwargs)
```

## Expected Results

For mathematical concept evolution experiment:
- Before: 50 concepts × 2 API calls = 100 calls
- After: 50 concepts × 1 API call = 50 calls (50% reduction minimum)
- With spike detection optimization: ~20-30 calls (70-80% reduction)

## Risks and Mitigations

1. **Risk**: May not find spikes with limited exploration
   - **Mitigation**: Fallback to standard processing after max attempts

2. **Risk**: Increased latency from multiple L1-L3 cycles
   - **Mitigation**: Parallelize exploration attempts when possible

3. **Risk**: Memory/computation overhead
   - **Mitigation**: Cache intermediate results, reuse graph constructions

## Success Criteria

1. API calls reduced by at least 50%
2. Response quality maintained or improved
3. Processing time within 2x of standard mode
4. All existing tests pass
5. New tests for adaptive mode pass

## References

- Old AdaptiveExplorer: `/ISbackups/0717/InsightSpike-AI/src/insightspike/core/query_transformation/enhanced_query_transformer.py`
- MainAgent Advanced: `/ISbackups/0717/InsightSpike-AI/src/insightspike/core/agents/main_agent_advanced.py`
- Current MainAgent: `/src/insightspike/implementations/agents/main_agent.py`
- Layer1 Bypass: `/docs/development/done/layer1_bypass_mechanism.md`