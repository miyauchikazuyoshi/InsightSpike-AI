# PoC Implementation Plan for Paper Update

## Overview
This document outlines the concrete implementation plan for the Proof of Concept (PoC) features that will support the paper restructuring. We separate realistically implementable features from speculative concepts.

## Realistically Implementable Features (PoC)

### 1. Wake-Sleep Cycle Implementation ✅
**What**: Dual-mode optimization using the same geDIG objective function
- Wake mode: minimize f(G₁, G₂) for cost efficiency
- Sleep mode: maximize f(G₁, G₂) for quality improvement

**Implementation**:
```python
class WakeSleepManager:
    def __init__(self, gediq_calculator):
        self.gediq = gediq_calculator
        self.mode = "wake"  # or "sleep"
        
    def process(self, input_data):
        if self.mode == "wake":
            return self.wake_process(input_data)
        else:
            return self.sleep_process(input_data)
            
    def wake_process(self, query):
        # Minimize cost: Find efficient solution
        candidates = self.donut_search(query)
        return self.minimize_gediq(candidates)
        
    def sleep_process(self):
        # Maximize reward: Consolidate and optimize
        contradictions = self.find_contradictions()
        return self.maximize_gediq(contradictions)
```

### 2. Donut Search Implementation ✅
**What**: Query-centric search with inner and outer radius filtering
- Inner radius: Filter out known information
- Outer radius: Filter out irrelevant information

**Implementation**:
```python
def donut_search(query_vec, knowledge_vectors, inner_r=0.2, outer_r=0.8):
    results = []
    for node_id, node_vec in knowledge_vectors.items():
        distance = np.linalg.norm(query_vec - node_vec)
        if inner_r < distance < outer_r:
            results.append({
                'node_id': node_id,
                'distance': distance,
                'novelty': (distance - inner_r) / (outer_r - inner_r)
            })
    return results
```

### 3. Maze Navigation with Obstacles as Queries ✅
**What**: Treat spatial obstacles as implicit queries to the system
- Unifies language and spatial reasoning
- Same geDIG mechanism for both domains

**Implementation**:
```python
class ObstacleAsQueryNavigator:
    def __init__(self, gediq_system):
        self.gediq = gediq_system
        
    def encounter_obstacle(self, position, obstacle_type):
        # Convert obstacle to query
        query = self.obstacle_to_query(position, obstacle_type)
        
        # Process through wake mode
        solution = self.gediq.wake_process(query)
        
        # Execute navigation action
        return self.query_result_to_action(solution)
```

### 4. Experimental Validation ✅
**What**: Concrete experiments to validate the approach
- Wake mode efficiency tests
- Sleep mode consolidation metrics
- Full cycle performance over time

## Speculative but Inspiring Concepts (Future Work)

### 1. Base-200 Atomic Concepts 🔮
**Idea**: Human cognition operates on ~200 fundamental concepts
- Could revolutionize AI architecture
- Needs extensive cognitive science research
- Keep as "future directions" in paper

### 2. Magic Circle Formation 🔮
**Idea**: Concept formation as drawing circles in vector space
- Beautiful metaphor for attention mechanisms
- Difficult to operationalize currently
- Mention in discussion as theoretical insight

### 3. Generative Grammar Shape Matching 🔮
**Idea**: Syntactic trees as graph shapes
- Profound connection to memory
- Requires new mathematical framework
- Save for follow-up research

## Implementation Priority

### Phase 1 (2 weeks) - Core Wake-Sleep
1. Implement WakeSleepManager class
2. Create mode switching logic
3. Implement cost minimization (wake)
4. Implement reward maximization (sleep)

### Phase 2 (1 week) - Donut Search
1. Implement donut_search function
2. Integrate with wake mode
3. Add parameter tuning for radii

### Phase 3 (2 weeks) - Maze Experiment
1. Create 2.5D maze environment
2. Implement obstacle-as-query system
3. Run comparative experiments

### Phase 4 (1 week) - Paper Writing
1. Document experimental results
2. Create visualizations
3. Write Section 4 (Wake-Sleep Framework)

## Success Metrics

### Quantitative
- Wake mode: 20% faster query resolution
- Sleep mode: 30% reduction in graph redundancy
- Donut search: 50% reduction in irrelevant processing

### Qualitative
- Clear separation of concerns (wake vs sleep)
- Intuitive biological analogies
- Unified framework across domains

## Code Structure

```
experiments/
├── wake_sleep_poc/
│   ├── src/
│   │   ├── wake_sleep_manager.py
│   │   ├── donut_search.py
│   │   └── dual_optimizer.py
│   ├── tests/
│   │   ├── test_wake_mode.py
│   │   └── test_sleep_mode.py
│   └── results/
│       └── cycle_metrics.json
│
└── maze_queries_poc/
    ├── src/
    │   ├── maze_environment.py
    │   ├── obstacle_query_converter.py
    │   └── unified_navigator.py
    └── results/
        └── navigation_performance.json
```

## Paper Impact

### What to Emphasize
1. **Biological Plausibility**: Wake-sleep cycles are fundamental to cognition
2. **Mathematical Elegance**: Same function, different optimization directions
3. **Practical Benefits**: Improved efficiency and quality
4. **Unified Framework**: Language and space use same principles

### What to Mention Briefly
1. **Base-200**: As a theoretical insight in discussion
2. **Magic Circles**: As an intuitive explanation of attention
3. **Future Directions**: Rich research opportunities

## Timeline

- Week 1-2: Core implementation
- Week 3: Donut search integration
- Week 4-5: Maze experiments
- Week 6: Paper writing and results

---
*This PoC plan focuses on implementable features that will strengthen the paper while keeping speculative ideas as future research directions.*