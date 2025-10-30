# Donut-Shaped Query Processing and Known Information Ignorance

## Core Insight: Query Processing as a Donut

The query processing region forms a **donut shape** in vector space:
- **Inner hole**: Known information (ignored)
- **Donut region**: Optimal processing zone
- **Outer space**: Irrelevant information (ignored)

## Mathematical Formulation

```python
def donut_search(query_vec, inner_radius, outer_radius):
    """
    Process only nodes in the donut-shaped region.
    
    Args:
        query_vec: Query point (origin of local coordinates)
        inner_radius: Below this = too similar (known)
        outer_radius: Above this = too different (irrelevant)
    """
    relevant_nodes = []
    
    for node in all_nodes:
        distance = np.linalg.norm(query_vec - node.vector)
        
        if inner_radius < distance < outer_radius:
            relevant_nodes.append({
                'node': node,
                'distance': distance,
                'novelty_score': (distance - inner_radius) / (outer_radius - inner_radius)
            })
    
    return relevant_nodes
```

## Visual Representation

```
      Too far (ignored)
    · · · · · · · ·
  · · · · · · · · · ·
 · · ┌─────────┐ · · ·
· · │ ╱─────╲ │ · · ·   Outer: Irrelevant
· · │╱ ┌───┐ ╲│ · · ·
· · │ │ Q☆ │ │ · · ·   Donut: Processing zone
· · │╲ └───┘ ╱│ · · ·
· · │ ╲─────╱ │ · · ·   Inner: Known (ignored)
 · · └─────────┘ · · ·
  · · · · · · · · · ·
    · · · · · · · ·
```

## Why This Works

### 1. **Inner Circle (Known Information)**
- Distance < inner_radius
- Already strongly connected in graph
- No learning value
- Waste of computational resources
- Biological analogy: Repetition suppression

### 2. **Donut Region (Sweet Spot)**
- inner_radius < distance < outer_radius
- Related but novel
- Maximum potential for Aha! moments
- Optimal balance of familiarity and surprise
- Biological analogy: Zone of Proximal Development

### 3. **Outer Space (Irrelevant)**
- Distance > outer_radius
- Too different to be meaningful
- Would add noise to processing
- No reasonable connections possible
- Biological analogy: Attention filter

## Integration with Wake-Sleep Cycles

### Wake Phase Processing
```python
def wake_mode_with_donut_search(query):
    # 1. Convert query to vector
    query_vec = encode(query)
    
    # 2. Donut search for candidates
    candidates = donut_search(
        query_vec,
        inner_radius=0.2,  # Known threshold
        outer_radius=0.8   # Relevance threshold
    )
    
    # 3. Process only donut region
    if not candidates:
        return "Query too familiar or too alien"
    
    # 4. Find minimum cost solution in donut
    best_combination = minimize_gedig_objective(candidates)
    
    # 5. Potential for Aha! (unexpected connections)
    check_for_shortcuts(best_combination)
```

### Sleep Phase Benefits
```python
# Sleep consolidation affects donut parameters
# - Strong edges → easier known detection (inner circle)
# - Pruned noise → cleaner outer boundary
# - Optimized paths → better donut navigation
```

## Implementation Parameters

```python
class DonutSearchConfig:
    # Strict known detection
    conservative = {
        'inner_radius': 0.1,  # Very similar ignored
        'outer_radius': 0.6,  # Narrow search
        'donut_width': 0.5
    }
    
    # Balanced approach
    balanced = {
        'inner_radius': 0.2,  # Moderate filtering
        'outer_radius': 0.8,  # Reasonable range
        'donut_width': 0.6
    }
    
    # Creative exploration
    creative = {
        'inner_radius': 0.3,  # Less filtering
        'outer_radius': 0.95, # Wide search
        'donut_width': 0.65
    }
```

## Advantages

### 1. **Computational Efficiency**
- Skip redundant processing (inner circle)
- Avoid noise (outer space)
- Focus resources on high-value region

### 2. **Learning Efficiency**
- Don't relearn known facts
- Don't waste time on irrelevant tangents
- Maximize discovery potential

### 3. **Biological Plausibility**
- Matches attention mechanisms
- Implements habituation
- Optimizes energy usage

## Connection to Refactoring Plan

This donut search mechanism directly implements the query-centric sphere search described in the [Wake Mode Refactoring document](/docs/development/wake_mode_refactoring.md), with the crucial addition of the inner radius for known information filtering.

### Updated Implementation
```python
# From wake_mode_refactoring.md, enhanced:
def find_neighbors_in_sphere(query_vec, node_vectors, inner_radius, outer_radius):
    """
    Enhanced sphere search with donut shape.
    Implements both novelty detection and relevance filtering.
    """
    neighbors = []
    for node_id, node_vec in node_vectors.items():
        distance = np.linalg.norm(node_vec - query_vec)
        
        # Donut filter
        if inner_radius < distance < outer_radius:
            neighbors.append({
                'node_id': node_id,
                'distance': distance,
                'in_donut': True,
                'novelty': (distance - inner_radius) / (outer_radius - inner_radius)
            })
    
    return sorted(neighbors, key=lambda x: x['distance'])
```

## Connection to Original Layer1 Design

### Layer1's Original Intent
The donut search realizes what Layer1 originally aimed to achieve - intelligent filtering based on relevance and novelty. See [Layer1 Implementation Plan](/docs/development/layer1_implementation_plan.md) for the original design.

### Old Layer1 Approach
```python
# Sequential filtering with unclear parameters
def layer1_filter(query, candidates):
    # Step 1: Relevance filter
    relevant = filter_by_relevance(candidates, threshold=0.7)  # What does 0.7 mean?
    
    # Step 2: Novelty filter  
    novel = filter_by_novelty(relevant, threshold=0.3)  # How to tune?
    
    # Step 3: Value assessment
    valuable = assess_processing_value(novel)  # Complex logic
    
    return valuable
```

### New Donut Implementation
```python
# Single geometric operation with clear parameters
def modern_layer1(query):
    return donut_search(
        query_vec=encode(query),
        inner_radius=0.2,  # Known information boundary (clear!)
        outer_radius=0.8   # Relevance boundary (intuitive!)
    )
```

### Why Donut Search Succeeds Where Layer1 Struggled

| Aspect | Old Layer1 | Donut Search |
|--------|-----------|--------------|
| **Clarity** | Multiple abstract filters | Single geometric concept |
| **Parameters** | Unclear thresholds | Physical distances |
| **Computation** | Sequential, redundant | Single pass |
| **Tunability** | Trial and error | Intuitive radii |
| **Theory** | Ad-hoc filtering | Principled geometry |

### Evolution Path
1. **Layer1 Concept** (2024): Filter by relevance and novelty
2. **Wake-Sleep Understanding**: Need for query-centric processing  
3. **Donut Discovery**: Geometric unification of Layer1's goals
4. **Final Integration**: Layer1's dream realized through spatial reasoning

## Key Principle

**"Process only what is neither too familiar nor too foreign - the donut of optimal learning."**

**"Layer1's filtering dream is now realized through elegant geometry."**

---
*This donut-shaped processing region maximizes both computational efficiency and discovery potential, implementing a biologically plausible attention mechanism while fulfilling Layer1's original vision.*