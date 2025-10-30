---
status: active
category: meta
owner: refactor-wg
created: 2025-08-25
last_review: 2025-08-25
---

# Wake Mode Query Processing Refactoring

## Overview
Major refactoring to implement proper wake mode (覚醒期) query processing with **cost minimization** instead of reward maximization.

## Correct Process Flow

### 1. Query Input & Vectorization
```python
query_vector = sentence_transformer.encode(query)
```

### 2. Neighbor Collection (Query-centric sphere search)
- Collect nodes within **radius r from query point** in vector space
- Query becomes the origin of local coordinate system
- Search within spherical neighborhood: ||node - query|| < radius
- This is mathematically equivalent to distance-based search but conceptually cleaner

### 3. Combination Search (Cost Minimization)
- From n candidates, select top-k nodes
- Find combination that **minimizes geDIG cost**
- NOT maximizing reward - this is wake mode!

### 4. Message Passing & Intermediate Node Formation
- Apply message passing on selected k nodes
- Form intermediate node at centroid or optimal position

### 5. Context Preparation
- Calculate cosine similarities between intermediate node and related episodes
- These similarities are for LLM context only, NOT for geDIG calculation

### 6. LLM Generation
- Create prompt with query, related episodes, and similarities
- Generate response

## Key Insights

### Role Separation
- **Wake Mode**: Cost minimization (efficient knowledge access)
- **Sleep Mode**: Reward maximization (quality improvement)

### Cosine Similarity Usage
- **NOT needed**: geDIG calculation
- **Needed**: LLM prompt context

### Current Implementation Issues
1. Using reward maximization in wake mode (wrong!)
2. Mixing cosine similarity into geDIG calculation (unnecessary!)
3. Threshold is for reward, not for distance/cost

## Proposed Configuration

```yaml
query_processing:
  mode: "wake"
  
  neighbor_collection:
    method: "query_centric_sphere"
    radius: 0.8  # Sphere radius from query point
    max_neighbors: 20  # n
    metric: "euclidean"  # ||node - query||
    
  optimization:
    objective: "minimize_cost"  # Critical change!
    search_method: "beam_search"  # or "exhaustive" for small n
    top_k: 5
    
  gedig_cost:
    node_cost: 1.0
    edge_cost: 0.5
    # No lambda/mu - those are for sleep mode reward
    
  prompt_generation:
    include_similarities: true  # For context only
    max_episodes: 10
```

## Implementation Details

### Query-Centric Sphere Search
```python
def find_neighbors_in_sphere(query_vec, node_vectors, radius):
    """
    Find all nodes within radius r from query point.
    Query is treated as the origin of local coordinate system.
    """
    neighbors = []
    for node_id, node_vec in node_vectors.items():
        # Query-relative position
        relative_vec = node_vec - query_vec
        distance = np.linalg.norm(relative_vec)
        
        if distance < radius:
            neighbors.append({
                'node_id': node_id,
                'distance': distance,
                'relative_position': relative_vec
            })
    
    return sorted(neighbors, key=lambda x: x['distance'])
```

### Optimization for Large-Scale
```python
# Use FAISS for efficient sphere search
import faiss

class EfficientSphereSearch:
    def __init__(self, vectors):
        self.index = faiss.IndexFlatL2(vectors.shape[1])
        self.index.add(vectors)
    
    def search_sphere(self, query_vec, radius, max_results=100):
        # Search for candidates
        distances, indices = self.index.search(
            query_vec.reshape(1, -1), 
            k=max_results
        )
        
        # Filter by radius (distances are squared in FAISS)
        mask = distances[0] < radius**2
        return indices[0][mask], np.sqrt(distances[0][mask])
```

## Implementation Steps

1. **Refactor query processor**
   - Implement query-centric sphere search
   - Separate geometric search from semantic similarity
   - Implement cost minimization search

2. **Update geDIG calculator**
   - Add cost-only mode (no reward calculation)
   - Remove cosine similarity dependency

3. **Create wake/sleep mode manager**
   - Clear separation of objectives
   - Mode-specific parameter sets

## Expected Benefits

1. **Cleaner theory**: No more confusion about when to maximize vs minimize
2. **Better performance**: Correct objective for each phase
3. **Simpler code**: Each component has single responsibility
4. **Biological plausibility**: Matches human wake/sleep cycles

## Testing Strategy

1. **Unit tests**
   - Distance-based collection
   - Cost minimization search
   - Mode switching

2. **Integration tests**
   - Full query processing pipeline
   - Wake mode with various queries
   - Performance comparison with old approach

3. **Validation**
   - Measure actual costs of selected combinations
   - Verify no reward calculation in wake mode
   - Check LLM response quality

## Key Concept: Query-Centric Geometry

The query point becomes the origin of a local coordinate system:
- All distances are measured from the query point
- Creates a spherical neighborhood for search
- Intuitive: "What's within radius r of my question?"
- Mathematically equivalent to standard distance search but conceptually cleaner

This aligns perfectly with InsightSpike's spatial understanding of knowledge organization.

## Migration Notes

- Current experiments can be rerun with new configuration
- No need to change the normalized geDIG implementation
- Only the usage pattern changes (cost vs reward)
- Replace "norm distance" with "query-centric sphere radius"

---
*This refactoring represents a fundamental improvement in InsightSpike's architecture, aligning implementation with theoretical understanding of wake/sleep cycles and spatial knowledge organization.*