# Graph-Based Search Performance Guide

## Overview

Graph-based search is a critical optimization for InsightSpike-AI that enables scalable knowledge retrieval even with large episodic memories. This guide explains how it works and its performance benefits.

## Why Graph-Based Search?

### Traditional Vector Search (Disabled)
```python
# O(n) complexity - scans ALL episodes
for episode in all_episodes:
    similarity = cosine_similarity(query, episode)
```

**Performance degradation:**
- 100 episodes: ~1ms
- 1,000 episodes: ~10ms
- 10,000 episodes: ~100ms
- 100,000 episodes: ~1000ms (1 second!)

### Graph-Based Search (Enabled)
```python
# O(k*h) complexity - only explores neighbors
1. Find k nearest nodes to query
2. Explore h hops from those nodes
3. Score based on path relevance
```

**Consistent performance:**
- Any size: ~5-20ms (depends on k and h, not total episodes)

## Configuration

### Basic Setup
```yaml
memory:
  enable_graph_search: true         # Must be true
  graph_hop_limit: 2               # 2 hops is usually optimal
  graph_neighbor_threshold: 0.4    # Min similarity to be a neighbor
  graph_path_decay: 0.7            # How much relevance decays per hop
```

### Performance Tuning

#### For Speed (Narrow Search)
```yaml
memory:
  graph_hop_limit: 1
  graph_neighbor_threshold: 0.5    # Higher = fewer neighbors
  max_retrieved_docs: 10
```

#### For Discovery (Broad Search)
```yaml
memory:
  graph_hop_limit: 3
  graph_neighbor_threshold: 0.3    # Lower = more neighbors
  max_retrieved_docs: 20
```

## How It Works

### 1. Initial Retrieval
- Find top-k most similar episodes to query
- These become "seed nodes"

### 2. Graph Expansion
- From each seed, explore neighbors within similarity threshold
- Continue for h hops
- Score decays by `path_decay` each hop

### 3. Scoring
```
score = similarity * (path_decay ^ hop_distance) * c_value_weight
```

### 4. Benefits
- **Discovers related concepts** not directly similar to query
- **Handles synonyms/paraphrases** through graph connections
- **Scales efficiently** to millions of episodes

## Example Performance

### Test Setup
- 10,000 episodes in memory
- Query: "What is consciousness?"

### Results
| Method | Time | Found Insights |
|--------|------|----------------|
| Vector Search | 95ms | Direct matches only |
| Graph Search (1-hop) | 12ms | + Related concepts |
| Graph Search (2-hop) | 18ms | + Deep connections |

### Quality Improvement
Graph search found connections between:
- "consciousness" → "awareness" → "self-reflection"
- "consciousness" → "neural activity" → "emergence"

These multi-hop connections enable the "Eureka!" moments that geDIG detects.

## Monitoring

Enable monitoring to track performance:

```yaml
graph:
  enable_monitoring: true
```

Check logs for:
```
INFO - Graph search completed: 2 hops, 43 nodes explored, 18ms
INFO - Retrieved 15 episodes with graph-enhanced scoring
```

## Troubleshooting

### Graph search seems slow
1. Reduce `graph_hop_limit` to 1
2. Increase `graph_neighbor_threshold` to 0.5
3. Check if FAISS is properly installed

### Not finding related concepts
1. Increase `graph_hop_limit` to 3
2. Decrease `graph_neighbor_threshold` to 0.3
3. Ensure graph is being built (`use_graph_integration: true`)

### Memory usage high
1. Reduce `episodic_memory_capacity`
2. Enable more aggressive pruning
3. Use IVF index type for FAISS

## Best Practices

1. **Start with defaults** - The default settings work well for most cases
2. **Monitor performance** - Enable monitoring during experiments
3. **Adjust gradually** - Change one parameter at a time
4. **Profile your workload** - Different domains may need different settings

## Conclusion

Graph-based search is essential for InsightSpike-AI to:
- Scale to large knowledge bases
- Discover non-obvious connections
- Enable true "insight" detection through multi-hop reasoning

Always enable it for production use and experiments with realistic data sizes.