# Graph-Based Memory Search Configuration Guide

## Overview

InsightSpike-AI supports graph-based multi-hop memory search, which enhances traditional vector similarity search by exploring related concepts through the knowledge graph structure.

## Configuration Options

### 1. Basic Enable/Disable

```python
# Using config presets
from insightspike.config import load_config

# Enable graph search with preset
config = load_config(preset="graph_enhanced")

# Or manually enable
config = load_config(preset="experiment")
config.memory.enable_graph_search = True
```

### 2. Detailed Configuration

```yaml
# config.yaml
memory:
  enable_graph_search: true         # Master switch
  graph_hop_limit: 2               # Maximum hops (1-3 recommended)
  graph_neighbor_threshold: 0.4    # Min similarity to explore neighbor (0.0-1.0)
  graph_path_decay: 0.7           # Score decay per hop (0.1-1.0)
```

### 3. Configuration Parameters Explained

#### `enable_graph_search`
- **Type**: boolean
- **Default**: false
- **Description**: Enables multi-hop graph traversal during memory search

#### `graph_hop_limit`
- **Type**: integer (1-3)
- **Default**: 2
- **Description**: Maximum number of hops from initial search results
- **Note**: Higher values increase search time exponentially

#### `graph_neighbor_threshold`
- **Type**: float (0.0-1.0)
- **Default**: 0.4
- **Description**: Minimum cosine similarity to include a neighbor node
- **Lower values**: More exploratory, finds distant relations
- **Higher values**: More conservative, only strong connections

#### `graph_path_decay`
- **Type**: float (0.1-1.0)
- **Default**: 0.7
- **Description**: Score multiplier applied per hop
- **Example**: With 0.7, a 2-hop neighbor with similarity 0.8 gets final score: 0.8 × 0.7² = 0.392

## How It Works

1. **Initial Search**: Standard vector similarity finds top candidates
2. **Graph Exploration**: From top 5 results, explore neighbors up to hop_limit
3. **Score Calculation**: 
   - Direct matches get 20% bonus
   - Multi-hop matches get path_decay penalty
   - Highly connected nodes get connectivity bonus
4. **Re-ranking**: All results re-ranked by combined score

## Example Usage

```python
from insightspike import InsightSpike

# Initialize with graph search
spike = InsightSpike(preset="graph_enhanced")

# Add interconnected knowledge
spike.add_knowledge("Neural networks process information through layers")
spike.add_knowledge("Information theory measures entropy and uncertainty")  
spike.add_knowledge("Entropy relates to disorder in thermodynamics")

# Query that benefits from multi-hop
response = spike.query("How do neural networks relate to disorder?")
# Finds connection: neural networks → information → entropy → disorder
```

## Performance Considerations

- **Memory Usage**: Graph structure requires additional memory (~2x episodes)
- **Search Time**: Multi-hop search is slower (2-hop ≈ 3x slower than direct)
- **Quality**: Better for conceptual questions, may add noise for specific lookups

## Recommended Settings

### For Conceptual Reasoning
```yaml
memory:
  enable_graph_search: true
  graph_hop_limit: 2
  graph_neighbor_threshold: 0.35
  graph_path_decay: 0.7
```

### For Precise Retrieval
```yaml
memory:
  enable_graph_search: true
  graph_hop_limit: 1
  graph_neighbor_threshold: 0.6
  graph_path_decay: 0.8
```

### For Exploration
```yaml
memory:
  enable_graph_search: true
  graph_hop_limit: 3
  graph_neighbor_threshold: 0.3
  graph_path_decay: 0.6
```

## Troubleshooting

### Graph search not working?

1. **Check graph builder is enabled**:
   ```python
   # L2MemoryManager needs use_graph_integration=True
   config.memory.use_graph_integration = True  # If using old config
   ```

2. **Ensure sufficient episodes**:
   - Need at least 10-20 episodes for meaningful graph
   
3. **Verify graph is built**:
   ```python
   # Check if graph exists
   if agent.current_graph is not None:
       print(f"Graph has {agent.current_graph.num_nodes} nodes")
   ```

### Performance issues?

- Reduce `graph_hop_limit` to 1
- Increase `graph_neighbor_threshold` to 0.5+
- Disable for large knowledge bases (>10k episodes)

## Integration with Other Features

Graph search works well with:
- **Insight Registration**: Insights become highly connected nodes
- **Layer1 Bypass**: Can skip graph search for known queries
- **Adaptive Learning**: Can learn optimal threshold values

## Future Enhancements

- Concept-based edges (not just similarity)
- Hierarchical graph structure
- Learned edge weights
- Query-specific hop limits