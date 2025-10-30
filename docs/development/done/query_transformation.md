---
status: active
category: meta
owner: refactor-wg
created: 2025-08-25
last_review: 2025-08-25
---

# Query Transformation in InsightSpike

## Overview

Query Transformation is a revolutionary feature in InsightSpike that mimics human thinking processes. Instead of simple retrieval, queries evolve through knowledge graphs via message passing, discovering insights through conceptual connections.

## Visual Examples

### 1. Query Transformation Stages

![Query Transformation Stages](images/query_transformation/query_transformation_stages.png)

The query evolves through four stages:
- **Initial (Yellow)**: Query is placed on the knowledge graph
- **Exploring (Orange)**: Finding connections to key concepts
- **Transforming (Dark Orange)**: Absorbing concepts and forming connections
- **Insight (Green)**: Deep understanding achieved through multi-hop connections

### 2. Transformation Metrics

![Query Transformation Metrics](images/query_transformation/query_transformation_metrics.png)

Track the evolution of:
- **Confidence**: Sigmoid growth as understanding deepens
- **Transformation Magnitude**: Cumulative change in query embedding
- **Insights Discovery**: Key moments of understanding

### 3. Query Evolution in Embedding Space

![Query Embedding Evolution](images/query_transformation/query_embedding_evolution.png)

Watch as the query travels through conceptual space:
- Starting from initial position
- Passing through concept regions (Thermodynamics, Information Theory)
- Absorbing key concepts along the path
- Reaching final insight

## Key Features

### Phase 1: Basic Query Transformation
- Query-as-node placement on knowledge graphs
- Color-coded evolution (yellow → orange → green)
- Concept absorption and insight tracking
- Transformation history visualization

### Phase 2: Advanced Features
- **Multi-hop Reasoning**: Discover connections across multiple steps
- **Adaptive Exploration**: Temperature-controlled search strategy
- **Query Branching**: Explore from multiple perspectives simultaneously
- **Insight Synthesis**: Combine discoveries across branches

## Usage Examples

### Basic Query Transformation

```python
from insightspike import MainAgentWithQueryTransform

agent = MainAgentWithQueryTransform(enable_query_transformation=True)
result = agent.process_question("How are thermodynamic and information entropy related?")

# Access transformation history
history = result["transformation_history"]
insights = result["query_evolution"]["insights_discovered"]
```

### Advanced Multi-hop Reasoning

```python
from insightspike import MainAgentAdvanced

agent = MainAgentAdvanced(
    enable_multi_hop=True,
    enable_branching=True,
    max_branches=3
)

result = agent.process_question(
    "How do quantum mechanics and consciousness relate through information theory?"
)

# Access multi-hop paths and branch insights
paths = result["synthesis"]["discovered_paths"]
branch_insights = result["synthesis"]["total_insights"]
```

## Demo Scripts

1. **Basic Transformation Demo**
   ```bash
   python scripts/demo_query_transformation.py
   ```

2. **Simple Demo**
   ```bash
   python scripts/demo_query_transformation_simple.py
   ```

3. **Advanced Features Demo**
   ```bash
   python scripts/demo_advanced_query_transformation.py
   ```

4. **Visualization Generation**
   ```bash
   python scripts/visualize_query_transformation_graph.py
   ```

## How It Works

1. **Query Placement**: The query is embedded and placed on the knowledge graph based on semantic similarity
2. **Message Passing**: Graph neural networks propagate information between the query and knowledge nodes
3. **Concept Absorption**: The query absorbs relevant concepts from strongly connected nodes
4. **Transformation**: The query embedding evolves based on absorbed information
5. **Insight Discovery**: When sufficient transformation occurs, insights emerge
6. **Branching** (Phase 2): Multiple exploration paths investigate different perspectives
7. **Synthesis** (Phase 2): Insights from branches are combined for deeper understanding

## Benefits

- **Human-like Thinking**: Mimics how humans explore and connect concepts
- **Emergent Insights**: Discovers non-obvious connections through graph exploration
- **Explainable Process**: Track exactly how conclusions were reached
- **Multi-perspective**: Explore questions from multiple angles simultaneously
- **Adaptive**: Learns successful exploration patterns over time

## Technical Details

- **Embedding Model**: Sentence-BERT (all-MiniLM-L6-v2)
- **Graph Neural Network**: 3-layer message passing with attention
- **Dimension**: 384-dimensional query embeddings
- **Convergence**: Confidence-based with early stopping
- **Branching Strategy**: Priority-based with adaptive temperature