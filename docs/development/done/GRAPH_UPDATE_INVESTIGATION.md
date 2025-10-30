---
status: active
category: gedig
owner: refactor-wg
created: 2025-08-25
last_review: 2025-08-25
---

# Investigation: Why graph_pyg.pt is not being updated

## Summary of Findings

After thorough investigation, I've identified why `graph_pyg.pt` is not being updated during experiments:

### 1. **The graph IS being updated correctly in the main codebase**
- The `MainAgent` class properly updates the graph through `L3GraphReasoner`
- When `save_state()` is called, the graph is correctly saved to `graph_pyg.pt`
- This was confirmed through the debug script which showed the graph file being updated

### 2. **The experiments are NOT using the official MainAgent**
- The experiments in `gedig_embedding_evaluation` are using custom implementations
- For example, `insightspike_with_real_graph_embedding.py` has its own `RealInsightSpike` class
- These custom implementations have their own `add_episode` methods that don't interact with the official InsightSpike-AI components

### 3. **Key differences in the experiment implementations**
- Custom experiments maintain their own graph structures (e.g., `self.graph_memory`, `self.knowledge_graph`)
- They don't use `L2MemoryManager` or `L3GraphReasoner` from the main codebase
- They don't call `save_state()` to persist the graph to disk

## How the Official System Works

1. **Adding episodes through MainAgent**:
   ```python
   agent = MainAgent()
   agent.initialize()
   
   # Add episode with graph update
   result = agent.add_episode_with_graph_update(text, c_value)
   
   # Or process a question (which also updates the graph)
   result = agent.process_question(question)
   
   # Save state to persist graph
   agent.save_state()
   ```

2. **Graph update flow**:
   - `MainAgent.add_episode_with_graph_update()` → adds to L2 memory
   - Calls `L3GraphReasoner.analyze_documents()` → updates internal graph
   - `MainAgent.save_state()` → calls `L3GraphReasoner.save_graph()` → saves to `graph_pyg.pt`

## Solution for Experiments

To ensure graph updates in experiments, they should either:

### Option 1: Use the official MainAgent
```python
from insightspike.core.agents.main_agent import MainAgent

# Initialize agent
agent = MainAgent()
agent.initialize()

# Add episodes
for question, document in dataset:
    agent.add_episode_with_graph_update(f"Q: {question}\nA: {document}", 0.8)

# Save state to persist graph
agent.save_state()
```

### Option 2: Manually save graphs in custom implementations
If using custom implementations, add graph saving:
```python
def save_graph_state(self):
    """Save the current graph state"""
    import torch
    from pathlib import Path
    
    # Convert your graph to PyTorch Geometric format
    graph_data = self.convert_to_pyg_format()
    
    # Save to standard location
    graph_path = Path("data/graph_pyg.pt")
    torch.save(graph_data, graph_path)
```

## Verification

The debug script (`debug_graph_update.py`) demonstrates the correct behavior:
- Episodes are added to memory (episodes.json updated)
- Graph is updated in L3GraphReasoner (nodes and edges tracked)
- When `save_state()` is called, graph_pyg.pt is updated with new data

## Recommendations

1. **For production use**: Always use the official `MainAgent` API
2. **For experiments**: Either use `MainAgent` or ensure custom implementations save graph state
3. **For monitoring**: Check both file modification times and file sizes to verify updates
4. **For debugging**: Use the provided debug script to verify the update mechanism