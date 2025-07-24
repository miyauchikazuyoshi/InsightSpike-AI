# Recent InsightSpike-AI Features (July 2024)

This document provides comprehensive documentation of the major features implemented in July 2024, including usage notes, specifications, and configuration details.

## Table of Contents
1. [Layer1 Bypass Mechanism](#layer1-bypass-mechanism)
2. [Insight Auto-Registration and Search](#insight-auto-registration-and-search)
3. [Mode-Aware Prompt Building](#mode-aware-prompt-building)
4. [Graph-Based Memory Search](#graph-based-memory-search)

---

## Layer1 Bypass Mechanism

### Overview
The Layer1 bypass mechanism enables fast-path processing for known concepts with low uncertainty, dramatically improving performance for production deployments with large knowledge bases.

### How It Works
1. **Query Analysis**: Layer1 analyzes query uncertainty and identifies known/unknown elements
2. **Bypass Decision**: If uncertainty < threshold AND known elements exist, skip to Layer4
3. **Direct Response**: Layer4 generates response using only known elements, avoiding graph construction
4. **Performance**: 10x speedup for known queries, 95% memory reduction

### Configuration
```python
from insightspike.config import load_config

# Enable bypass via preset
config = load_config(preset="production_optimized")

# Or configure manually
config.processing.enable_layer1_bypass = True
config.processing.bypass_uncertainty_threshold = 0.2  # Max uncertainty for bypass
config.processing.bypass_known_ratio_threshold = 0.9  # Min ratio of known elements
```

### Usage Example
```python
from insightspike.implementations.agents import MainAgent

# Create agent with bypass enabled
agent = MainAgent(config)
agent.initialize()

# Add knowledge base
agent.add_knowledge("The capital of France is Paris.")
agent.add_knowledge("Paris is known for the Eiffel Tower.")

# Simple query will bypass Layer2/3
result = agent.process_question("What is the capital of France?")
# Response generated in ~100ms instead of ~1s

# Complex query will use full pipeline
result = agent.process_question("How does Paris relate to European culture?")
# Full graph analysis performed
```

### When To Use
- **Production systems** with large knowledge bases (>10K episodes)
- **FAQ systems** where most queries have direct answers
- **Legal/medical knowledge** systems with standard lookups
- **Real-time applications** requiring fast response times

### Implementation Details
- **Location**: `src/insightspike/implementations/agents/main_agent.py`
- **Method**: `_execute_cycle()` checks bypass conditions after Layer1
- **Logging**: Bypass activation logged at INFO level
- **Metrics**: Tracks bypass usage in agent statistics

---

## Insight Auto-Registration and Search

### Overview
Automatically captures and stores insights discovered during spike detection, making them available for future queries. This enables knowledge accumulation and reuse of discovered patterns.

### How It Works
1. **Spike Detection**: Layer3 detects insight spike (ΔGED ≤ -0.5, ΔIG ≥ 0.2)
2. **Insight Extraction**: Response analyzed for insight patterns
3. **Quality Evaluation**: Multiple criteria assess insight value
4. **Storage**: High-quality insights stored in registry
5. **Future Retrieval**: Insights searched and included in memory retrieval

### Configuration
```python
# Enable insight features (default in most presets)
config.processing.enable_insight_registration = True
config.processing.enable_insight_search = True
config.processing.max_insights_per_query = 5

# Disable for minimal mode
config = load_config(preset="minimal")  # Insights disabled
```

### Usage Example
```python
# Discover an insight
agent.add_knowledge("Water freezes at 0°C")
agent.add_knowledge("Ice expands when it freezes")
result = agent.process_question("Why does ice float on water?")

if result.has_spike:
    print("Insight discovered!")
    # Insight auto-registered: "Ice floats because it's less dense than water"

# Later queries benefit from the insight
result2 = agent.process_question("What happens to density when water freezes?")
# Retrieved documents now include: "[INSIGHT] Ice floats because..."
```

### Insight Categories
- **Causal**: Cause-effect relationships ("because", "due to", "results in")
- **Structural**: Part-whole relationships ("consists of", "components")
- **Analogical**: Similarity patterns ("like", "similar to", "as if")
- **Synthetic**: Novel combinations ("therefore", "thus", "implies")

### Quality Criteria
1. **Text Quality**: Grammar, coherence, clarity
2. **Concept Richness**: Number of meaningful concepts
3. **Graph Metrics**: GED/IG improvements
4. **Novelty**: Uniqueness compared to existing knowledge

### Implementation Details
- **Registry**: `src/insightspike/database/insight_fact_registry.py`
- **Integration**: `MainAgent._execute_cycle()` handles registration
- **Search**: `MainAgent._search_memory()` includes insights
- **Storage**: SQLite database with full-text search

---

## Mode-Aware Prompt Building

### Overview
Dynamically adjusts prompt size and complexity based on the LLM provider's capabilities, preventing token limit issues and optimizing response quality.

### Prompt Modes

#### 1. **Minimal Mode** (DistilGPT2, TinyLlama)
- Max 2 documents + 1 insight
- Simple prompt format without metadata
- Focuses on essential information only
- Token limit: ~512

#### 2. **Standard Mode** (GPT-3.5, most models)
- Max 7 documents + 3 insights  
- Includes basic metadata
- Balanced information density
- Token limit: ~2048

#### 3. **Detailed Mode** (GPT-4, Claude)
- Max 10 documents + 5 insights
- Full metadata and graph analysis
- Rich context for complex reasoning
- Token limit: ~4096+

### Configuration
```python
# LLM config includes prompt settings
llm_config = {
    "provider": "openai",
    "model": "gpt-3.5-turbo",
    "prompt_style": "standard",  # auto-detected from model
    "max_context_docs": 5,
    "use_simple_prompt": False,
    "include_metadata": True
}

# Presets configure appropriate modes
config = load_config(preset="experiment")  # Uses minimal mode
config = load_config(preset="production")  # Uses standard mode
config = load_config(preset="research")    # Uses detailed mode
```

### Implementation
```python
# Layer4 automatically selects prompt builder
def _select_prompt_mode(self, model_name: str) -> str:
    """Auto-detect appropriate prompt mode based on model."""
    model_lower = model_name.lower()
    
    if any(small in model_lower for small in ["distilgpt2", "tinyllama", "small"]):
        return "minimal"
    elif any(large in model_lower for large in ["gpt-4", "claude", "large"]):
        return "detailed"
    else:
        return "standard"
```

### Usage Notes
- Mode is auto-detected but can be overridden
- Documents are truncated to fit token limits
- Most relevant documents prioritized
- Insights always included if available

---

## Graph-Based Memory Search

### Overview
Implements multi-hop graph traversal for associative memory retrieval, enabling the system to find conceptually related information through graph connections rather than just direct similarity.

### How It Works
1. **Initial Search**: Standard cosine similarity finds seed nodes
2. **Graph Traversal**: Explores neighbors up to N hops
3. **Path Scoring**: Relevance decays exponentially with distance
4. **Re-ranking**: Combines similarity and graph connectivity scores

### Configuration
```python
# Enable graph search
config.graph.enable_graph_search = True
config.graph.hop_limit = 2              # Max traversal depth
config.graph.neighbor_threshold = 0.4   # Min similarity for neighbors
config.graph.path_decay = 0.7          # Relevance decay per hop

# Use graph-enhanced preset
config = load_config(preset="graph_enhanced")
```

### Usage Example
```python
# Build connected knowledge
agent.add_knowledge("Einstein developed the theory of relativity")
agent.add_knowledge("Relativity changed our understanding of space and time")
agent.add_knowledge("GPS satellites must account for time dilation")

# Query triggers multi-hop search
result = agent.process_question("How does Einstein's work affect modern technology?")

# Retrieved documents include:
# 1. Direct: "Einstein developed..." (similarity: 0.8)
# 2. 1-hop: "Relativity changed..." (path score: 0.8 * 0.7 = 0.56)
# 3. 2-hop: "GPS satellites..." (path score: 0.56 * 0.7 = 0.39)
```

### Algorithm Details
```python
class GraphMemorySearch:
    def search_with_graph(self, query_embedding, top_k=5):
        # 1. Get initial candidates
        candidates = vector_search(query_embedding, k=top_k*2)
        
        # 2. Build local subgraph
        subgraph = extract_subgraph(candidates, hop_limit=self.hop_limit)
        
        # 3. Score all paths
        for node in subgraph:
            if node in candidates:
                node.score = similarity(query_embedding, node.embedding)
            else:
                # Path-based score
                best_path = shortest_path(candidates, node)
                decay = self.path_decay ** len(best_path)
                node.score = max_similarity_on_path * decay
        
        # 4. Re-rank and return top-k
        return sorted(subgraph, key=lambda n: n.score)[:top_k]
```

### Benefits
- **Associative Retrieval**: Finds related concepts not directly similar
- **Context Enrichment**: Provides broader context through neighborhoods  
- **Human-like Memory**: Mimics spreading activation in neural networks
- **Insight Discovery**: Graph paths can reveal hidden connections

### Performance Considerations
- **Overhead**: ~20-30% slower than direct search
- **Memory**: Requires loading graph structure
- **Scalability**: Use with <100K nodes for real-time
- **Optimization**: Prune graph edges below threshold

---

## Integration Example

Here's how all features work together:

```python
from insightspike.config import load_config
from insightspike.implementations.agents import MainAgent

# Configure with all features
config = load_config(preset="production_optimized")
config.graph.enable_graph_search = True
config.processing.enable_insight_registration = True

# Initialize agent
agent = MainAgent(config)
agent.initialize()

# Build knowledge base
knowledge = [
    "Machine learning uses neural networks",
    "Neural networks are inspired by the brain",
    "The brain has billions of neurons",
    "Neurons communicate through synapses"
]
for fact in knowledge:
    agent.add_knowledge(fact)

# Simple query - uses Layer1 bypass
simple = agent.process_question("What are neural networks inspired by?")
# Fast response: "Neural networks are inspired by the brain"

# Complex query - uses full pipeline with graph search
complex = agent.process_question("How do artificial and biological networks differ?")
# Graph search finds: neural networks → brain → neurons → synapses
# May discover insight about fundamental differences

# Check for insights
if complex.has_spike:
    insights = agent.get_insights()
    print(f"Discovered {len(insights['insights'])} new insights!")
```

## Best Practices

1. **Choose appropriate presets** for your use case
2. **Monitor performance metrics** with `agent.get_stats()`
3. **Save insights periodically** with `agent.save_state()`
4. **Configure thresholds** based on your domain
5. **Test with representative queries** before production

## Troubleshooting

### Layer1 Bypass Not Activating
- Check uncertainty threshold (default 0.2)
- Verify knowledge base has relevant facts
- Enable verbose logging to see decisions

### Insights Not Being Found
- Ensure spike detection thresholds are appropriate
- Check insight quality thresholds
- Verify registry database is accessible

### Graph Search Too Slow
- Reduce hop_limit (try 1 instead of 2)
- Increase neighbor_threshold to prune edges
- Consider using bypass for simple queries

### Prompt Size Issues
- Check model-specific token limits
- Adjust max_context_docs if needed
- Use appropriate prompt_style for model

## Future Enhancements

1. **Learning Loops**: Use insights to improve future responses
2. **Dynamic Thresholds**: Auto-adjust based on performance
3. **Graph Pruning**: Remove low-value edges automatically
4. **Insight Clustering**: Group related insights
5. **Prompt Optimization**: Learn optimal prompt formats per model