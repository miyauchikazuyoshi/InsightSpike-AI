# Feature Demonstrations

This directory contains demonstration scripts for InsightSpike-AI's advanced features.

## July 2024 Features

### Core Features

- **`demo_adaptive_learning.py`** - Shows how the system learns from patterns and optimizes strategies
- **`demo_graph_memory_search.py`** - Demonstrates 2-hop graph traversal for associative memory
- **`demo_insight_registration.py`** - Auto-registration and reuse of discovered insights
- **`demo_mode_aware_prompts.py`** - Mode-aware prompt building for different LLM sizes

### Configuration & Integration

- **`demo_insight_config.py`** - Configuration options for insight features
- **`demo_prompt_modes.py`** - Different prompt modes (detailed, standard, minimal)
- **`demo_insight_prompt_building.py`** - How insights are included in prompts

## Running Demos

All demos can be run directly:

```bash
# Run a specific demo
python demo_adaptive_learning.py

# Run all demos
for demo in demo_*.py; do
    echo "Running $demo..."
    python $demo
    echo "---"
done
```

## Key Concepts Demonstrated

1. **Pattern Learning**: The system remembers successful strategies
2. **Graph Traversal**: Finding related concepts through knowledge graph
3. **Insight Capture**: Automatic discovery and reuse of insights
4. **Adaptive Prompts**: Optimal context for each LLM type

These demos show practical usage patterns and help understand the system's capabilities.