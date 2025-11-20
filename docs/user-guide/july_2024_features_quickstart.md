# July 2024 Features Quick Start Guide

This guide helps you quickly start using the new features added to InsightSpike-AI in July 2024.

## 🚀 Quick Feature Overview

1. **Layer1 Bypass** - 10x faster responses for known facts
2. **Insight Auto-Registration** - Automatically saves and reuses discoveries
3. **Mode-Aware Prompts** - Optimized prompts for each LLM model
4. **Graph-Based Search** - Find related concepts through connections

## 🎯 Using Layer1 Bypass

### For Production Systems
```python
from insightspike.config import load_config
from insightspike.implementations.agents import MainAgent

# Use production preset with bypass enabled
config = load_config(preset="production_optimized")
agent = MainAgent(config)
agent.initialize()

# Fast queries bypass full pipeline
result = agent.process_question("What is the capital of France?")
# ~100ms instead of ~1s
```

### Custom Configuration
```python
config = load_config()
config.processing.enable_layer1_bypass = True
config.processing.bypass_uncertainty_threshold = 0.3  # More aggressive
```

## 💡 Using Insight Auto-Registration

### Basic Usage (Enabled by Default)
```python
# Insights are automatically captured
agent.add_knowledge("Water freezes at 0°C")
agent.add_knowledge("Ice is less dense than water")

result = agent.process_question("Why does ice float?")
if result.has_spike:
    print("New insight discovered and saved!")

# View captured insights
insights = agent.get_insights(limit=10)
for insight in insights['insights']:
    print(f"- {insight['content']}")
```

### Search Previous Insights
```python
# Find insights about a concept
water_insights = agent.search_insights("water", limit=5)
for insight in water_insights:
    print(f"- {insight['content']} (quality: {insight['quality_score']})")
```

## 📝 Using Mode-Aware Prompts

### Automatic Mode Selection
```python
# Small model - minimal prompt
config.llm.model = "distilgpt-2"
agent = MainAgent(config)
# Automatically uses minimal mode (2 docs max)

# Large model - detailed prompt  
config.llm.model = "gpt-4"
agent = MainAgent(config)
# Automatically uses detailed mode (10 docs max)
```

### Manual Mode Override
```python
config.llm.prompt_style = "minimal"  # Force minimal mode
config.llm.max_context_docs = 3      # Custom limit
```

## 🕸️ Using Graph-Based Search

### Enable Graph Search
```python
# Use graph-enhanced preset
config = load_config(preset="graph_enhanced")

# Or enable manually
config.graph.enable_graph_search = True
config.graph.hop_limit = 2  # Search 2 hops away
```

### Example: Finding Connections
```python
# Build connected knowledge
agent.add_knowledge("Einstein developed relativity")
agent.add_knowledge("Relativity explains gravity as curved spacetime")
agent.add_knowledge("GPS satellites account for gravitational time dilation")

# Graph search finds indirect connections
result = agent.process_question("How does Einstein relate to GPS?")
# Finds: Einstein → relativity → time dilation → GPS
```

## 🎛️ Configuration Presets

### Available Presets
```python
# Minimal - Fast, no extras
config = load_config(preset="minimal")

# Experiment - Balanced for testing
config = load_config(preset="experiment")

# Production - Optimized with bypass
config = load_config(preset="production_optimized")

# Graph Enhanced - Full graph features
config = load_config(preset="graph_enhanced")

# Research - All features, detailed mode
config = load_config(preset="research")
```

## 🔧 Common Configurations

### Fast FAQ System
```python
config = load_config(preset="production_optimized")
config.processing.bypass_uncertainty_threshold = 0.3
config.processing.enable_insight_search = False  # Skip insight search
```

### Research System
```python
config = load_config(preset="graph_enhanced")
config.processing.enable_insight_registration = True
config.processing.max_insights_per_query = 10
config.graph.hop_limit = 3  # Deeper graph search
```

### Minimal Testing
```python
config = load_config(preset="minimal")
config.llm.provider = "mock"  # Use mock LLM
```

## 📊 Monitoring Performance

### Check Bypass Usage
```python
stats = agent.get_stats()
print(f"Total queries: {stats['reasoning_cycles']}")
# TODO: Add bypass counter to stats
```

### Monitor Insights
```python
insights = agent.get_insights()
print(f"Total insights: {len(insights['insights'])}")
print(f"Categories: {insights['categories']}")
```

## 🐛 Troubleshooting

### Bypass Not Working?
```bash
# Enable verbose mode to see decisions
result = agent.process_question("Simple query", verbose=True)
# Look for: "Layer1 bypass activated - low uncertainty query"
```

### Insights Not Saving?
```python
# Check configuration
print(f"Registration enabled: {config.processing.enable_insight_registration}")
print(f"Spike thresholds: GED={config.graph.spike_ged_threshold}, IG={config.graph.spike_ig_threshold}")
```

### Graph Search Too Slow?
```python
# Reduce hop limit
config.graph.hop_limit = 1  # Only immediate neighbors

# Increase similarity threshold
config.graph.neighbor_threshold = 0.6  # Fewer edges
```

## 🚀 Next Steps

1. Run performance benchmarks with your data
2. Experiment with different presets
3. Monitor insight quality and adjust thresholds
4. Explore archived demos under `archive/examples_legacy/` if you need older PoC scripts
5. Share your results and feedback!
