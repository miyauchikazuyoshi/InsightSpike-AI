# InsightSpike CLI User Guide

## Overview

The improved InsightSpike CLI (`spike`) provides a user-friendly interface for:
- Asking questions and discovering insights
- Learning from documents
- Managing configurations
- Running experiments
- Interactive chat mode

## Installation

After installing InsightSpike, the `spike` command will be available:

```bash
poetry install
poetry shell
spike --help
```

## Basic Commands

### Ask Questions

```bash
# Basic query
spike query "What is the relationship between quantum computing and AI?"

# Using alias
spike q "How does entropy relate to learning?"

# With specific configuration preset
spike query "Complex question" --preset experiment

# Verbose output
spike query "Question" --verbose
# or
spike query "Question" -v

# Legacy compatibility
spike ask "Question"  # Still works, redirects to query
```

### Embed Documents

```bash
# Embed a single file
spike embed data/knowledge.txt

# Embed all documents in a directory
spike embed data/documents/

# Using alias
spike e data/knowledge.txt

# Legacy aliases
spike learn data/knowledge.txt  # Still works, redirects to embed
spike l data/knowledge.txt      # Still works, redirects to embed
```

### Interactive Chat Mode

```bash
# Start chat
spike chat
# or
spike c

# Commands in chat mode:
# help   - Show available commands
# stats  - Show agent statistics  
# config - Show current configuration
# clear  - Clear conversation history
# exit   - Exit chat mode
```

## Configuration Management

### View Configuration

```bash
spike config show
```

### Change Settings

```bash
# Set individual values
spike config set safe_mode false
spike config set max_tokens 512
spike config set debug true
spike config set spike_sensitivity 1.5
```

### Use Presets

```bash
# Development (fast, safe mode)
spike config preset development

# Experiment (real LLM, moderate settings)
spike config preset experiment  

# Production (optimized performance)
spike config preset production

# Testing (isolated paths)
spike config preset testing

# Cloud (API-based LLMs)
spike config preset cloud
```

### Save and Load Configurations

```bash
# Save current configuration
spike config save my_config.json

# Load saved configuration
spike config load my_config.json
```

## Environment Variables

Override any setting using environment variables:

```bash
export INSIGHTSPIKE_SAFE_MODE=false
export INSIGHTSPIKE_MAX_TOKENS=1024
export INSIGHTSPIKE_DEBUG=true
export INSIGHTSPIKE_SPIKE_SENSITIVITY=2.0

spike ask "Question using env config"
```

## Running Experiments

```bash
# Simple spike detection experiment
spike experiment --name simple --episodes 10

# Insight synthesis experiment
spike experiment --name insight --episodes 5

# Mathematical foundations experiment
spike experiment --name math --episodes 7
```

## Other Commands

### Show Statistics

```bash
spike stats
```

### Version Information

```bash
spike version
```

## Configuration Options

| Setting | Default | Description |
|---------|---------|-------------|
| `mode` | `"cpu"` | Execution mode: cpu, gpu, mps |
| `safe_mode` | `true` | Use mock LLM (no model loading) |
| `debug` | `false` | Enable debug output |
| `max_tokens` | `256` | Maximum response tokens |
| `temperature` | `0.3` | LLM temperature |
| `spike_sensitivity` | `1.0` | Spike detection sensitivity multiplier |

## Tips

1. **Start with presets**: Use `spike config preset experiment` for real LLM testing
2. **Use chat mode**: Great for exploratory conversations
3. **Adjust sensitivity**: Increase `spike_sensitivity` to detect more insights
4. **Save good configs**: Use `spike config save` to preserve working configurations
5. **Use aliases**: `q` for query, `c` for chat, `e` for embed, `l` also works for embed

## Examples

### Quick Start

```bash
# 1. Set up for experiments
spike config preset experiment

# 2. Learn from documents
spike learn data/knowledge/

# 3. Ask questions
spike q "What connects quantum mechanics and information theory?"

# 4. Start interactive exploration
spike chat
```

### Research Workflow

```bash
# 1. Load research papers
spike learn research_papers/

# 2. Use production settings
spike config preset production

# 3. Ask synthesis questions
spike ask "How do these papers relate to each other?" -v

# 4. Check insights detected
spike stats
```

### Development Workflow

```bash
# 1. Use safe mode for testing
spike config preset development

# 2. Enable debug output
spike config set debug true

# 3. Run simple experiment
spike experiment --name simple --episodes 5

# 4. Save working configuration
spike config save dev_config.json
```

## Troubleshooting

### Command not found

Make sure you're in the Poetry shell:
```bash
poetry shell
```

### Model not loading

Use safe mode for testing without models:
```bash
spike config set safe_mode true
```

### GPU issues

Force CPU mode:
```bash
spike config set mode cpu
```

### Configuration issues

Reset to defaults:
```bash
spike config preset development
```