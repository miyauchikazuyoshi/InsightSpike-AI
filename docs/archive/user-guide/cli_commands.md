# InsightSpike CLI Commands

This document describes the unique CLI commands that showcase InsightSpike's core capabilities.

## Table of Contents

### Core Commands
1. [spike query - Ask Questions](#spike-query---ask-questions)
2. [spike embed - Add Knowledge](#spike-embed---add-knowledge)
3. [spike interactive - Interactive Chat](#spike-interactive---interactive-chat)
4. [spike insights - View Insights](#spike-insights---view-insights)

### Advanced Commands
5. [spike discover - Insight Discovery](#spike-discover---insight-discovery)
6. [spike experiment - Run Experiments](#spike-experiment---run-experiments)
7. [spike demo - Quick Demo](#spike-demo---quick-demo)

### Utility Commands
8. [spike config - Configuration Management](#spike-config---configuration-management)
9. [spike stats - View Statistics](#spike-stats---view-statistics)
10. [spike version - Version Information](#spike-version---version-information)

### Advanced Features
11. [spike bridge - Concept Bridging](#spike-bridge---concept-bridging)
12. [spike graph - Knowledge Graph Analytics](#spike-graph---knowledge-graph-analytics)

---

# Core Commands

---

# spike query - Ask Questions

## Overview

The `spike query` command is the main interface for asking questions and getting insight-powered responses from InsightSpike.

## Usage

```bash
# Basic question
spike query "How does entropy relate to information theory?"

# Legacy alias: 'ask' is still available (spike ask → query). Prefer 'query'.
spike query "What is the connection between consciousness and information?"

# With options
spike query "Complex question" --verbose --preset experiment
```

## Options

- `--preset, -p`: Config preset (development, experiment, production)
- `--verbose, -v`: Show detailed analysis
- `--temperature, -t`: LLM temperature (0.0-1.0)
- `--max-tokens`: Maximum response tokens

## Features

- 🧠 **Insight Detection**: Automatically detects when your question triggers an insight spike
- 📊 **Graph Analysis**: Shows knowledge graph metrics when insights are found
- 🔗 **Knowledge Integration**: Combines information from multiple sources

---

# spike embed - Add Knowledge

## Overview

The `spike embed` command adds documents to your knowledge base for future insight discovery.

## Usage

```bash
# Add a single file
spike embed document.txt

# Add a directory of files
spike embed docs/

# Legacy aliases: 'learn', 'l', 'e' are available (e.g., spike learn → embed). Prefer 'embed'.
spike embed papers/
spike embed research.md
spike embed data.txt
```

## Options

- `--preset, -p`: Config preset for processing
- `--recursive, -r`: Process directories recursively
- `--format, -f`: Specify file format (auto-detected by default)

## Supported Formats

- `.txt` - Plain text files
- `.md` - Markdown files
- Future: `.pdf` - PDF documents (with optional dependencies)
- `.json` - Structured data

---

# spike interactive - Interactive Chat

## Overview

The `spike interactive` command (alias: `spike chat`) provides an interactive conversation mode with persistent context.

## Usage

```bash
# Start interactive mode
spike interactive

# Legacy alias: 'chat' is available (spike chat → interactive). Prefer 'interactive'.
spike interactive

# With specific preset
spike chat --preset research
```

## Features

- 💬 **Persistent Context**: Maintains conversation history
- 🔄 **Multi-turn Reasoning**: Builds on previous insights
- 📝 **Session Management**: Save and load chat sessions
- 🎯 **Commands**: Special commands like `/save`, `/load`, `/clear`

## Interactive Commands

- `/save [filename]` - Save current session
- `/load [filename]` - Load previous session
- `/clear` - Clear conversation history
- `/help` - Show available commands
- `/exit` or `/quit` - Exit interactive mode

---

# spike insights - View Insights

## Overview

The `spike insights` command displays previously discovered insights from your knowledge base.

## Usage

```bash
# View all insights
spike insights

# Filter by date
spike insights --since "2024-01-01"

# Filter by confidence
spike insights --min-confidence 0.8

# Export insights
spike insights --export insights.json
```

## Options

- `--limit`: Maximum number of insights to show (default: 5)

---

# Advanced Commands

---

# spike discover - Insight Discovery

## Overview

The `spike discover` command analyzes your knowledge base to find hidden insights, unexpected connections, and emergent patterns using InsightSpike's unique geDIG (Graph Edit Distance + Information Gain) algorithm.

## Features

- 🔍 **Automatic Insight Detection**: Finds non-obvious connections between concepts
- ⚡ **Spike Detection**: Identifies high-value insights based on graph metrics
- 🌉 **Bridge Concept Discovery**: Finds concepts that connect different domains
- 📊 **Pattern Recognition**: Detects emergent patterns in your knowledge base

## Usage

```bash
# Basic usage - analyze current knowledge base
spike discover

# Analyze a specific corpus
spike discover --corpus papers/

# Set minimum spike threshold
spike discover --min-spike 0.8

# Export results to JSON
spike discover --export insights.json

# Filter by insight categories
spike discover --categories "causal,structural"
```

## Options

- `--corpus, -c PATH`: Path to document corpus to analyze
- `--min-spike, -s FLOAT`: Minimum spike threshold (0-1, default: 0.7)
- `--max-insights, -m INTEGER`: Maximum insights to display (default: 20)
- `--categories TEXT`: Filter by categories (comma-separated)
- `--export, -e PATH`: Export insights to JSON file
- `--verbose, -v`: Show detailed information

## Output Format

The command displays insights in a visually rich format:

```
⚡ Discovered 4 insights

╭─ 💡 Insight #1 [Spike: 0.40] ────────────────────────────────────────────────╮
│ Description: Recurring concept 'attention' found across 2 knowledge items    │
│ Confidence: 32%                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

╭─ 💡 Insight #2 [Spike: 0.40] ────────────────────────────────────────────────╮
│ Description: Connection discovered between concepts: networks, neural        │
│ Confidence: 28%                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

🌉 Bridge Concepts:
┏━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Concept   ┃ Frequency ┃ Bridge Score ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ neural    │         2 │         0.40 │
│ networks  │         2 │         0.40 │
│ attention │         2 │         0.40 │
└───────────┴───────────┴──────────────┘
```

## Insight Types

1. **reasoning_spike**: High-quality reasoning chains that reveal insights
2. **recurring_concept**: Concepts that appear frequently across multiple documents
3. **concept_bridge**: Connections discovered between related concepts
4. **structural_pattern**: Patterns in the knowledge graph structure (planned)
5. **causal**: Cause-and-effect relationships (planned)
6. **analogical**: Similarities between different domains (planned)

## Examples

### Discovering Research Insights

```bash
# Load research papers and find insights
spike embed research-papers/
spike discover --corpus research-papers/ --min-spike 0.8

# Output:
# ⚡ Discovered connection between "quantum entanglement" and "neural synchronization"
# Bridge concepts: information transfer, non-locality, coherence
```

### Analyzing Technical Documentation

```bash
# Find patterns in your codebase documentation
spike discover --corpus docs/ --categories "structural" --export code-insights.json
```

### Cross-Domain Discovery

```bash
# Find connections between different fields
spike embed biology-papers/
spike embed physics-papers/
spike discover --min-spike 0.7 --verbose

# Might discover: "Protein folding exhibits phase transition behavior similar to magnetic materials"
```

## How It Works

1. **Graph Construction**: Builds a knowledge graph from your documents
2. **Pattern Analysis**: Analyzes graph structure for unusual patterns
3. **Spike Detection**: Calculates geDIG metrics to identify insight spikes
4. **Bridge Finding**: Identifies concepts that connect disparate areas
5. **Ranking**: Sorts insights by spike value and confidence

## Tips

- **Lower thresholds** (0.5-0.7) for exploratory discovery
- **Higher thresholds** (0.8-0.95) for high-confidence insights only
- **Use categories** to focus on specific types of insights
- **Export results** for further analysis or sharing

## Integration with Other Commands

```bash
# Discover insights, then explore specific connections
spike discover --export insights.json
spike bridge "concept1" "concept2"  # Explore a discovered connection

# Visualize the insight network
spike graph visualize --insights-only
```

## Understanding Spike Values

- **0.9-1.0**: 🔥 Exceptional insights with strong evidence
- **0.8-0.9**: ⚡ High-value insights worth investigating
- **0.7-0.8**: 💡 Interesting patterns that may lead to insights
- **0.5-0.7**: 🔍 Potential connections requiring more exploration

## Troubleshooting

If no insights are found:
1. Add more documents to your knowledge base
2. Lower the minimum spike threshold
3. Ensure documents contain related concepts
4. Try without category filters

## Future Enhancements

- Real-time insight monitoring
- Collaborative insight validation
- Integration with external knowledge bases
- Machine learning-based insight prediction

---

# spike experiment - Run Experiments

## Overview

The `spike experiment` command runs predefined experiments to test InsightSpike's capabilities.

## Usage

```bash
# Run default experiment
spike experiment

# Run specific experiment type
spike experiment --name simple
spike experiment --name insight
spike experiment --name math

# With custom parameters
spike experiment --cycles 20 --verbose
```

## Options

- `--name`: Experiment type (simple, insight, math)
- `--episodes`: Number of episodes to run (default: 5)
- `--preset`: Config preset to use

---

# spike demo - Quick Demo

## Overview

The `spike demo` command runs a quick demonstration of InsightSpike's capabilities.

## Usage

```bash
# Run demo
spike demo

# Run specific demo
spike demo --type insight
```

## Demo Types

- `insight` - Insight detection demo
- `graph` - Knowledge graph demo
- `chat` - Interactive chat demo
- `full` - Complete feature demo

---

# Utility Commands

---

# spike config - Configuration Management

## Overview

The `spike config` command manages InsightSpike configuration settings using the Pydantic-based configuration system.

## Usage

```bash
# Show current config
spike config show

# Set a config value (nested keys supported)
spike config set core.temperature 0.5
spike config set memory.episodic_memory_capacity 1000

# Save config to file
spike config save my_config.yaml

# Load config from file  
spike config load my_config.yaml

# Use a preset
spike config preset development
spike config preset production
spike config preset experiment

# Validate configuration
spike config validate

# Export full config with defaults
spike config export
```

## Actions

- `show` - Display current configuration as JSON
- `set <key> <value>` - Set a configuration value (supports nested keys)
- `save <file>` - Save configuration to YAML file
- `load <file>` - Load configuration from YAML file
- `preset <name>` - Load a configuration preset (development, production, experiment, testing, cloud)
- `validate` - Validate current configuration
- `export` - Export full configuration including defaults

## Configuration Keys

Common configuration keys:
- `core.llm_provider`: LLM provider (local, openai, anthropic, mock, clean)
- `core.temperature`: Response creativity (0.0-1.0)
- `core.max_tokens`: Maximum response tokens
- `memory.episodic_memory_capacity`: Number of episodes to retain
- `memory.enable_aging`: Enable time-based memory decay
- `graph.spike_ged_threshold`: Threshold for spike detection
- `retrieval.top_k`: Number of documents to retrieve

---

# spike stats - View Statistics

## Overview

The `spike stats` command displays statistics about your knowledge base and usage.

## Usage

```bash
# Show all stats
spike stats

# Show specific category
spike stats --category memory
```

## Statistics Categories

- `memory` - Memory usage and episode count
- `insights` - Insight discovery statistics
- `performance` - Performance metrics
- `usage` - Usage statistics

---

# spike version - Version Information

## Overview

The `spike version` command displays version and system information.

## Usage

```bash
# Show version
spike version

# Show detailed info
spike version --verbose
```

---

# Coming Soon

---

# spike bridge - Concept Bridging

## Overview

The `spike bridge` command finds conceptual paths between seemingly unrelated ideas, revealing hidden connections through intermediate concepts.

## Features

- 🌉 **Semantic Pathfinding**: Finds connections between distant concepts
- 🔗 **Multi-hop Analysis**: Discovers intermediate bridge concepts
- 📊 **Path Confidence**: Scores connection strength
- 🎯 **Visualization**: Shows the conceptual path graphically

## Usage

```bash
# Find bridges between concepts
spike bridge "machine learning" "biological evolution"

# With visualization
spike bridge "quantum computing" "consciousness" --visualize

# Limit path length
spike bridge "concept1" "concept2" --max-hops 3
```

## Options

- `--max-hops`: Maximum intermediate concepts (default: 4)
- `--visualize`: Show visual representation
- `--min-confidence`: Minimum path confidence
- `--export`: Export results to file

---

# spike graph - Knowledge Graph Analytics

## Overview

The `spike graph` command provides analytics and visualization of your knowledge structure.

## Features

- 📊 **Graph Metrics**: Calculate structural properties
- 🎯 **Centrality Analysis**: Find key concepts
- 🔍 **Cluster Detection**: Identify knowledge domains
- 🎨 **Visualization**: Export interactive graph views

## Usage

```bash
# Analyze graph structure
spike graph analyze

# Visualize knowledge graph
spike graph visualize

# Export visualization
spike graph visualize --output graph.html

# Show specific metrics
spike graph analyze --metrics centrality,clustering
```

## Subcommands

- `analyze`: Compute graph metrics and statistics
- `visualize`: Create visual representation of knowledge graph

## Options

### For analyze:
- `--metrics`: Specific metrics to calculate (all, centrality, clustering, density)
- `--export`: Export results to JSON

### For visualize:
- `--output`: Output file path (HTML format)
- `--layout`: Graph layout algorithm
- `--filter`: Filter nodes by criteria
