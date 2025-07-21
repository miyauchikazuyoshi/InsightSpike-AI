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

### Coming Soon
11. [spike bridge - Concept Bridging](#spike-bridge---concept-bridging) (Coming Soon)
12. [spike graph - Knowledge Graph Analytics](#spike-graph---knowledge-graph-analytics) (Coming Soon)

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

# Aliases
spike ask "What is the connection between consciousness and information?"
spike q "Explain quantum entanglement"

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

# Aliases
spike learn papers/
spike l research.md
spike e data.txt
```

## Options

- `--preset, -p`: Config preset for processing
- `--recursive, -r`: Process directories recursively
- `--format, -f`: Specify file format (auto-detected by default)

## Supported Formats

- `.txt` - Plain text files
- `.md` - Markdown files
- `.pdf` - PDF documents (with optional dependencies)
- `.json` - Structured data

---

# spike interactive - Interactive Chat

## Overview

The `spike interactive` command (alias: `spike chat`) provides an interactive conversation mode with persistent context.

## Usage

```bash
# Start interactive mode
spike interactive

# Alias
spike chat

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

- `--limit, -l`: Maximum number of insights to show
- `--since, -s`: Show insights since date
- `--min-confidence`: Minimum confidence threshold
- `--category, -c`: Filter by insight category
- `--export, -e`: Export to file

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

# Run specific experiment
spike experiment --name "english_insight"

# With custom parameters
spike experiment --cycles 20 --verbose
```

## Options

- `--name, -n`: Experiment name
- `--cycles`: Maximum reasoning cycles
- `--output, -o`: Output directory for results
- `--verbose, -v`: Show detailed progress

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

The `spike config` command manages InsightSpike configuration settings.

## Usage

```bash
# Show current config
spike config show

# Set a config value
spike config set llm.model "gpt-4"

# Save config to file
spike config save my_config.yaml

# Load config from file
spike config load my_config.yaml

# Use a preset
spike config preset production
```

## Actions

- `show` - Display current configuration
- `set <key> <value>` - Set a configuration value
- `save <file>` - Save configuration to file
- `load <file>` - Load configuration from file
- `preset <name>` - Load a configuration preset

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

# spike bridge - Concept Bridging (Coming Soon)

## Overview

The `spike bridge` command finds conceptual paths between seemingly unrelated ideas, revealing hidden connections through intermediate concepts.

## Planned Features

- Semantic pathfinding between concepts
- Multiple path discovery
- Bridge concept identification
- Path confidence scoring

## Usage (Planned)

```bash
# Find bridges between concepts
spike bridge "machine learning" "biological evolution"

# Multi-hop bridging
spike bridge "quantum computing" "consciousness" --max-hops 4
```

---

# spike graph - Knowledge Graph Analytics (Coming Soon)

## Overview

The `spike graph` command provides analytics and visualization of your knowledge structure.

## Planned Features

- Graph metrics calculation
- Centrality analysis
- Cluster detection
- Interactive visualization export

## Usage (Planned)

```bash
# Analyze graph structure
spike graph analyze --metrics all

# Visualize knowledge graph
spike graph visualize --output graph.html
```