# InsightSpike CLI Commands

This document describes the unique CLI commands that showcase InsightSpike's core capabilities.

## Table of Contents

1. [spike discover - Insight Discovery](#spike-discover---insight-discovery)
2. [spike bridge - Concept Bridging](#spike-bridge---concept-bridging) (Coming Soon)
3. [spike graph - Knowledge Graph Analytics](#spike-graph---knowledge-graph-analytics) (Coming Soon)

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